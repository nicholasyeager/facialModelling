"""Bitwise equivalence across execution modes and actual grid transitions."""

import numpy as np
import pytest

from facial_fire._native import Simulation, openmp_available
from facial_fire.simulation import Propagation


@pytest.mark.skipif(not openmp_available, reason="Build has no OpenMP support")
@pytest.mark.parametrize("mode", ["perimeter", "smooth"])
@pytest.mark.parametrize("shape", [(2, 2), (3, 7), (128, 128)])
@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_parallel_is_bitwise_equal_at_each_tick(mode, shape, threads):
    serial = Simulation(*shape, mode=mode, seed=123)
    parallel = Simulation(*shape, mode=mode, seed=123)
    # Includes cold, frontier, active and saturated cells, including boundaries.
    initial = np.random.default_rng(77).random(shape, dtype=np.float32)
    initial[initial < 0.2] = 0
    initial[initial > 0.9] = 1
    for kernel in (serial, parallel):
        kernel.set_grid(initial)
    parallel.set_execution(True, threads)
    for _ in range(40):
        serial.step(1 / 60, 12, 0.4)
        parallel.step(1 / 60, 12, 0.4)
        np.testing.assert_array_equal(parallel.snapshot(), serial.snapshot())
    assert serial.last_threads == 1
    assert 1 <= parallel.last_threads <= threads


@pytest.mark.skipif(not openmp_available, reason="Build has no OpenMP support")
@pytest.mark.parametrize("mode", ["perimeter", "smooth"])
def test_switching_execution_and_threads_preserves_grid_and_random_sequence(mode):
    reference = Simulation(41, 41, mode=mode, seed=42)
    switched = Simulation(41, 41, mode=mode, seed=42)
    for kernel in (reference, switched):
        kernel.ignite(0, 0, radius=0.04)
        kernel.ignite(0.3, 0.7, radius=0.06)
        kernel.ignite(1, 1, radius=0.04)
    for parallel, threads, steps in [(True, 4, 21), (False, 2, 19), (True, 8, 37)]:
        before = switched.snapshot()
        switched.set_execution(parallel, threads)
        np.testing.assert_array_equal(switched.snapshot(), before)
        switched.step(1 / 60, 12, 0.4, steps=steps)
        for _ in range(steps):
            reference.step(1 / 60, 12, 0.4)
        np.testing.assert_array_equal(switched.snapshot(), reference.snapshot())
    for kernel in (reference, switched):
        kernel.reset()
        kernel.step(1 / 60, 12, 0.4, steps=3)
        assert not kernel.snapshot().any()
        kernel.ignite(0.5, 0.5)
        kernel.step(1 / 60, 12, 0.4, steps=25)
    np.testing.assert_array_equal(switched.snapshot(), reference.snapshot())


@pytest.mark.skipif(not openmp_available, reason="Build has no OpenMP support")
def test_parallel_controller_clock_pause_loss_and_switch():
    reference = Propagation(size=32, seed=123)
    parallel = Propagation(size=32, seed=123, execution="parallel", threads=4)
    for propagation in (reference, parallel):
        propagation.ignite(0.5, 0.5)
    for frame in range(31):
        assert parallel.advance(frame / 30, True) == reference.advance(frame / 30, True)
    np.testing.assert_array_equal(parallel.kernel.snapshot(), reference.kernel.snapshot())
    parallel.toggle_pause()
    frozen = parallel.kernel.snapshot()
    parallel.advance(1.1, True)
    parallel.toggle_execution()
    assert parallel.execution == "serial"
    np.testing.assert_array_equal(parallel.kernel.snapshot(), frozen)
    parallel.toggle_execution()
    parallel.advance(1.2, False)
    parallel.advance(3.2, True)
    assert not parallel.kernel.snapshot().any()  # Timeout clears even while paused.


@pytest.mark.parametrize("threads", [0, -1, 257])
def test_native_execution_validation_is_atomic(threads):
    kernel = Simulation(8, 8)
    kernel.ignite(0.5, 0.5)
    original = kernel.snapshot()
    with pytest.raises(ValueError):
        kernel.set_execution(False, threads)
    assert not kernel.parallel
    assert kernel.threads == 4
    np.testing.assert_array_equal(kernel.snapshot(), original)


@pytest.mark.parametrize("kwargs", [
    {"execution": "unknown"}, {"threads": 0}, {"threads": 257}, {"threads": 1.5},
])
def test_controller_execution_validation(kwargs):
    with pytest.raises(ValueError):
        Propagation(**kwargs)


def test_serial_only_build_rejects_parallel_instead_of_silent_fallback():
    if openmp_available:
        pytest.skip("Requires serial-only build")
    kernel = Simulation(8, 8)
    with pytest.raises(RuntimeError, match="OpenMP is unavailable"):
        kernel.set_execution(True, 2)
    assert not kernel.parallel
