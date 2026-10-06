"""Check native code against independently computed small-grid expectations."""

import numpy as np
import pytest

from facial_fire._native import Simulation


def reference_step(grid, dt, speed, cooling):
    padded = np.pad(grid, 1, mode="constant")
    neighbors = np.maximum.reduce([
        padded[:-2, 1:-1], padded[2:, 1:-1],
        padded[1:-1, :-2], padded[1:-1, 2:],
    ])
    return np.clip(grid + dt * (speed * neighbors * (1 - grid) - cooling * grid), 0, 1)


@pytest.mark.parametrize("shape", [(2, 2), (3, 7), (16, 13)])
def test_native_matches_reference_over_many_steps(shape):
    expected = np.random.default_rng(123).random(shape, dtype=np.float32)
    kernel = Simulation(*shape, mode="smooth")
    kernel.set_grid(expected)
    for _ in range(40):
        expected = reference_step(expected, np.float32(1 / 60), np.float32(12), np.float32(0.4))
        kernel.step(1 / 60, 12, 0.4)
    np.testing.assert_allclose(kernel.snapshot(), expected, atol=2e-6, rtol=2e-6)


def test_one_step_reads_only_previous_buffer_and_four_neighbors():
    kernel = Simulation(5, 5, mode="smooth")
    kernel.ignite(0.5, 0.5, radius=0)
    kernel.step(0.1, 2, 1)
    expected = np.zeros((5, 5), np.float32)
    expected[2, 2] = 0.9
    expected[1, 2] = expected[3, 2] = expected[2, 1] = expected[2, 3] = 0.2
    # Diagonals and cells two hops away remain zero (no in-place cascade).
    np.testing.assert_allclose(kernel.snapshot(), expected, atol=1e-7)


def test_corner_boundary_has_no_wraparound():
    kernel = Simulation(3, 4, mode="smooth")
    kernel.ignite(1, 0, radius=0)
    kernel.step(0.1, 2, 0)
    expected = np.zeros((3, 4), np.float32)
    expected[0, 3] = 1
    expected[0, 2] = expected[1, 3] = 0.2
    np.testing.assert_allclose(kernel.snapshot(), expected, atol=1e-7)


def test_repeatability_and_batch_steps_match_exactly():
    first, second = Simulation(32, 32), Simulation(32, 32)
    for kernel in (first, second):
        kernel.ignite(0.3, 0.7, radius=0.04)
    first.step(1 / 60, 12, 0.4, steps=100)
    for _ in range(100):
        second.step(1 / 60, 12, 0.4)
    np.testing.assert_array_equal(first.snapshot(), second.snapshot())


def test_reset_clears_both_buffers_and_reignition_is_repeatable():
    kernel = Simulation(16, 16)
    kernel.ignite(0.25, 0.75)
    initial = kernel.snapshot()
    kernel.step(1 / 60, 12, 0.4, steps=7)  # Odd swaps exercise both buffers.
    kernel.reset()
    kernel.step(1 / 60, 12, 0.4, steps=3)
    assert not kernel.snapshot().any()
    kernel.ignite(0.25, 0.75)
    np.testing.assert_array_equal(kernel.snapshot(), initial)


def test_zero_spread_cools_and_zero_step_is_noop():
    kernel = Simulation(3, 3)
    kernel.ignite(0.5, 0.5, radius=0)
    original = kernel.snapshot()
    kernel.step(0.1, 0, 1, steps=0)
    np.testing.assert_array_equal(kernel.snapshot(), original)
    kernel.step(0.1, 0, 1)
    np.testing.assert_allclose(kernel.snapshot(), original * 0.9, atol=1e-7)


def test_snapshot_and_set_grid_do_not_alias_native_state():
    kernel = Simulation(3, 4)
    source = np.full((3, 4), 0.3, np.float64)[:, ::-1]  # Noncontiguous conversion.
    kernel.set_grid(source)
    source[:] = 0
    snapshot = kernel.snapshot()
    snapshot[:] = 1
    np.testing.assert_allclose(kernel.snapshot(), 0.3)
    kernel.reset()
    assert (snapshot == 1).all()


def test_intensities_stay_finite_and_bounded():
    kernel = Simulation(8, 8)
    kernel.ignite(0.5, 0.5)
    kernel.step(1, 60, 60, steps=50)  # Extreme valid parameters exercise clipping.
    grid = kernel.snapshot()
    assert np.isfinite(grid).all()
    assert ((grid >= 0) & (grid <= 1)).all()


@pytest.mark.parametrize("shape", [(1, 4), (4, 1), (-1, 4), (2049, 2)])
def test_invalid_dimensions(shape):
    with pytest.raises(ValueError):
        Simulation(*shape)


@pytest.mark.parametrize("kwargs", [
    {"dt": float("nan")}, {"dt": -1}, {"dt": 2},
    {"spread_speed": float("inf")}, {"spread_speed": -1}, {"spread_speed": 61},
    {"cooling": -1}, {"steps": -1},
])
def test_invalid_step_is_rejected_without_mutation(kwargs):
    kernel = Simulation(3, 3)
    kernel.ignite(0.5, 0.5)
    original = kernel.snapshot()
    parameters = dict(dt=1 / 60, spread_speed=12, cooling=0.4, steps=1)
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        kernel.step(**parameters)
    np.testing.assert_array_equal(kernel.snapshot(), original)


@pytest.mark.parametrize("u,v", [(-0.1, 0.5), (1.1, 0.5), (0.5, float("nan"))])
def test_invalid_ignition_coordinates(u, v):
    kernel = Simulation(3, 3)
    with pytest.raises(ValueError):
        kernel.ignite(u, v)
    assert not kernel.snapshot().any()


@pytest.mark.parametrize("value", [-1, 2, float("nan")])
def test_invalid_grid_is_atomic(value):
    kernel = Simulation(3, 3)
    kernel.ignite(0.5, 0.5)
    original = kernel.snapshot()
    invalid = original.copy()
    invalid[-1, -1] = value
    with pytest.raises(ValueError):
        kernel.set_grid(invalid)
    np.testing.assert_array_equal(kernel.snapshot(), original)
    with pytest.raises(ValueError):
        kernel.set_grid(np.zeros((2, 2)))
