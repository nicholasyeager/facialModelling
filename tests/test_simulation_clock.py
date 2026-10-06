import numpy as np
import pytest

from facial_fire.simulation import Propagation


def run_at_fps(fps):
    propagation = Propagation(size=16)
    propagation.ignite(0.5, 0.55)
    propagation.advance(0, tracked=True)
    ticks = sum(propagation.advance(frame / fps, tracked=True) for frame in range(1, fps + 1))
    return ticks, propagation.kernel.snapshot()


def test_same_elapsed_time_is_independent_of_frame_rate():
    slow_ticks, slow = run_at_fps(30)
    fast_ticks, fast = run_at_fps(144)
    assert slow_ticks == fast_ticks == 60
    np.testing.assert_array_equal(slow, fast)


def test_user_pause_freezes_and_does_not_catch_up():
    propagation = Propagation(size=16)
    propagation.ignite(0.5, 0.5)
    propagation.advance(0, True)
    propagation.advance(0.1, True)
    propagation.toggle_pause()
    original = propagation.kernel.snapshot()
    assert propagation.advance(10, True) == 0
    np.testing.assert_array_equal(propagation.kernel.snapshot(), original)
    propagation.toggle_pause()
    assert propagation.advance(20, True) == 0
    np.testing.assert_array_equal(propagation.kernel.snapshot(), original)
    assert propagation.advance(20 + 1 / 60, True) == 1


def test_short_tracking_loss_preserves_state_and_discards_lost_time():
    propagation = Propagation(size=16, loss_timeout=2)
    propagation.ignite(0.5, 0.5)
    propagation.advance(0, True)
    original = propagation.kernel.snapshot()
    assert propagation.advance(0.1, False) == 0
    assert propagation.advance(1, False) == 0
    assert propagation.advance(1.5, True) == 0
    np.testing.assert_array_equal(propagation.kernel.snapshot(), original)
    assert propagation.advance(1.5 + 1 / 60, True) == 1


@pytest.mark.parametrize("reacquire_directly", [False, True])
def test_timeout_clears_even_when_next_observation_is_reacquisition(reacquire_directly):
    propagation = Propagation(size=16, loss_timeout=2)
    propagation.ignite(0.5, 0.5)
    propagation.advance(0, True)
    propagation.advance(0.1, False)
    assert propagation.advance(2.1, reacquire_directly) == 0
    assert not propagation.kernel.snapshot().any()
    assert propagation.advance(2.2, True) == (6 if reacquire_directly else 0)
    assert not propagation.kernel.snapshot().any()


def test_pause_does_not_disable_tracking_timeout():
    propagation = Propagation(size=8, loss_timeout=0)
    propagation.ignite(0.5, 0.5)
    propagation.toggle_pause()
    propagation.advance(0, False)
    assert not propagation.kernel.snapshot().any()


def test_stalls_have_bounded_work_without_future_backlog():
    propagation = Propagation(size=8)
    propagation.advance(0, True)
    assert propagation.advance(10, True) == 15
    assert propagation.dropped_seconds == pytest.approx(9.75)
    assert propagation.advance(10 + 1 / 60, True) == 1


def test_reset_discards_fractional_tick_and_speed_controls_clamp():
    propagation = Propagation(size=8)
    propagation.advance(0, True)
    propagation.advance(0.01, True)
    propagation.reset()
    assert propagation.advance(0.02, True) == 0
    propagation.adjust_speed(-100)
    assert propagation.spread_speed == 0
    propagation.adjust_speed(100)
    assert propagation.spread_speed == 60
    with pytest.raises(ValueError):
        propagation.adjust_speed(float("nan"))


@pytest.mark.parametrize("kwargs", [
    {"spread_speed": -1}, {"spread_speed": float("nan")},
    {"cooling": float("inf")}, {"loss_timeout": -1},
    {"timestep": 0}, {"ignition_radius": -0.1},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Propagation(**kwargs)


def test_clock_rejects_nonfinite_or_backwards_values():
    propagation = Propagation(size=8)
    propagation.advance(1, True)
    for value in (0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            propagation.advance(value, True)
