import numpy as np
import pytest

from facial_fire.gestures import SnapDetector
from facial_fire.mapping import FaceMapping
from facial_fire.rendering import EFFECT_COLORS, blend_effect


def snap_hand(gap, offset=(100, 100), scale=1, mirrored=False):
    hand = np.zeros((21, 2), dtype=float)
    hand[0] = (0, 80)
    hand[5], hand[9], hand[17] = (-40, 0), (0, 0), (40, 0)
    hand[4], hand[12] = (0, -30), (80 * gap, -30)
    if mirrored:
        hand[:, 0] *= -1
    return hand * scale + offset


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("scale", [0.5, 1, 2])
def test_snap_on_either_hand_at_different_sizes(mirrored, scale):
    detector = SnapDetector()
    def update(gap, now):
        return detector.update([snap_hand(gap, scale=scale, mirrored=mirrored)], now)
    assert update(0.9, 0) == 0
    assert update(0.1, 0.02) == 0
    assert update(0.1, 0.07) == 0
    assert update(0.9, 0.1) == 1
    assert update(0.9, 0.15) == 0


def test_slow_release_brief_pinch_and_open_hand_do_not_snap():
    for sequence in (
        [(0, 0.1), (0.05, 0.1), (0.15, 0.3), (0.25, 0.45), (0.3, 0.7)],
        [(0, 0.1), (0.02, 0.9)],
        [(0, 0.8), (0.05, 0.9), (0.1, 1.1)],
        [(0, 0.1), (0.05, 0.1), (0.15, 0.3), (0.26, 0.5), (0.35, 0.9)],
    ):
        detector = SnapDetector()
        assert not any(detector.update([snap_hand(gap)], now) for now, gap in sequence)


def test_cooldown_requires_fresh_pinch_and_allows_later_snap():
    detector = SnapDetector()
    events = []
    for now, gap in [(0, .1), (.05, .1), (.1, .9), (.15, .1), (.2, .1), (.25, .9),
                     (.4, .9), (.51, .1), (.56, .1), (.6, .9)]:
        events.append(detector.update([snap_hand(gap)], now))
    assert events == [0, 0, 1, 0, 0, 0, 0, 0, 0, 1]


def test_hand_order_changes_do_not_mix_histories():
    detector = SnapDetector()
    a, b = (100, 100), (500, 100)
    detector.update([snap_hand(.1, a), snap_hand(.9, b)], 0)
    detector.update([snap_hand(.9, b), snap_hand(.1, a)], .05)
    assert detector.update([snap_hand(.9, b), snap_hand(.9, a)], .1) == 1


def test_simultaneous_hands_produce_two_detector_events():
    detector = SnapDetector()
    for now in (0, .05):
        assert detector.update([snap_hand(.1), snap_hand(.1, (500, 100))], now) == 0
    assert detector.update([snap_hand(.9), snap_hand(.9, (500, 100))], .1) == 2


@pytest.mark.parametrize("interrupt", ["loss", "stall", "jump", "crossing", "reset"])
def test_interrupted_history_cannot_trigger_release(interrupt):
    detector = SnapDetector()
    detector.update([snap_hand(.1)], 0)
    detector.update([snap_hand(.1)], .05)
    if interrupt == "loss":
        detector.update([], .06)
    if interrupt == "reset":
        detector.reset()
    hands = [snap_hand(.9, (800, 100) if interrupt == "jump" else (100, 100))]
    if interrupt == "crossing":
        hands.append(snap_hand(.9, (110, 100)))
    assert detector.update(hands, .4 if interrupt == "stall" else .1) == 0


@pytest.mark.parametrize("fps", [20, 30, 60])
def test_time_based_detection_across_frame_rates(fps):
    detector = SnapDetector()
    events = [detector.update([snap_hand(.1 if frame / fps < .12 else .9)], frame / fps)
              for frame in range(int(fps * .4))]
    assert sum(events) == 1


def test_invalid_geometry_and_clock():
    detector = SnapDetector()
    bad = snap_hand(.1)
    bad[4] = np.nan
    assert detector.update([bad, np.zeros((21, 2)), np.zeros((20, 2))], 1) == 0
    for now in (0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            detector.update([], now)


@pytest.mark.parametrize("kwargs", [{"close": .8}, {"release": .1}, {"min_speed": 0},
                                   {"close": float("nan")}, {"cooldown": -1}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        SnapDetector(**kwargs)


def test_tint_changes_only_rendering_and_stays_face_confined():
    frame = np.full((30, 30, 3), 100, dtype=np.uint8)
    grid = np.ones((16, 16), dtype=np.float32)
    before = grid.copy()
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:20, 5:20] = 255
    transform = np.array([[25, 0, 0], [0, 25, 0]], dtype=float)
    mapping = FaceMapping(transform, np.array([[.04, 0, 0], [0, .04, 0]]), mask)
    orange = blend_effect(frame, grid, mapping)
    blue = blend_effect(frame, grid, mapping, tint_bgr=EFFECT_COLORS[1][1])
    assert not np.array_equal(orange[10, 10], blue[10, 10])
    for rendered in (orange, blue):
        np.testing.assert_array_equal(rendered[mask == 0], frame[mask == 0])
    np.testing.assert_array_equal(grid, before)


@pytest.mark.parametrize("tint", [(1, 2), (0, 0, 256), (0, float("nan"), 1)])
def test_invalid_tint(tint):
    mapping = FaceMapping(np.eye(2, 3), np.eye(2, 3), np.ones((5, 5), np.uint8))
    with pytest.raises(ValueError, match="Tint"):
        blend_effect(np.zeros((5, 5, 3), np.uint8), np.ones((2, 2)), mapping, tint_bgr=tint)
