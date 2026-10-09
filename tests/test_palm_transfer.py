import cv2
import numpy as np
import pytest

from facial_fire.interaction import FINGERTIPS
from facial_fire.mapping import FaceMapping
from facial_fire.rendering import draw_fingertip_fire
from facial_fire.simulation import Propagation
from facial_fire.transfer import PalmTransfer


def hand(x=100, y=250, scale=1, mirrored=False):
    points = np.zeros((21, 2), dtype=float)
    points[0], points[5], points[9], points[13], points[17] = (0, 40), (-30, -20), (0, -30), (15, -25), (30, -20)
    points[list(FINGERTIPS)] = [(-40, -100), (-30, -100), (0, -100), (20, -100), (40, -100)]
    if mirrored:
        points[:, 0] *= -1
    return points * scale + (x, y)


def face():
    transform = np.array([[100, 0, 20], [0, 100, 20]], dtype=float)
    mask = np.zeros((400, 400), np.uint8)
    mask[20:121, 20:121] = 255
    return FaceMapping(transform, cv2.invertAffineTransform(transform), mask)


def charged(detector=None):
    detector = detector or PalmTransfer()
    for now in (0, .1, .18):
        assert detector.update([hand(), hand(140)], None, now, labels=("Left", "Right")) == []
    assert detector.charged_count == 10
    return detector


def test_palm_charge_requires_dwell_and_two_hands():
    detector = PalmTransfer()
    for now in (0, .1, .2):
        detector.update([hand()], None, now)
        assert detector.charged_count == 0
    detector.update([hand(), hand(140)], None, .3)
    detector.update([hand(), hand(140)], None, .4)
    assert detector.charged_count == 0
    detector.update([hand(), hand(140)], None, .48)
    assert detector.charged_count == 10


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("scale", [.5, 1, 2])
def test_palm_connection_normalized_by_size(mirrored, scale):
    detector = PalmTransfer()
    hands = [hand(100, scale=scale, mirrored=mirrored), hand(100 + 40 * scale, scale=scale, mirrored=mirrored)]
    for now in (0, .1, .18):
        detector.update(hands, None, now)
    assert detector.charged_count == 10


def test_single_tip_transfer_consumes_only_that_tip_and_seeds_face():
    detector = charged()
    a, b = hand(), hand(220)
    a[8] = (70, 80)
    assert detector.update([a, b], face(), .2, labels=("Left", "Right")) == []
    events = detector.update([b, a], face(), .32, labels=("Right", "Left"))
    np.testing.assert_allclose(events, [[.5, .6]])
    assert detector.charged_count == 9
    assert len(detector.flames) == 9
    propagation = Propagation(size=32)
    propagation.ignite(*events[0])
    assert propagation.kernel.snapshot()[19, 16] == 1
    assert detector.update([a, b], face(), .44, labels=("Left", "Right")) == []
    a[8] = (150, 150)
    detector.update([a, b], face(), .5, labels=("Left", "Right"))
    a[8] = (70, 80)
    detector.update([a, b], face(), .55, labels=("Left", "Right"))
    assert detector.update([a, b], face(), .68, labels=("Left", "Right")) == []
    assert detector.charged_count == 9


def test_reconnecting_palms_recharges_and_connection_blocks_transfers():
    detector = charged(PalmTransfer(dwell=0))
    a = hand()
    a[8] = (70, 80)
    detector.update([a, hand(220)], face(), .2, labels=("Left", "Right"))
    assert detector.charged_count == 9
    for now in (.25, .35, .43, .5):
        assert detector.update([a, hand(140)], face(), now, labels=("Left", "Right")) == []
    assert detector.charged_count == 10
    assert len(detector.update([a, hand(220)], face(), .55, labels=("Left", "Right"))) == 1
    assert detector.charged_count == 9


def test_only_charged_fingertips_transfer_and_multiple_tips_consume_individually():
    a, b = hand(), hand(220)
    a[4], a[8] = (50, 60), (70, 80)
    empty = PalmTransfer(dwell=0)
    assert empty.update([a, b], face(), 0) == []
    detector = charged(PalmTransfer(dwell=0))
    assert len(detector.update([a, b], face(), .2, labels=("Left", "Right"))) == 2
    assert detector.charged_count == 8


def test_pause_and_face_loss_preserve_charges_but_restart_transfer_dwell():
    detector = charged()
    a, b = hand(), hand(220)
    a[8] = (70, 80)
    detector.update([a, b], face(), .2, labels=("Left", "Right"))
    assert detector.update([a, b], face(), .3, paused=True, labels=("Left", "Right")) == []
    assert detector.charged_count == 10
    detector.update([a, b], face(), .4, labels=("Left", "Right"))
    detector.update([a, b], None, .5, labels=("Left", "Right"))
    assert detector.update([a, b], face(), .6, labels=("Left", "Right")) == []
    assert len(detector.update([a, b], face(), .72, labels=("Left", "Right"))) == 1


def test_lost_hand_drops_only_its_charge_and_reacquisition_is_empty():
    detector = charged()
    detector.update([hand()], None, .2, labels=("Left",))
    assert detector.charged_count == 5
    detector.update([hand(), hand(220)], None, .3, labels=("Left", "Right"))
    assert detector.charged_count == 5


@pytest.mark.parametrize("kind", ["stall", "reset", "motion", "label_change"])
def test_uncertain_history_loses_charges(kind):
    detector = charged()
    if kind == "reset":
        detector.reset()
    x = 700 if kind == "motion" else 100
    labels = ("Right",) if kind == "label_change" else ("Left",)
    detector.update([hand(x)], None, .6 if kind == "stall" else .2, labels=labels)
    assert detector.charged_count == 0


def test_ambiguous_crossing_does_not_assign_consumable_state():
    detector = PalmTransfer(palm_dwell=0)
    detector.update([hand(), hand()], None, 0)
    assert detector.charged_count == 10
    detector.update([hand(40), hand(160)], None, .1)
    assert detector.charged_count == 0


def test_invalid_points_outside_mask_and_uv_do_not_consume():
    detector = charged(PalmTransfer(dwell=0))
    a = hand()
    a[list(FINGERTIPS)] = [(-1, 80), (400, 80), (70, 80), (130, 80), (70, 150)]
    mapping = face()
    mapping.mask[80, 70] = 0
    assert detector.update([a, hand(220)], mapping, .2, labels=("Left", "Right")) == []
    assert detector.charged_count == 10
    mapping.mask[:] = 255
    assert len(detector.update([a, hand(220)], mapping, .3, labels=("Left", "Right"))) == 1
    assert detector.charged_count == 9


def test_invalid_geometry_and_clock():
    detector = PalmTransfer()
    bad = hand()
    bad[0] = np.nan
    detector.update([bad, np.zeros((20, 2))], None, 1)
    assert detector.charged_count == 0
    for now in (0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            detector.update([], None, now)


@pytest.mark.parametrize("kwargs", [{"dwell": -1}, {"palm_dwell": 3}, {"palm_distance": 0}, {"palm_distance": float("nan")}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        PalmTransfer(**kwargs)


def test_flame_rendering_clips_edges_and_empty_charges_draw_nothing():
    frame = np.full((80, 80, 3), 30, dtype=np.uint8)
    before = frame.copy()
    draw_fingertip_fire(frame, [], (20, 90, 255), 0)
    np.testing.assert_array_equal(frame, before)
    draw_fingertip_fire(frame, [(np.array([40, 40]), 65), (np.array([0, 0]), 65),
                              (np.array([-1, 20]), 65), (np.array([np.nan, 1]), 65)], (20, 90, 255), 0)
    assert np.any(frame[30:45, 35:45] != before[30:45, 35:45])
    np.testing.assert_array_equal(frame[60:], before[60:])
