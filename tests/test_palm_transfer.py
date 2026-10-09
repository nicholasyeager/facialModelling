import cv2
import numpy as np
import pytest

from facial_fire.interaction import FINGERTIPS
from facial_fire.mapping import FaceMapping
from facial_fire.rendering import draw_fingertip_fire
from facial_fire.simulation import Propagation
from facial_fire.transfer import FingertipTransfer


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


def connecting_hand():
    points = hand(140)
    points[list(FINGERTIPS)] = hand()[list(FINGERTIPS)] + (8, 0)
    return points


def charged(detector=None):
    detector = detector or FingertipTransfer()
    for now in (0, .1, .18):
        assert detector.update([hand(), connecting_hand()], None, now, labels=("Left", "Right")) == []
    assert detector.charged_count == 10
    return detector


def test_fingertip_charge_requires_dwell_and_two_hands():
    detector = FingertipTransfer()
    for now in (0, .1, .2):
        detector.update([hand()], None, now)
        assert detector.charged_count == 0
    detector.update([hand(), connecting_hand()], None, .3)
    detector.update([hand(), connecting_hand()], None, .4)
    assert detector.charged_count == 0
    detector.update([hand(), connecting_hand()], None, .48)
    assert detector.charged_count == 10


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("scale", [.5, 1, 2])
def test_fingertip_connection_normalized_by_size(mirrored, scale):
    detector = FingertipTransfer()
    hands = [hand(100, scale=scale, mirrored=mirrored), hand(100 + 40 * scale, scale=scale, mirrored=mirrored)]
    hands[1][list(FINGERTIPS)] = hands[0][list(FINGERTIPS)] + (8 * scale, 0)
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


def test_reconnecting_fingertips_recharges_and_connection_blocks_transfers():
    detector = charged(FingertipTransfer(dwell=0))
    a = hand()
    a[8] = (70, 80)
    detector.update([a, hand(220)], face(), .2, labels=("Left", "Right"))
    assert detector.charged_count == 9
    for now in (.25, .35, .43, .5):
        assert detector.update([a, connecting_hand()], face(), now, labels=("Left", "Right")) == []
    assert detector.charged_count == 10
    assert len(detector.update([a, hand(220)], face(), .55, labels=("Left", "Right"))) == 1
    assert detector.charged_count == 9


def test_only_charged_fingertips_transfer_and_multiple_tips_consume_individually():
    a, b = hand(), hand(220)
    a[4], a[8] = (50, 60), (70, 80)
    empty = FingertipTransfer(dwell=0)
    assert empty.update([a, b], face(), 0) == []
    detector = charged(FingertipTransfer(dwell=0))
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
    detector = FingertipTransfer(charge_dwell=0)
    detector.update([hand(), hand()], None, 0)
    assert detector.charged_count == 10
    detector.update([hand(40), hand(160)], None, .1)
    assert detector.charged_count == 0


def test_invalid_points_outside_mask_and_uv_do_not_consume():
    detector = charged(FingertipTransfer(dwell=0))
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
    detector = FingertipTransfer()
    bad = hand()
    bad[0] = np.nan
    detector.update([bad, np.zeros((20, 2))], None, 1)
    assert detector.charged_count == 0
    for now in (0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            detector.update([], None, now)


@pytest.mark.parametrize("kwargs", [{"dwell": -1}, {"charge_dwell": 3}, {"tip_distance": 0}, {"tip_distance": float("nan")}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        FingertipTransfer(**kwargs)


def test_flame_rendering_clips_edges_and_empty_charges_draw_nothing():
    frame = np.full((80, 80, 3), 30, dtype=np.uint8)
    before = frame.copy()
    draw_fingertip_fire(frame, [], (20, 90, 255), 0)
    np.testing.assert_array_equal(frame, before)
    draw_fingertip_fire(frame, [(np.array([40, 40]), 65), (np.array([0, 0]), 65),
                              (np.array([-1, 20]), 65), (np.array([np.nan, 1]), 65)], (20, 90, 255), 0)
    assert np.any(frame[30:45, 35:45] != before[30:45, 35:45])
    np.testing.assert_array_equal(frame[60:], before[60:])


def test_close_palms_with_far_fingertips_do_not_charge():
    detector = FingertipTransfer(charge_dwell=0)
    a, b = hand(), hand(105)
    b[list(FINGERTIPS)] += (100, 0)
    detector.update([a, b], None, 0)
    assert detector.charged_count == 0


def test_separated_palms_with_corresponding_fingertips_close_charge():
    detector = FingertipTransfer(charge_dwell=0)
    a, b = hand(), hand(260)
    b[list(FINGERTIPS)] = a[list(FINGERTIPS)] + (8, 0)
    detector.update([a, b], None, 0)
    assert detector.charged_count == 10
    assert "T 0.12" in detector.lines[1]


@pytest.mark.parametrize("required", [1, 2, 3, 4, 5])
def test_configurable_corresponding_pair_count(required):
    detector = FingertipTransfer(charge_dwell=0, min_pairs=required)
    a, b = hand(), hand(220)
    for tip in FINGERTIPS[:required - 1]:
        b[tip] = a[tip] + (8, 0)
    detector.update([a, b], None, 0)
    assert detector.charged_count == 0
    b[FINGERTIPS[required - 1]] = a[FINGERTIPS[required - 1]] + (8, 0)
    detector.update([a, b], None, .1)
    assert detector.charged_count == 10


def test_noncorresponding_tips_do_not_count_as_contact():
    detector = FingertipTransfer(charge_dwell=0)
    a, b = hand(), hand(220)
    a[list(FINGERTIPS)] = [(0, 150), (100, 150), (200, 150), (300, 150), (400, 150)]
    b[list(FINGERTIPS)] = np.roll(a[list(FINGERTIPS)], 1, axis=0)
    detector.update([a, b], None, 0)
    assert detector.charged_count == 0


def test_fingertip_connection_release_hysteresis():
    detector = charged()
    a, b = hand(), connecting_hand()
    # Entry distance .35; .4 still holds an established connection.
    b[list(FINGERTIPS)] = a[list(FINGERTIPS)] + (26, 0)
    detector.update([a, b], None, .2, labels=("Left", "Right"))
    assert "separate to transfer" in detector.lines[0]
    b[list(FINGERTIPS)] = a[list(FINGERTIPS)] + (40, 0)
    detector.update([a, b], None, .3, labels=("Left", "Right"))
    assert "separate to transfer" not in detector.lines[0]


@pytest.mark.parametrize("pairs", [0, 6, 2.5, True])
def test_invalid_pair_count(pairs):
    with pytest.raises(ValueError):
        FingertipTransfer(min_pairs=pairs)
