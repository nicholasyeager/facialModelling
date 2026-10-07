from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from facial_fire.interaction import FINGERTIPS, FingertipIgnition, face_fingertips
from facial_fire.mapping import FaceMapping
from facial_fire.simulation import Propagation
from facial_fire.tracking import HandTracker


def mapping():
    transform = np.array([[100, 0, 20], [0, 100, 20]], dtype=float)
    mask = np.zeros((160, 160), np.uint8)
    mask[20:121, 20:121] = 255
    return FaceMapping(transform, cv2.invertAffineTransform(transform), mask)


def hand_at(*tips):
    hand = np.full((21, 2), -10, dtype=np.float32)
    for index, tip in zip(FINGERTIPS, tips):
        hand[index] = tip
    return hand


def test_dwell_ignites_local_uv_once_then_release_rearms():
    detector = FingertipIgnition()
    face = mapping()
    hands = [hand_at((70, 80))]
    assert detector.update(hands, face, 0) == []
    assert detector.update(hands, face, 0.06) == []
    events = detector.update(hands, face, 0.12)
    np.testing.assert_allclose(events, [[0.5, 0.6]])
    propagation = Propagation(size=32)
    propagation.ignite(*events[0])
    assert propagation.kernel.snapshot()[19, 16] == 1
    assert detector.update(hands, face, 0.24) == []
    assert detector.update([], face, 0.25) == []
    assert detector.update(hands, face, 0.26) == []
    assert len(detector.update(hands, face, 0.38)) == 1


def test_multiple_regions_and_hand_order_changes():
    detector = FingertipIgnition()
    hands = [hand_at((40, 50)), hand_at((90, 100))]
    detector.update(hands, mapping(), 0)
    np.testing.assert_allclose(detector.update(hands[::-1], mapping(), 0.12), [[0.2, 0.3], [0.7, 0.8]])
    assert detector.update(hands, mapping(), 0.24) == []


def test_nearby_tips_and_jitter_do_not_repeat_ignition():
    detector = FingertipIgnition()
    face = mapping()
    detector.update([hand_at((70, 80), (71, 81))], face, 0)
    assert len(detector.update([hand_at((71, 80), (72, 81))], face, 0.12)) == 1
    assert detector.update([hand_at((72, 81))], face, 0.24) == []
    assert detector.update([hand_at((90, 80))], face, 0.25) == []
    np.testing.assert_allclose(detector.update([hand_at((90, 80))], face, 0.37), [[0.7, 0.6]])


@pytest.mark.parametrize("interrupt", ["face_loss", "hand_loss", "pause", "stall", "reset"])
def test_interruption_requires_fresh_dwell(interrupt):
    detector = FingertipIgnition()
    face, hands = mapping(), [hand_at((70, 80))]
    detector.update(hands, face, 0)
    if interrupt == "reset":
        detector.reset()
    elif interrupt == "stall":
        assert detector.update(hands, face, 0.4) == []
    else:
        assert detector.update([] if interrupt == "hand_loss" else hands,
                               None if interrupt == "face_loss" else face,
                               0.05, paused=interrupt == "pause") == []
    assert detector.update(hands, face, 0.41) == []
    assert len(detector.update(hands, face, 0.53)) == 1


def test_rejects_invalid_hands_outside_frame_mask_and_uv():
    face = mapping()
    face.mask[80, 70] = 0
    bad = hand_at((70, 80))
    bad[0] = np.nan
    assert face_fingertips([bad, np.zeros((20, 2)), hand_at((-1, 70), (160, 70), (70, 80), (130, 80))], face) == []
    face.mask[:] = 255
    assert face_fingertips([hand_at((130, 80))], face) == []  # Outside canonical square.


def test_rotated_scaled_mapping_produces_same_uv():
    transform = cv2.getRotationMatrix2D((0, 0), 30, 75)
    transform[:, 2] = [40, 90]
    face = FaceMapping(transform, cv2.invertAffineTransform(transform), np.full((200, 200), 255, np.uint8))
    xy = transform @ [0.5, 0.6, 1]
    np.testing.assert_allclose(face_fingertips([hand_at(xy)], face), [[0.5, 0.6]], atol=1e-6)


@pytest.mark.parametrize("dwell", [-1, 3, float("nan"), float("inf")])
def test_invalid_dwell(dwell):
    with pytest.raises(ValueError):
        FingertipIgnition(dwell=dwell)


def test_invalid_clock_and_zero_dwell():
    detector = FingertipIgnition(dwell=0)
    assert len(detector.update([hand_at((70, 80))], mapping(), 1)) == 1
    for now in (0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            detector.update([], mapping(), now)


def test_hand_tracker_pixel_conversion_timestamps_and_cleanup(monkeypatch):
    from facial_fire import tracking

    calls = []
    class FakeTracker:
        def detect_for_video(self, image, timestamp):
            calls.append(timestamp)
            return SimpleNamespace(hand_landmarks=[[SimpleNamespace(x=0.25, y=0.75)] * 21])

        def close(self):
            calls.append("closed")

    fake = FakeTracker()
    monkeypatch.setattr(tracking.vision.HandLandmarker, "create_from_options", lambda options: fake)
    model = Path("synthetic-hand.task")
    monkeypatch.setattr(Path, "is_file", lambda path: path == model)
    tracker = HandTracker(model)
    frame = np.zeros((100, 200, 3), np.uint8)
    for timestamp in (10, 10, 9):
        np.testing.assert_array_equal(tracker.detect(frame, timestamp)[0], np.tile([50, 75], (21, 1)))
    tracker.close()
    assert calls == [10, 11, 12, "closed"]
    with pytest.raises(FileNotFoundError, match="--kind hand"):
        HandTracker(Path("missing.task"))


@pytest.mark.parametrize("enabled", [False, True])
def test_app_optional_hand_ignition_and_resource_cleanup(monkeypatch, enabled):
    from facial_fire import app
    import sys

    calls = []
    instances = []
    frame = np.zeros((160, 160, 3), np.uint8)
    frame[:, 0] = 99

    class FakeFace:
        def __init__(self, model):
            calls.append("face opened")

        def detect(self, image, timestamp):
            assert (image[:, -1] == 99).all()  # Both trackers see mirrored capture.
            return np.zeros((478, 2))

        def close(self):
            calls.append("face closed")

    class FakeHand(FakeFace):
        def __init__(self, model):
            calls.append("hand opened")

        def detect(self, image, timestamp):
            assert (image[:, -1] == 99).all()
            return [hand_at((70, 80))]

        def close(self):
            calls.append("hand closed")

    class FakeCamera:
        def isOpened(self):
            return True

        def set(self, *args):
            pass

        def read(self):
            return True, frame.copy()

        def release(self):
            calls.append("camera released")

    class FakeWindow:
        def __init__(self, *args):
            pass

        def size(self, width, height):
            return width, height

    class ObservedPropagation(Propagation):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            instances.append(self)

    monkeypatch.setattr(sys, "argv", ["facial-fire", "--hand-dwell", "0"] + (["--hand-ignition"] if enabled else []))
    monkeypatch.setattr(app, "FaceTracker", FakeFace)
    monkeypatch.setattr(app, "HandTracker", FakeHand)
    monkeypatch.setattr(app, "Propagation", ObservedPropagation)
    monkeypatch.setattr(app.FaceMapping, "from_landmarks", lambda points, shape: mapping())
    monkeypatch.setattr(app.cv2, "VideoCapture", lambda index: FakeCamera())
    monkeypatch.setattr(app, "DemoWindow", FakeWindow)
    monkeypatch.setattr(app.cv2, "setMouseCallback", lambda *args: None)
    monkeypatch.setattr(app.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(app.cv2, "waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: calls.append("windows closed"))
    app.main()
    grid = instances[0].kernel.snapshot()
    assert bool(grid.any()) == enabled
    if enabled:
        assert grid[76, 64] == 1
        assert "hand closed" in calls
    else:
        assert "hand opened" not in calls
    assert "face closed" in calls and "camera released" in calls and "windows closed" in calls
