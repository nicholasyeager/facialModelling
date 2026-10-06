import cv2
import numpy as np
import pytest

from facial_fire.mapping import ANCHORS, CANONICAL_ANCHORS, FACE_OVAL, FaceMapping
from facial_fire.rendering import blend_effect, make_preview


def landmarks():
    points = np.zeros((478, 2), np.float32)
    angle = np.linspace(-np.pi / 2, 3 * np.pi / 2, len(FACE_OVAL), endpoint=False)
    points[FACE_OVAL] = np.column_stack((100 + 65 * np.cos(angle), 105 + 85 * np.sin(angle)))
    points[ANCHORS] = [[65, 75], [135, 75], [100, 190]]
    return points


def test_anchor_mapping_and_roundtrip():
    points = landmarks()
    mapping = FaceMapping.from_landmarks(points, (240, 240, 3))
    assert mapping is not None
    projected = cv2.transform(CANONICAL_ANCHORS[None], mapping.to_frame)[0]
    np.testing.assert_allclose(projected, points[ANCHORS], atol=1e-4)
    for uv in ([0.4, 0.5], [0.7, 0.8]):
        xy = mapping.to_frame @ np.array([*uv, 1])
        np.testing.assert_allclose(mapping.canonical_point(*xy), uv, atol=1e-6)


def test_overlay_follows_rotation_scale_and_translation():
    points = landmarks()
    mapping = FaceMapping.from_landmarks(points, (300, 300, 3))
    motion = cv2.getRotationMatrix2D((100, 100), 18, 0.8)
    motion[:, 2] += [30, 15]
    moved = cv2.transform(points[None], motion)[0]
    moved_mapping = FaceMapping.from_landmarks(moved, (300, 300, 3))
    uv = [0.5, 0.55, 1]
    original_xy = mapping.to_frame @ uv
    expected_xy = motion @ np.array([*original_xy, 1])
    np.testing.assert_allclose(moved_mapping.to_frame @ uv, expected_xy, atol=1e-4)
    frame = np.zeros((300, 300, 3), np.uint8)
    rendered = blend_effect(frame, make_preview(128), moved_mapping)
    y, x = np.unravel_index(rendered[..., 2].argmax(), frame.shape[:2])
    np.testing.assert_allclose([x, y], expected_xy, atol=2)


def test_effect_confined_even_with_texture_at_face_boundary():
    frame = np.random.default_rng(123).integers(0, 256, (240, 240, 3), dtype=np.uint8)
    mapping = FaceMapping.from_landmarks(landmarks(), frame.shape)
    before = frame.copy()
    rendered = blend_effect(frame, np.ones((32, 32), np.float32), mapping)
    np.testing.assert_array_equal(rendered[mapping.mask == 0], frame[mapping.mask == 0])
    assert np.any(rendered[mapping.mask != 0] != frame[mapping.mask != 0])
    np.testing.assert_array_equal(frame, before)


def test_empty_intensity_preserves_frame():
    frame = np.full((240, 240, 3), 113, np.uint8)
    mapping = FaceMapping.from_landmarks(landmarks(), frame.shape)
    np.testing.assert_array_equal(blend_effect(frame, np.zeros((16, 16), np.float32), mapping), frame)


@pytest.mark.parametrize("kind", ["degenerate", "nan", "short"])
def test_invalid_tracking_geometry_is_rejected(kind):
    points = landmarks()
    if kind == "degenerate":
        points[ANCHORS] = 0
    elif kind == "nan":
        points[0, 0] = np.nan
    else:
        points = points[:10]
    assert FaceMapping.from_landmarks(points, (240, 240, 3)) is None


def test_face_partly_outside_frame_is_safely_clipped():
    mapping = FaceMapping.from_landmarks(landmarks() - [80, 40], (100, 100, 3))
    assert mapping is not None
    assert mapping.mask.shape == (100, 100)
    rendered = blend_effect(np.zeros((100, 100, 3), np.uint8), np.ones((16, 16), np.float32), mapping)
    assert not rendered[mapping.mask == 0].any()
