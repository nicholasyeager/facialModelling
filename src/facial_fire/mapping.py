"""Canonical [0, 1] coordinates anchored to outer eyes and chin.

An affine map handles translation, scale and in-plane rotation. It is an
intentional moderate-pose baseline, not a 3D surface reconstruction.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Ordered MediaPipe face oval; a convex hull would include extra background.
FACE_OVAL = np.array([
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
])
ANCHORS = [33, 263, 152]
CANONICAL_ANCHORS = np.array([[0.25, 0.35], [0.75, 0.35], [0.5, 0.9]], np.float32)


@dataclass(frozen=True)
class FaceMapping:
    to_frame: np.ndarray
    to_canonical: np.ndarray
    mask: np.ndarray

    @classmethod
    def from_landmarks(cls, points: np.ndarray, shape: tuple) -> "FaceMapping | None":
        if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2:
            return None
        if not np.isfinite(points).all():
            return None
        anchors = points[ANCHORS].astype(np.float32)
        eye_axis, chin_axis = anchors[1] - anchors[0], anchors[2] - anchors[0]
        area = eye_axis[0] * chin_axis[1] - eye_axis[1] * chin_axis[0]
        if np.linalg.norm(eye_axis) < 8 or abs(area) < 32:
            return None
        transform = cv2.getAffineTransform(CANONICAL_ANCHORS, anchors)
        mask = np.zeros(shape[:2], dtype=np.uint8)
        # Bound extreme model coordinates before converting to integer pixels.
        bound = 4 * max(shape[:2])
        oval = np.rint(np.clip(points[FACE_OVAL], -bound, bound)).astype(np.int32)
        cv2.fillPoly(mask, [oval], 255)
        return cls(transform, cv2.invertAffineTransform(transform), mask)

    def canonical_point(self, x: float, y: float) -> np.ndarray:
        return self.to_canonical @ np.array([x, y, 1.0])
