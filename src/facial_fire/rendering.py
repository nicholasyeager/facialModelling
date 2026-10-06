"""Warp a canonical intensity texture, then clip alpha to the face mask."""

import cv2
import numpy as np

from .mapping import FaceMapping


def make_preview(size: int, center=(0.5, 0.55), radius=0.18) -> np.ndarray:
    if size < 2 or radius <= 0:
        raise ValueError("Preview requires size >= 2 and radius > 0")
    y, x = np.mgrid[0:size, 0:size] / (size - 1)
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return np.clip(1.0 - distance / radius, 0, 1).astype(np.float32)


def blend_effect(frame: np.ndarray, grid: np.ndarray, mapping: FaceMapping,
                 strength: float = 0.7) -> np.ndarray:
    if grid.ndim != 2 or min(grid.shape) < 2:
        raise ValueError("Grid must be a 2D array with dimensions >= 2")
    height, width = frame.shape[:2]
    # Grid pixel indices -> canonical UV -> camera pixel coordinates.
    grid_to_uv = np.diag([1 / (grid.shape[1] - 1), 1 / (grid.shape[0] - 1), 1])
    transform = mapping.to_frame @ grid_to_uv
    intensity = cv2.warpAffine(
        grid, transform, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    alpha = np.clip(intensity * strength, 0, 1)
    alpha[mapping.mask == 0] = 0  # After interpolation: no leakage outside oval.
    alpha = alpha[..., None]
    tint_bgr = np.array([20, 90, 255], np.float32)
    return np.rint(frame * (1 - alpha) + tint_bgr * alpha).astype(np.uint8)
