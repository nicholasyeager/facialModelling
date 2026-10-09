"""Warp a canonical intensity texture, then clip alpha to the face mask."""

import cv2
import math
import numpy as np

from .mapping import FaceMapping

# Named BGR tints; the original appearance is the first entry.
EFFECT_COLORS = (
    ("orange", (20, 90, 255)),
    ("blue", (255, 140, 30)),
    ("violet", (230, 60, 190)),
    ("green", (50, 220, 70)),
)


def make_preview(size: int, center=(0.5, 0.55), radius=0.18) -> np.ndarray:
    if size < 2 or radius <= 0:
        raise ValueError("Preview requires size >= 2 and radius > 0")
    y, x = np.mgrid[0:size, 0:size] / (size - 1)
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return np.clip(1.0 - distance / radius, 0, 1).astype(np.float32)


def blend_effect(frame: np.ndarray, grid: np.ndarray, mapping: FaceMapping,
                 strength: float = 0.7, tint_bgr=(20, 90, 255)) -> np.ndarray:
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
    tint_bgr = np.asarray(tint_bgr, dtype=np.float32)
    if tint_bgr.shape != (3,) or not np.isfinite(tint_bgr).all() or not ((tint_bgr >= 0) & (tint_bgr <= 255)).all():
        raise ValueError("Tint must contain three finite BGR values in [0, 255]")
    return np.rint(frame * (1 - alpha) + tint_bgr * alpha).astype(np.uint8)


def draw_fingertip_fire(frame, flames, tint_bgr, now):
    """Small animated flame markers on charged tips; no extra simulation grid."""
    height, width = frame.shape[:2]
    tint = np.asarray(tint_bgr, dtype=np.float32)
    for index, (point, scale) in enumerate(flames):
        if not np.isfinite(point).all() or not math.isfinite(scale):
            continue
        x, y = map(float, point)
        if not (0 <= x < width and 0 <= y < height):
            continue
        radius = int(np.clip(scale * 0.12, 5, 24))
        x, y = int(round(x)), int(round(y))
        left, top = max(0, x - 2 * radius), max(0, y - 3 * radius)
        right, bottom = min(width, x + 2 * radius + 1), min(height, y + radius + 1)
        cx, cy = x - left, y - top
        mask = np.zeros((bottom - top, right - left), np.uint8)
        sway = math.sin(now * 12 + index) * radius * 0.3
        outline = np.array([
            [cx - radius, cy], [cx - radius * .7, cy - radius],
            [cx + sway, cy - radius * 2.4], [cx + radius * .6, cy - radius],
            [cx + radius, cy], [cx, cy + radius * .5],
        ], dtype=np.int32)
        cv2.fillPoly(mask, [outline], 255, cv2.LINE_AA)
        alpha = (cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255 * .85)[..., None]
        region = frame[top:bottom, left:right]
        region[:] = np.rint(region * (1 - alpha) + tint * alpha).astype(np.uint8)
        cv2.ellipse(region, (cx, cy - radius // 3), (max(1, radius // 3), radius // 2),
                    0, 0, 360, (210, 240, 255), -1, cv2.LINE_AA)
