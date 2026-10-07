"""Debounced image-space fingertip overlap"""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .mapping import FaceMapping

FINGERTIPS = (4, 8, 12, 16, 20)
HAND_CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (5, 9, 10, 11, 12),
               (9, 13, 14, 15, 16), (13, 17, 18, 19, 20), (0, 17))


def face_fingertips(hands: list[np.ndarray], mapping: FaceMapping) -> list[np.ndarray]:
    """Accept only finite, on-frame tips inside the face mask and UV square."""
    height, width = mapping.mask.shape
    candidates = []
    for hand in hands:
        if hand.shape != (21, 2) or not np.isfinite(hand).all():
            continue
        for x, y in hand[list(FINGERTIPS)]:
            if not (0 <= x < width and 0 <= y < height):
                continue
            if not mapping.mask[int(y), int(x)]:
                continue
            uv = mapping.canonical_point(float(x), float(y))
            if np.isfinite(uv).all() and (uv >= 0).all() and (uv <= 1).all():
                candidates.append(uv)
    # Sorting removes dependence on MediaPipe's hand ordering/handedness labels.
    return sorted(candidates, key=lambda uv: (uv[0], uv[1]))


@dataclass(eq=False)
class _Contact:
    anchor: np.ndarray
    since: float
    fired: bool = False


class FingertipIgnition:
    """One event per stable overlap region; moving to a new region re-arms it.

    Anchors remain fixed to avoid jitter accumulating into a slow drift. Missing
    observations immediately release contacts; stalls above 0.25 s restart dwell.
    Spatial regions, rather than inferred hand identities, debounce all tips.
    """

    def __init__(self, dwell=0.12, radius=0.04):
        if not math.isfinite(dwell) or not 0 <= dwell <= 2:
            raise ValueError("Hand dwell must be finite and in [0, 2] seconds")
        if not math.isfinite(radius) or not 0 < radius <= 1:
            raise ValueError("Contact radius must be finite and in (0, 1]")
        self.dwell = dwell
        self.radius = radius
        self._contacts = []
        self._last_time = None

    def reset(self):
        self._contacts = []
        self._last_time = None

    def update(self, hands, mapping, now, *, paused=False) -> list[np.ndarray]:
        if not math.isfinite(now) or (self._last_time is not None and now < self._last_time):
            raise ValueError("Interaction clock must be finite and monotonic")
        if self._last_time is not None and now - self._last_time > 0.25:
            self._contacts = []
        self._last_time = now
        if mapping is None or paused:
            self._contacts = []
            return []
        remaining = list(self._contacts)
        active, events = [], []
        for uv in face_fingertips(hands, mapping):
            # Nearby fingertips count as one region instead of repeatedly igniting.
            if any(np.linalg.norm(uv - contact.anchor) <= self.radius for contact in active):
                continue
            nearest = min(remaining, key=lambda c: np.linalg.norm(uv - c.anchor), default=None)
            if nearest is not None and np.linalg.norm(uv - nearest.anchor) <= self.radius:
                remaining.remove(nearest)
                contact = nearest
            else:
                contact = _Contact(uv.copy(), now)
            if not contact.fired and now - contact.since + 1e-12 >= self.dwell:
                events.append(uv.copy())
                contact.fired = True
            active.append(contact)
        self._contacts = active
        return events


def draw_hands(frame: np.ndarray, hands: list[np.ndarray]) -> None:
    """Draw debug skeletons in capture coordinates before display letterboxing."""
    height, width = frame.shape[:2]
    for hand in hands:
        if hand.shape != (21, 2) or not np.isfinite(hand).all():
            continue
        points = np.rint(np.clip(hand, -4 * max(width, height), 4 * max(width, height))).astype(np.int32)
        for chain in HAND_CHAINS:
            cv2.polylines(frame, [points[list(chain)]], False, (255, 200, 0), 1, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, tuple(point), 2, (255, 200, 0), -1)
        for index in FINGERTIPS:
            cv2.circle(frame, tuple(points[index]), 4, (0, 255, 255), -1)
