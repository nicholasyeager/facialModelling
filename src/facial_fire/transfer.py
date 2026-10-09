"""Corresponding fingertip charging and consumable face transfers in image space."""

from dataclasses import dataclass, field
from itertools import permutations
import math

import numpy as np

from .interaction import FINGERTIPS


@dataclass(eq=False)
class _HandFire:
    points: np.ndarray
    center: np.ndarray
    scale: float
    label: str | None
    charged: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=bool))
    contacts: dict = field(default_factory=dict)


class FingertipTransfer:
    """Charge once per fingertip connection; consume tips on stable face overlap.

    Tracks are paired globally using palm position, scale and optional handedness.
    Missing or ambiguous identity clears charges. Corresponding tip distances
    trigger charging with hysteresis; they cannot establish physical 3D contact.
    """

    def __init__(self, dwell=0.12, charge_dwell=0.18, tip_distance=0.35, min_pairs=3):
        if not all(math.isfinite(value) for value in (dwell, charge_dwell, tip_distance)):
            raise ValueError("Transfer settings must be finite")
        if not (0 <= dwell <= 2 and 0 <= charge_dwell <= 2 and 0 < tip_distance <= 3
                and isinstance(min_pairs, int) and not isinstance(min_pairs, bool) and 1 <= min_pairs <= 5):
            raise ValueError("Invalid transfer dwell, fingertip distance or pair count")
        self.dwell, self.charge_dwell, self.tip_distance = dwell, charge_dwell, tip_distance
        self.min_pairs = min_pairs
        self.reset()

    def reset(self):
        self._hands = []
        self._last_time = None
        self._charge_since = None
        self._connected = False
        self.lines = []

    @property
    def charged_count(self):
        return sum(int(hand.charged.sum()) for hand in self._hands)

    @property
    def flames(self):
        return [(hand.points[index].copy(), hand.scale) for hand in self._hands
                for slot, index in enumerate(FINGERTIPS) if hand.charged[slot]]

    @staticmethod
    def _pair_cost(old, new):
        if old.label is not None and new.label is not None and old.label != new.label:
            return math.inf
        if not 0.6 <= new.scale / old.scale <= 1.65:
            return math.inf
        distance = float(np.linalg.norm(new.center - old.center) / max(new.scale, old.scale))
        return distance if distance <= 1.5 else math.inf

    def _associate(self, samples):
        # Each new sample can use a distinct old track or a fresh, empty track.
        old = self._hands
        conflicts = set()
        for sample_index, sample in enumerate(samples):
            nearest = sorted((float(np.linalg.norm(sample.center - hand.center) / max(sample.scale, hand.scale)), index)
                             for index, hand in enumerate(old))
            if nearest:
                distance, index = nearest[0]
                distinct = len(nearest) == 1 or nearest[1][0] - distance > 0.1
                if (distinct and sample.label is not None and old[index].label is not None
                        and sample.label != old[index].label):
                    conflicts.add(sample_index)
        choices = []
        for assignment in permutations(range(len(old) + len(samples)), len(samples)):
            costs = [(math.inf if sample_index in conflicts else self._pair_cost(old[index], sample))
                     if index < len(old) else 1.6
                     for sample_index, (index, sample) in enumerate(zip(assignment, samples))]
            if all(math.isfinite(cost) for cost in costs):
                # Fresh-track permutations are equivalent; deduplicate below.
                key = tuple(index if index < len(old) else -1 for index in assignment)
                choices.append((sum(costs), key))
        choices = sorted(set(choices))
        if not choices:
            return
        # A near-tie means identity is uncertain (e.g. palms crossing).
        if len(choices) > 1 and choices[1][0] - choices[0][0] < 0.1:
            return
        for sample, index in zip(samples, choices[0][1]):
            if index >= 0:
                sample.charged = old[index].charged.copy()
                sample.contacts = old[index].contacts.copy()

    def update(self, hands, mapping, now, *, paused=False, labels=()):
        if not math.isfinite(now) or (self._last_time is not None and now < self._last_time):
            raise ValueError("Transfer clock must be finite and monotonic")
        if self._last_time is not None and now - self._last_time > 0.25:
            self.reset()
        self._last_time = now
        samples = []
        for index, points in enumerate(hands[:2]):
            if points.shape != (21, 2) or not np.isfinite(points).all():
                continue
            scale = float((np.linalg.norm(points[0] - points[9]) + np.linalg.norm(points[5] - points[17])) / 2)
            if scale < 8:
                continue
            label = labels[index] if index < len(labels) else None
            samples.append(_HandFire(points.copy(), points[[0, 5, 9, 13, 17]].mean(axis=0), scale, label))
        self._associate(samples)
        self._hands = samples
        if paused:
            for hand in samples:
                hand.contacts.clear()
            self._charge_since = None
            self.lines = [f"Transfer paused | charged tips {self.charged_count}"]
            return []
        gaps = np.full(5, math.inf)
        if len(samples) == 2:
            gaps = np.linalg.norm(samples[0].points[list(FINGERTIPS)] - samples[1].points[list(FINGERTIPS)], axis=1)
            gaps /= (samples[0].scale + samples[1].scale) / 2
        close_pairs = int(np.count_nonzero(gaps <= self.tip_distance))
        held_pairs = int(np.count_nonzero(gaps <= self.tip_distance * 1.4))
        diagnostics = " | ".join(f"{name} {gap:.2f}" if math.isfinite(gap) else f"{name} --"
                                 for name, gap in zip(("T", "I", "M", "R", "P"), gaps))
        # Release needs more separation than entry, to reject boundary jitter.
        if held_pairs < self.min_pairs:
            self._connected = False
            self._charge_since = None
        if close_pairs >= self.min_pairs:
            if self._charge_since is None:
                self._charge_since = now
            if now - self._charge_since + 1e-12 >= self.charge_dwell:
                self._connected = True
        elif not self._connected:
            self._charge_since = None
        if self._connected:
            # While connected every tip has the same state. This safely survives
            # ambiguous palm association; consumption begins only after release.
            for hand in samples:
                hand.charged[:] = True
                hand.contacts.clear()
            self.lines = [f"Fingertips charged - separate to transfer | tips {self.charged_count}", diagnostics]
            return []
        # Suppress transfer during an incomplete fingertip connection as well.
        if self._charge_since is not None:
            for hand in samples:
                hand.contacts.clear()
            self.lines = [f"Fingertips connecting: {close_pairs}/5 pairs", diagnostics]
            return []
        events = []
        for hand in samples:
            current = {}
            if mapping is not None:
                height, width = mapping.mask.shape
                for slot, tip in enumerate(FINGERTIPS):
                    if not hand.charged[slot]:
                        continue
                    x, y = hand.points[tip]
                    if not (0 <= x < width and 0 <= y < height and mapping.mask[int(y), int(x)]):
                        continue
                    uv = mapping.canonical_point(float(x), float(y))
                    if not (np.isfinite(uv).all() and (uv >= 0).all() and (uv <= 1).all()):
                        continue
                    anchor, since = hand.contacts.get(slot, (uv.copy(), now))
                    if np.linalg.norm(uv - anchor) > 0.04:
                        anchor, since = uv.copy(), now
                    if now - since + 1e-12 >= self.dwell:
                        events.append(uv.copy())
                        hand.charged[slot] = False
                    else:
                        current[slot] = (anchor, since)
            hand.contacts = current
        self.lines = [f"Close fingertip pairs {close_pairs}/5 | charged tips {self.charged_count}", diagnostics]
        return events
