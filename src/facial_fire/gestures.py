"""Temporal thumb/middle release heuristic over existing 2D hand landmarks."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(eq=False)
class _HandState:
    center: np.ndarray
    scale: float
    gap: float
    observed: float
    pinch_since: float | None = None
    armed: bool = False
    closed_at: float = 0.0
    closed_gap: float = 0.0
    cooldown_until: float = 0.0


class SnapDetector:
    """Require stable pinch, rapid release, then a fresh pinch after cooldown.

    Distances use palm-size units; speed uses palm-size units/second. Hands are
    matched by palm-center proximity, not list order or handedness. Ambiguous
    crossings, missing observations and >0.2 s frame gaps discard history.
    """

    def __init__(self, close=0.25, release=0.65, min_speed=3.0,
                 pinch_dwell=0.04, release_window=0.25, cooldown=0.4):
        values = (close, release, min_speed, pinch_dwell, release_window, cooldown)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Snap settings must be finite")
        if not (0 < close < release <= 4 and 0 < min_speed <= 100
                and 0 <= pinch_dwell <= 2 and 0 < release_window <= 2 and 0 <= cooldown <= 5):
            raise ValueError("Invalid snap thresholds, speed, dwell, release window or cooldown")
        self.close = close
        self.release = release
        self.min_speed = min_speed
        self.pinch_dwell = pinch_dwell
        self.release_window = release_window
        self.cooldown = cooldown
        self._hands = []
        self._last_time = None
        self.lines = []

    def reset(self):
        self._hands = []
        self._last_time = None
        self.lines = []

    def update(self, hands, now) -> int:
        if not math.isfinite(now) or (self._last_time is not None and now < self._last_time):
            raise ValueError("Gesture clock must be finite and monotonic")
        if self._last_time is not None and now - self._last_time > 0.2:
            self._hands = []
        self._last_time = now
        samples = []
        for hand in hands:
            if hand.shape != (21, 2) or not np.isfinite(hand).all():
                continue
            scale = (np.linalg.norm(hand[0] - hand[9]) + np.linalg.norm(hand[5] - hand[17])) / 2
            if scale < 8:
                continue
            center = hand[[0, 5, 9, 17]].mean(axis=0)
            gap = float(np.linalg.norm(hand[4] - hand[12]) / scale)
            samples.append((center, float(scale), gap))
        samples.sort(key=lambda sample: (sample[0][0], sample[0][1]))
        # Close/crossing palms cannot be safely associated: start new histories.
        ambiguous = any(np.linalg.norm(a[0] - b[0]) < 0.5 * max(a[1], b[1])
                        for i, a in enumerate(samples) for b in samples[i + 1:])
        previous = [] if ambiguous else list(self._hands)
        active, events, self.lines = [], 0, []
        for center, scale, gap in samples:
            state = min(previous, key=lambda hand: np.linalg.norm(center - hand.center), default=None)
            if state is not None and (np.linalg.norm(center - state.center) > 1.5 * max(scale, state.scale)
                                      or not 0.6 <= scale / state.scale <= 1.65):
                state = None
            if state is None:
                state = _HandState(center, scale, gap, now)
            else:
                previous.remove(state)
            dt = now - state.observed
            speed = (gap - state.gap) / dt if dt > 0 else 0.0
            phase = "open"
            if now < state.cooldown_until:
                state.pinch_since, state.armed = None, False
                phase = "cooldown"
            elif gap <= self.close:
                if state.pinch_since is None:
                    state.pinch_since = now
                state.armed = now - state.pinch_since + 1e-12 >= self.pinch_dwell
                state.closed_at, state.closed_gap = now, gap
                phase = "armed" if state.armed else "pinching"
            elif state.armed:
                elapsed = now - state.closed_at
                average_speed = (gap - state.closed_gap) / elapsed if elapsed > 0 else 0.0
                phase = "releasing"
                if elapsed > self.release_window or gap >= self.release:
                    if elapsed <= self.release_window and gap >= self.release and average_speed >= self.min_speed:
                        events += 1
                        state.cooldown_until = now + self.cooldown
                        phase = "snap"
                    state.pinch_since, state.armed = None, False
            else:
                state.pinch_since = None
            state.center, state.scale, state.gap, state.observed = center, scale, gap, now
            active.append(state)
            self.lines.append(f"Snap hand {len(active)}: gap {gap:.2f} | speed {speed:+.1f}/s | {phase}")
        self._hands = active
        return events
