"""Rolling frame metrics; completed frames are shown on the following frame."""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameTiming:
    tracking_ms: float
    mapping_ms: float
    simulation_ms: float
    rendering_ms: float
    total_ms: float


class FrameMetrics:
    def __init__(self, window=30):
        self.samples = deque(maxlen=window)
        self.presented = deque(maxlen=window + 1)

    def record(self, timing, presented_at):
        self.samples.append(timing)
        self.presented.append(presented_at)

    def lines(self):
        if not self.samples:
            return ["FPS: warming up | Processing: warming up"]
        count = len(self.samples)
        means = {name: sum(getattr(sample, name) for sample in self.samples) / count
                 for name in FrameTiming.__dataclass_fields__}
        interval = self.presented[-1] - self.presented[0]
        fps = (len(self.presented) - 1) / interval if interval > 0 else 0
        return [
            f"FPS {fps:.1f} | Processing {means['total_ms']:.2f} ms",
            f"Tracking {means['tracking_ms']:.2f} | Mapping {means['mapping_ms']:.2f} | "
            f"Simulation {means['simulation_ms']:.3f} | Rendering {means['rendering_ms']:.2f} ms",
        ]
