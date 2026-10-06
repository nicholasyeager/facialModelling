"""Fixed-step scheduling and tracking-loss policy for the native grid."""

import math

try:
    from ._native import Simulation, openmp_available
except ImportError as exc:
    raise ImportError(
        'Native simulation missing. Build with: python -m pip install -e ".[dev]" '
        '(requires a C++17 compiler; see README).'
    ) from exc


class Propagation:
    def __init__(self, size=128, spread_speed=12.0, cooling=0.4,
                 timestep=1 / 60, loss_timeout=2.0, ignition_radius=0.025,
                 spread_mode="perimeter", seed=1, execution="serial", threads=4):
        for value, low, high, name in (
            (spread_speed, 0, 60, "spread speed"),
            (cooling, 0, 60, "cooling"),
            (timestep, 1 / 240, 1, "timestep"),
            (loss_timeout, 0, 3600, "loss timeout"),
            (ignition_radius, 0, 1, "ignition radius"),
        ):
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")
        if not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("Seed must be an unsigned 64-bit integer")
        self.kernel = Simulation(size, size, mode=spread_mode, seed=seed)
        self.set_execution(execution, threads)
        self.spread_speed = spread_speed
        self.cooling = cooling
        self.timestep = timestep
        self.loss_timeout = loss_timeout
        self.ignition_radius = ignition_radius
        self.paused = False
        self._last_time = None
        self._lost_since = None
        self._loss_cleared = False
        self._was_running = False
        self._accumulator = 0.0
        self.dropped_seconds = 0.0

    @property
    def execution(self):
        return "parallel" if self.kernel.parallel else "serial"

    def set_execution(self, execution, threads=None):
        if execution not in ("serial", "parallel"):
            raise ValueError("Execution must be serial or parallel")
        threads = self.kernel.threads if threads is None else threads
        if not isinstance(threads, int) or not 1 <= threads <= 256:
            raise ValueError("Thread count must be an integer in [1, 256]")
        self.kernel.set_execution(execution == "parallel", threads)

    def toggle_execution(self):
        self.set_execution("serial" if self.kernel.parallel else "parallel")

    def ignite(self, u, v):
        self.kernel.ignite(float(u), float(v), self.ignition_radius)

    def reset(self):
        self.kernel.reset()
        self._accumulator = 0.0

    def toggle_pause(self):
        self.paused = not self.paused
        self._accumulator = 0.0
        self._was_running = False

    def adjust_speed(self, delta):
        if not math.isfinite(delta):
            raise ValueError("Speed adjustment must be finite")
        self.spread_speed = min(60.0, max(0.0, self.spread_speed + delta))

    def advance(self, now: float, tracked: bool) -> int:
        """Return executed ticks. No elapsed time is replayed after loss/pause.

        At most 0.25 seconds of an active frame interval is admitted. Excess
        wall time is counted in dropped_seconds to avoid a backlog after stalls.
        """
        if not math.isfinite(now) or (self._last_time is not None and now < self._last_time):
            raise ValueError("Simulation clock must be finite and monotonic")
        elapsed = 0.0 if self._last_time is None else now - self._last_time
        self._last_time = now
        if not tracked and self._lost_since is None:
            self._lost_since = now
            self._loss_cleared = False
        # Check before accepting a reacquired face, even if no lost frame was
        # processed at the precise timeout boundary.
        if self._lost_since is not None:
            if not self._loss_cleared and now - self._lost_since >= self.loss_timeout:
                self.reset()
                self._loss_cleared = True
        if tracked:
            self._lost_since = None
        running = tracked and not self.paused
        if not running or not self._was_running:
            self._accumulator = 0.0
            self._was_running = running
            return 0
        accepted = min(elapsed, 0.25)
        self.dropped_seconds += elapsed - accepted
        self._accumulator += accepted
        steps = int((self._accumulator + 1e-12) / self.timestep)
        if steps:
            self.kernel.step(self.timestep, self.spread_speed, self.cooling, steps)
            self._accumulator = max(0.0, self._accumulator - steps * self.timestep)
        return steps
