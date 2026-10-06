# Facial Fire

A local webcam facial color propagation project focused on computer vision,
parallel computing, correctness and honest performance measurement.

**Current milestone: stage 2 — deterministic serial C++ propagation.** OpenMP
and benchmarks follow after local verification of this checkpoint.
See [the delivery stages and acceptance checklist](docs/STAGES.md).

## Windows / VS Code setup

Open this repository folder (the inner `facialModelling` directory) in VS Code.
Use 64-bit Python 3.11; this project supports 3.11–3.12. Install Visual Studio
2022 Community or Build Tools with **Desktop development with C++**, the MSVC
x64/x86 toolset and a Windows SDK. An existing Visual Studio installation needs
that workload; the editor alone is insufficient. In PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts/download_model.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m facial_fire.app
```

Select `.venv\Scripts\python.exe` with **Python: Select Interpreter** in VS Code.
Activation is optional. On Linux/macOS use `python3.11` and `.venv/bin/python`.
OpenCV needs a desktop display; a headless package cannot show this demo.

For your Cygwin Bash terminal, use Windows Python with forward slashes:

```bash
./.venv/Scripts/python.exe -m pip install -e '.[dev]'
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m facial_fire.app
```

The editable install compiles the extension with CMake through scikit-build-core.
Build dependencies provide CMake automatically if needed. Python edits take effect
on restart; after any C++ or CMake edit, close the demo and repeat `pip install
-e '.[dev]'` to rebuild. This uses a native Windows compiler and Python ABI even
when launched from Cygwin; do not mix Cygwin Python/GCC with this Windows venv.
If an environment selects the wrong CMake generator, prefix the Bash install
command with `CMAKE_GENERATOR='Visual Studio 17 2022'` (PowerShell:
`$env:CMAKE_GENERATOR = 'Visual Studio 17 2022'`).

Linux needs a C++17 compiler such as GCC and Python development headers; macOS
needs Xcode Command Line Tools. CMake 3.20+ is required. No OpenMP runtime or
compiler OpenMP support is required in the current serial stage. The build uses
[pybind11's documented CMake packaging approach](https://pybind11.readthedocs.io/en/stable/compiling.html).

The explicit download fetches Google's version-1 `face_landmarker.task` bundle
into `models/` and prints its SHA256 for provenance. Models are ignored by Git.
The app does not download anything. Use `--model path/to/face_landmarker.task`
to supply an existing local model.

Tracking uses the maintained [MediaPipe Face Landmarker Tasks API](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python),
reviewed October 6, 2026, and the [official model bundle](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models).
The package release is pinned for repeatable setup. VIDEO mode pairs each
processed camera frame with its landmarks synchronously. Monotonic timestamps
and `num_faces=1` enable tracking and one-face smoothing.

## Controls

| Control | Stage 2 behavior |
| --- | --- |
| Left click inside face | Add an ignition seed at that face location |
| I | Ignite the default center location |
| Space | Pause/resume simulation (tracking continues) |
| R | Clear both simulation buffers; does not change pause/speed |
| + / = and - | Increase/decrease spread speed by 2, within 0–60 |
| O | Toggle overlay visibility; simulation continues |
| D | Toggle face-outline debug view |
| Q / Escape / close window | Quit and release camera |

```powershell
.\.venv\Scripts\python.exe -m facial_fire.app --camera 0 --grid-size 128 --width 960 --height 720
```

The preview is mirrored; use `--no-mirror` to disable this. Capture dimensions
are requests and may differ on your hardware. If capture fails, try `--camera 1`,
close other camera apps and check Windows camera permissions for desktop apps.

Start with an empty effect, then click or press I. For example, in Cygwin:

```bash
./.venv/Scripts/python.exe -m facial_fire.app --grid-size 128 --spread-speed 12 --cooling 0.4 --simulation-hz 60 --loss-timeout 2 --ignition-radius 0.025 --seed 1
```

Grid size is 2–2048 per axis (128 recommended); speed and cooling are rates in
0–60 per second, simulation frequency is 1–240 Hz (60 recommended), timeout is
0–3600 seconds and seed radius is 0–1 in canonical UV units. Speed is a growth
rate, not a promise of pixels/second. The front crosses more cells at larger
resolutions, so apparent spread speed changes with resolution. At speed 0,
cooling fades the existing effect without activating new cells. Paused ignition
is allowed, but the seed will not spread until resumed.

**Perimeter growth is now the default.** Choose another `--seed` for a different
irregular front; resetting repeats the same seed's sequence. Use `--spread-mode
smooth` to compare with the original uniform four-neighbor spread. Both modes
remain serial C++ and use the same fixed simulation clock and tracking-loss policy.

## Architecture

`tracking.py` wraps pretrained tracking and returns pixel landmarks. `mapping.py`
maps canonical UV anchors onto outer eyes and chin, exposes an inverse for clicks
and rasterizes the ordered face oval. `rendering.py` warps a canonical intensity
texture and clips alpha after interpolation to prevent background leakage.
`cpp/simulation.cpp` owns the deterministic serial update, ignition and buffers;
`cpp/bindings.cpp` exposes it through pybind11. `simulation.py` owns the fixed-step
clock, pause and loss policy. `app.py` owns capture, UI, cleanup and controls.
Tests use synthetic landmarks;
they require neither the model nor a webcam. `scripts/download_model.py` is the
only model download step.

On tracking loss the overlay hides and updates pause immediately. A short loss
preserves state; at the configured timeout both buffers clear, including when
the next observed frame already contains a reacquired face. Lost/paused time is
discarded instead of caught up. There is no identity recognition when a face
returns, so a different face can inherit state during a short loss.

## Simulation rule and timing

Each cell stores a float32 intensity in [0, 1]. Default `perimeter` mode treats
intensities >= 0.25 as active. Each tick:

1. Read the strongest of eight neighbors from `current`. Diagonal intensities
   are weighted by 1/sqrt(2) to account for distance.
2. Cold cells with a sufficiently active neighbor form the growth frontier.
   Each is selected with probability `1 - exp(-spread_speed * dt * neighbor)`.
   Selected cells ignite at `max(0.35, 0.75 * neighbor)`; unselected cold cells
   only cool. There is no random ignition away from the current frontier.
3. Already active cells use the continuous growth/cooling formula below. This
   strengthens newly grown regions gradually. Every cell writes only its own
   `next` location, so new ignition cannot cascade farther within that tick.

The frontier follows all ignition regions automatically; separate regions can
merge. It includes cold holes adjacent to active cells as well as exterior edges.
There is no separate contour sampling or fluid dynamics solver. This makes an
irregular stochastic growth front rather than a physically realistic flame.

Random draws use a 64-bit integer mixer of `(seed, simulation_tick, cell_index)`
and its top 24 bits, with no global `srand()/rand()` state. The same cell and tick
will receive the same draw irrespective of traversal order, supporting future
serial/OpenMP equivalence. `reset()` and `set_grid()` rewind the tick counter;
additional click ignitions do not rewind it. Seeds are unsigned 64-bit integers.

For the optional `smooth` mode, the original rule reads only four neighbors and
applies continuous growth/cooling to all cells:

```text
neighbor = max(north, south, west, east) from current
growth = spread_speed * neighbor * (1 - current_cell)
next_cell = clamp(current_cell + dt * (growth - cooling * current_cell), 0, 1)
```

Missing neighbors contribute zero; there is no edge wrapping. Every cell reads
only `current` and writes exactly one location in `next`, then the buffers swap.
An ignition raises intensities in a UV disk to 1 and always includes the nearest
cell. Reset clears both buffers. Python receives an owned snapshot for rendering
so it cannot mutate native state or retain a stale buffer view.

This is a reproducible color-spread model with cooling, without fuel depletion.
Neighboring active cells can sustain each other; it need not burn
out without reset or lower spread speed. At extreme rates/low tick frequencies,
clipping keeps intensities bounded but the dynamics become coarse. The canonical
square is simulated in full; the current face mask confines the *rendered* effect.

The Python clock accumulates active elapsed time and runs whole fixed ticks,
independent of camera FPS under normal load. For responsiveness, it admits at
most 0.25 seconds per active frame interval; longer stalls drop excess time,
recorded in `Propagation.dropped_seconds`, rather than building a backlog. The
first frame after pause/loss establishes a fresh clock baseline. Identical seeds,
parameters and tick counts give repeatable results on the same build; bitwise
identity across different compilers/hardware is not promised.

## Verification and limitations

Run `.\.venv\Scripts\python.exe -m pytest` and then the manual checklist.
Tests cover anchor correspondence, inverse mapping, motion/rotation/scale,
invalid geometry, off-frame clipping, zero intensity, unchanged input and exact
unchanged pixels outside the mask. Synthetic tests cannot establish webcam
tracking quality or camera performance.

Stage 2 also tests the native kernel against an independent NumPy reference,
single-step synchronous updates, corners/no wrapping, repeatability, both-buffer
reset, cooling, bounds/input validation, owned snapshots, fixed-step frame-rate
independence, pause/resume, tracking-loss timeout and bounded stall recovery.
See [recorded validation](docs/VALIDATION.md) and the stage 2 manual checklist.

Perimeter tests additionally cover identical/different seeds, irregularity,
connected growth, multiple ignition regions, one-tick frontier confinement,
diagonal distance weighting, corner clipping and random-sequence reset.

Target one face with moderate movement and good lighting. The affine mapping
does not model depth, strong yaw/pitch or facial expression deformation. The oval
is a landmark polygon, not skin segmentation: it includes eyes, lips, facial
hair and potentially occluding objects. No frames or footage are saved.
Inference and rendering run locally after setup; application code does not send
camera data over the network.

## Upcoming parallel and performance work

Stage 3 adds OpenMP using the same cell rule, an execution-mode switch, thread
configuration and serial/parallel equivalence tests. It will require compiler
OpenMP support (MSVC or GCC; Clang may need libomp).
MPI, CUDA, custom training and photorealistic flames are outside the baseline.

Stage 4 measures tracking, simulation, rendering and total processing separately,
with FPS and processing latency. Simulation-only benchmarks exclude capture and
rendering, warm up, reset identical initial conditions, repeat runs and report
medians across grid sizes/thread counts. Updates/s is `cells * steps / seconds`;
speedup is `serial_time / parallel_time`; efficiency is `speedup / threads`.
No performance results are claimed yet; small grids may be slower with OpenMP.
