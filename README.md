# Facial Fire

A local webcam facial color propagation project focused on computer vision,
parallel computing, correctness and honest performance measurement.

## Windows / VS Code setup

Use 64-bit Python 3.11–3.12 (3.11 recommended). Install Visual Studio
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

The editable install compiles the extension with CMake through scikit-build-core.
Build dependencies provide CMake automatically if needed. Python edits take effect
on restart; after any C++ or CMake edit, close the demo and repeat `pip install
-e '.[dev]'` to rebuild. Use a native Windows compiler with Windows Python.
If the wrong CMake generator is selected, set
`$env:CMAKE_GENERATOR = 'Visual Studio 17 2022'` in PowerShell before installing.

Linux needs a C++17 compiler such as GCC and Python development headers; macOS
needs Xcode Command Line Tools. CMake 3.20+ is required. OpenMP is detected
automatically using [CMake's FindOpenMP module](https://cmake.org/cmake/help/latest/module/FindOpenMP.html).
MSVC provides [OpenMP support](https://learn.microsoft.com/en-us/cpp/build/reference/openmp-enable-openmp-2-0-support?view=msvc-170)
with the C++ workload; GCC needs its OpenMP runtime, while Clang may need a
separate libomp installation. Without OpenMP the package builds serial-only;
requesting parallel execution then produces an error rather than silently
running serially. The build uses
[pybind11's documented CMake packaging approach](https://pybind11.readthedocs.io/en/stable/compiling.html).

To explicitly disable OpenMP (for example on a compiler without support):

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]" --config-settings "cmake.define.FACIAL_FIRE_OPENMP=OFF"
```

Use the same command with `ON` to re-enable detection. This setting persists in
the local CMake build cache. Restart the demo after rebuilding.

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

| Control | Behavior |
| --- | --- |
| Left click inside face | Add an ignition seed at that face location |
| I | Ignite the default center location |
| Space | Pause/resume simulation (tracking continues) |
| R | Clear both simulation buffers; does not change pause/speed |
| + / = and - | Increase/decrease spread speed by 2, within 0–60 |
| P | Switch serial/OpenMP execution without resetting the effect |
| O | Toggle overlay visibility; simulation continues |
| D | Toggle face-outline debug view |
| K | Toggle hand/finger skeleton independently (requires `--hand-ignition` or `--snap-colors`) |
| F | Toggle fullscreen / resizable window |
| H | Show/hide status, timing and controls panel |
| Escape | Leave fullscreen; quit when already windowed |
| Q / close window | Quit and release camera |

```powershell
.\.venv\Scripts\python.exe -m facial_fire.app --camera 0 --grid-size 128 --width 960 --height 720
```

The preview is mirrored; use `--no-mirror` to disable this. Capture dimensions
are requests and may differ on your hardware. If capture fails, try `--camera 1`,
close other camera apps and check Windows camera permissions for desktop apps.

The window is resizable. Press F to fill the screen, or launch with `--fullscreen`.
Camera resolution and display size are independent: `--width` / `--height` request
capture dimensions and set the initial window size, while fullscreen enlarges
the presentation without increasing tracking resolution. The image is letterboxed
to preserve face proportions. Clicks map back through that viewport; clicks in
the black bars are ignored. H hides the panel for an unobstructed view.

Fullscreen and geometry queries use [OpenCV HighGUI](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html)
and depend on the desktop backend. The Windows backend is the primary target;
some backends such as Wayland do not expose the same window-property support.

Start with an empty effect, then click or press I. For example:

```powershell
.\.venv\Scripts\python.exe -m facial_fire.app --grid-size 128 --spread-speed 12 --cooling 0.4 --simulation-hz 60 --loss-timeout 2 --ignition-radius 0.025 --seed 1 --execution parallel --threads 4
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
support serial and OpenMP execution and use the same fixed simulation clock and
tracking-loss policy.

## Optional fingertip ignition

Enable local tracking of up to two hands with the pretrained
[MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python).
It uses the existing MediaPipe dependency and a separate version-1 model bundle.
Download that model explicitly, then enable the feature:

```powershell
.\.venv\Scripts\python.exe scripts/download_model.py --kind hand
.\.venv\Scripts\python.exe -m facial_fire.app --hand-ignition --execution parallel --threads 4
```

Use `--hand-model path/to/hand_landmarker.task` for an existing model. Without
`--hand-ignition` or `--snap-colors`, the app neither loads the hand model nor runs hand inference.
The downloader's default remains the face model; `--output` works for either kind.

Hold a thumb or fingertip over the face in the preview for 0.12 seconds to ignite
at that point. All five fingertips are eligible. Points must be on-frame, inside
the face oval and inside the canonical simulation square. The inverse face map
places the ignition in the same UV coordinates used by mouse clicks; the seed
uses `--ignition-radius`. Face and hand inference process the same frame after
optional mirroring, so fullscreen/letterboxing does not affect gesture coordinates.

`--hand-dwell` sets the dwell time in seconds (0-2). Nearby tips within 0.04
canonical units count as one overlap region. A stationary region fires once;
moving farther than that radius starts a fresh dwell and can create more seeds.
Contacts are associated by spatial proximity rather than hand labels or list
order. This is region debouncing, not persistent hand identity tracking. Hand or
face loss, pause, and frame gaps above 0.25 seconds discard pending contacts.
R clears the effect and contact state; a finger still overlapping can ignite
again after a fresh dwell. Automatic hand ignition is suspended while paused.
K independently toggles the hand skeleton: connected finger/palm segments, all
21 joint landmarks and larger yellow fingertip markers. D toggles only the face
outline. Both views start hidden and can be combined. The status panel reports
the current detected hand count and whether the hand skeleton is visible.

This is **2D image overlap, not verified physical touch**. A hand in front of the
face can trigger ignition without touching it. Hand depth is wrist-relative and
its world coordinates are hand-centered, so they are not directly comparable
with the face model's coordinates. The dwell filter rejects brief overlaps but
cannot resolve depth ambiguity. Occlusion can disrupt either tracker; the face
oval is not a skin/hand segmentation mask, so the color effect can cover an
occluding finger. No body pose tracking is implemented. Live touch reliability
still requires webcam evaluation.

Hand inference adds work to the Tracking metric, including its RGB conversion.
Mapping includes fingertip filtering and dwell bookkeeping; Simulation includes
applying gesture seeds. Rendering includes hand debug graphics when enabled.
These totals remain rolling processing measurements; the existing simulation-only
benchmark results do not measure this additional ML workload. All inference stays
local, and webcam footage is not saved.

## Optional snap color changes

Use `--snap-colors` to cycle orange, blue, violet and green when either hand
performs a visually detected thumb/middle-finger snap. It uses the same hand
model; no extra ML dependency, gesture model, microphone or custom training is
required. The built-in [MediaPipe gesture classifier](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer)
does not include snapping. This project adds a temporal landmark heuristic:

```powershell
.\.venv\Scripts\python.exe -m facial_fire.app --snap-colors --hand-ignition --execution parallel --threads 4
```

`--snap-colors` also works independently of `--hand-ignition`. Download the hand
model as above if missing. Touch ignition and snap detection share one hand
inference per frame. The original orange tint remains the startup default.
Color changes affect rendering only; they preserve intensity, simulation ticks,
seed, speed and pause state. Snaps work while the simulation is paused or no face
is visible, so the selected color appears when the overlay returns. Simultaneous
snaps change color once per frame. R clears pending snap history but retains color.

The detector measures thumb-tip (4) to middle-tip (12) image distance divided by
the mean of wrist-to-middle-knuckle and index-to-pinky-knuckle distances. Defaults:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `--snap-close` | 0.25 | Maximum pinch distance in palm-size units |
| `--snap-release` | 0.65 | Minimum released distance in palm-size units |
| `--snap-speed` | 3.0 | Minimum average separation speed in palm-size units/second |

The tips must remain close for at least 0.04 s, then reach the release distance
within 0.25 s of the last close observation at sufficient average speed. A
0.4 s per-hand cooldown suppresses repeats; a fresh pinch is required afterward.
Distance/speed calculations use frame timestamps before inference work. K shows
hand skeletons plus per-hand normalized gap, frame-to-frame speed and detector
state in the status panel (H must be visible). The release decision uses average
speed since the last close observation, rather than the displayed instantaneous
speed. Thresholds must be finite, with `0 < close < release <= 4` and
`0 < speed <= 100`.

Hands are associated by palm-center proximity, not result order. Missing hands,
frame gaps above 0.2 s, large palm motion/scale changes and close palm crossings
discard uncertain histories. This association does not guarantee persistent
identity. Poor landmark estimates or an ambiguous hand replacement can still
produce false detections. This is a visual quick pinch-release approximation,
not proof of an audible or physical snap. Slow releases are rejected, but a fast
intentional pinch release can qualify. Low FPS, blur, occlusion and out-of-plane
hand rotation can cause misses or false positives; defaults need live evaluation.
Gesture bookkeeping is included in Mapping timings and tint/debug work in
Rendering timings. No webcam or gesture samples are saved automatically.

## Execution modes

Serial execution is the default. Start with `--execution parallel --threads 4`
to use OpenMP, or press P while the demo is running. Thread counts are integers
in 1–256; four is the default request. The serial path always uses one thread.
The overlay shows the last parallel update's actual/requested thread counts;
the runtime may provide fewer workers because of resource or environment limits.
Before the first parallel update, the last-update count is still one.

Both implementations call the same cell rule. OpenMP assigns distinct rows with
`schedule(static)`; all workers read `current`, each writes only its own `next`
cells, and the completion barrier precedes the buffer swap and tick increment.
There are no cross-cell reductions or shared random-generator writes. Switching
execution or thread count preserves the grid and random sequence. Native methods
keep the Python GIL to prevent simultaneous Python mutation of one instance;
OpenMP workers still run the cell calculations concurrently.

Equal initial conditions, seeds, parameters and tick counts produce bitwise
equal serial/parallel output within the same build. This does not promise equal
results across compilers or hardware. Parallel execution can be slower on small
grids due to thread coordination overhead. See the simulation-only measurements
below for observed results rather than assuming that more threads are faster.

## Architecture

`tracking.py` wraps pretrained tracking and returns pixel landmarks. `mapping.py`
maps canonical UV anchors onto outer eyes and chin, exposes an inverse for clicks
and rasterizes the ordered face oval. `rendering.py` warps a canonical intensity
texture and clips alpha after interpolation to prevent background leakage.
`cpp/simulation.cpp` owns the shared cell rule, serial/OpenMP loops, ignition and buffers;
`cpp/bindings.cpp` exposes it through pybind11. `simulation.py` owns the fixed-step
clock, pause and loss policy. `app.py` owns capture, UI, cleanup and controls.
`display.py` handles fullscreen, aspect-preserving presentation, inverse click
coordinates and the status panel. `metrics.py` aggregates completed-frame timings.
`tracking.py` also wraps optional hand inference; `interaction.py` filters
fingertips against the face mask and debounces canonical ignition regions.
`gestures.py` tracks temporal thumb/middle separation for optional snap events.
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
will receive the same draw irrespective of traversal order, supporting
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

## Live performance display

The status panel shows FPS, processing time and tracking/mapping/simulation/
rendering times in milliseconds. Values are arithmetic means over the last
30 completed frames, displayed on the next frame. FPS is the number of intervals
divided by elapsed time between `imshow` completions (up to 30 intervals); it
includes camera waits and UI event processing between frames. It is a loop
presentation rate, not a measurement of monitor refresh or camera-to-screen latency.

| Metric | Measured work |
| --- | --- |
| Tracking | Face tracker call, including RGB conversion/inference |
| Mapping | Canonical mapping and face-mask construction |
| Simulation | Fixed-step scheduling and all native ticks executed for this frame |
| Rendering | Grid snapshot, effect blend, debug outline, resizing/letterboxing, status panel |
| Processing | From successful capture return through `imshow` completion; includes mirroring and display submission |

Processing excludes `camera.read()` waits, `waitKey()` event processing and
key/mouse controls. It exceeds the sum of component times because it also
includes mirroring, submission and small orchestration costs. Simulation time
can vary as a frame executes zero, one or multiple fixed ticks; an average is
not a per-tick kernel benchmark. Pausing still tracks and renders the face.
Fullscreen can increase rendering cost while keeping the same simulation grid.
No live timings or camera frames are saved automatically.

## Simulation benchmarks

Run a headless comparison of both spread modes across grid sizes and thread counts:

```powershell
.\.venv\Scripts\python.exe -m facial_fire.benchmark --sizes 64 128 256 512 1024 --threads 1 2 4 8 --steps 300 --repeats 7 --warmup 30
```

The command saves JSON (every sample, median/min/max, environment and native binary
hash) and a Markdown report under `benchmark_results/`. Use `--output path.json`
to select a destination, `--modes perimeter` for only the default spread mode, and
`--cpu-label` / `--build-label` to record hardware and compiler details. OpenMP
must be available for these comparisons. Run with the webcam demo closed and
record power mode/background load when comparing machines or builds.

Every trial starts with identical seeded ignition disks and random tick state.
Each variant receives untimed warmup ticks; trial order is interleaved using a
seeded shuffle. Timing covers one native batch of fixed ticks, including OpenMP
coordination and buffer swaps. Capture, tracking, rendering, reset, grid copies,
snapshots and correctness checks are excluded. Each measured output is checked
bitwise against serial before being accepted.

Runtime is the median batch duration. Cell updates/s is `size² * ticks / seconds`;
speedup is `serial_median / parallel_median`; parallel efficiency is
`speedup / requested_threads`. Reports also record actual worker counts.

Example measurements on an AMD Ryzen 9 5900HX (8 cores / 16 logical CPUs), Windows
x64, MSVC 19.39.33523, Release build. Default perimeter mode; 300 ticks per trial,
seven trials, 30 warmup ticks. Actual workers matched requests in this run:

| Grid | Serial median ms | 4-thread median ms | Speedup | Efficiency | Parallel M cell updates/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64×64 | 6.293 | 2.604 | 2.42× | 60.4% | 471.89 |
| 128×128 | 25.894 | 9.654 | 2.68× | 67.1% | 509.15 |
| 256×256 | 103.680 | 36.048 | 2.88× | 71.9% | 545.41 |
| 512×512 | 403.832 | 127.229 | 3.17× | 79.4% | 618.13 |
| 1024×1024 | 1563.122 | 489.155 | 3.20× | 79.9% | 643.09 |

At 128×128, eight threads achieved 2.87× speedup, only slightly above four. At
64×64, four were faster than eight. The one-thread OpenMP path was slower than
the plain serial loop. See the [full measurements and methodology](benchmarks/ryzen_5900hx.md)
and [raw samples](benchmarks/ryzen_5900hx.json) for both modes and all thread counts.

These results describe this workload on one machine. The fixed tick count gives
different active fractions across resolutions. Power mode, thermals and competing
processes affect measurements; repeated runs can differ. Amortized ms/tick is
simulation processing time, not webcam FPS or end-to-end application latency.

## Verification and limitations

The [CI workflow](https://github.com/nicholasyeager/facialModelling/actions/workflows/ci.yml)
runs on pull requests targeting `main`, pushes to `main` (including merges), and
manual dispatch. It builds the native extension on Windows Server 2022 with
MSVC and Python 3.11, requires OpenMP support, and runs the headless test suite.
The OpenMP check fails the job if parallel support is absent, preventing parallel
correctness tests from silently skipping. The serial-only rejection test is
expected to skip in this OpenMP build. CI needs neither a webcam nor the model
bundle and does not assess live appearance or hardware performance. Merge
blocking requires a separate GitHub branch rule that makes the CI check required.

Run `.\.venv\Scripts\python.exe -m pytest`.
Tests cover anchor correspondence, inverse mapping, motion/rotation/scale,
invalid geometry, off-frame clipping, zero intensity, unchanged input and exact
unchanged pixels outside the mask. Synthetic tests cannot establish webcam
tracking quality or camera performance.

Tests also check the smooth native kernel against an independent NumPy reference,
single-step synchronous updates, corners/no wrapping, repeatability, both-buffer
reset, cooling, bounds/input validation, owned snapshots, fixed-step frame-rate
independence, pause/resume, tracking-loss timeout and bounded stall recovery.

Perimeter tests additionally cover identical/different seeds, irregularity,
connected growth, multiple ignition regions, one-tick frontier confinement,
diagonal distance weighting, corner clipping and random-sequence reset.

OpenMP tests compare grids exactly after each tick across both spread modes,
square/rectangular grids and 1/2/4/8 requested threads, including more threads
than rows. They also check switching execution and thread counts, batch versus
individual ticks, reset, fixed-step scheduling and timeout clearing. Parallel
checks skip on serial-only builds; rejection of unavailable parallel execution
is tested there instead.

Display tests cover portrait/landscape letterboxing, inverse click coordinates,
ignition after scaling, text wrapping and rolling timing/FPS calculations.
Hand interaction tests cover localized UV ignition, dwell/release, spatial jitter,
multiple regions, hand-order changes, motion mapping, invalid/outside landmarks,
loss/pause/stall/reset handling, and mocked hand-tracker timestamp/conversion/cleanup.
They require neither a hand model nor a webcam and do not establish touch accuracy.
Snap tests cover mirrored hands and size normalization, temporal separation,
slow/brief gesture rejection, cooldown/rearming, multiple hands and order changes,
loss/stall/motion/crossing resets, and frame-rate variation. Rendering tests verify
color changes preserve the grid and mask confinement. A synthetic app loop checks
snap-only tracking, color cycling, and independent debug toggles.

For a manual check, ignite the face, press Space, then P: the frozen pattern
should remain unchanged. Resume and confirm spreading continues. Try both spread
modes, change `--threads` between launches, and check that reset and short/long
tracking loss still behave as described. Automated tests do not verify webcam
appearance, frame latency or tracker quality.

Resize the window and press F several times; confirm the face stays proportionate
and clicks ignite at the pointer position. Check that black-bar clicks do nothing,
H hides/shows the panel, Escape exits fullscreen before quitting, and Q always
releases the camera. Watch component timings while changing execution mode,
pausing and losing/reacquiring tracking. These measurements do not imply a
performance improvement without a comparison under matching conditions.

Target one face with moderate movement and good lighting. The affine mapping
does not model depth, strong yaw/pitch or facial expression deformation. The oval
is a landmark polygon, not skin segmentation: it includes eyes, lips, facial
hair and potentially occluding objects. No frames or footage are saved.
Inference and rendering run locally after setup; application code does not send
camera data over the network.
