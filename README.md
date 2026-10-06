# Facial Fire

A local webcam facial color propagation project focused on computer vision,
parallel computing, correctness and honest performance measurement.

**Current milestone: stage 1 — face-attached color overlay.** C++ propagation,
OpenMP and benchmarks follow after local verification of this baseline.
See [the delivery stages and acceptance checklist](docs/STAGES.md).

## Windows / VS Code setup

Open this repository folder (the inner `facialModelling` directory) in VS Code.
Use 64-bit Python 3.11; this project supports 3.11–3.12. In PowerShell:

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

| Control | Stage 1 behavior |
| --- | --- |
| Left click inside face | Move canonical color patch |
| O | Toggle overlay |
| R | Reset patch position |
| D | Toggle face-outline debug view |
| Q / Escape / close window | Quit and release camera |

```powershell
.\.venv\Scripts\python.exe -m facial_fire.app --camera 0 --grid-size 128 --width 960 --height 720
```

The preview is mirrored; use `--no-mirror` to disable this. Capture dimensions
are requests and may differ on your hardware. If capture fails, try `--camera 1`,
close other camera apps and check Windows camera permissions for desktop apps.

## Architecture

`tracking.py` wraps pretrained tracking and returns pixel landmarks. `mapping.py`
maps canonical UV anchors onto outer eyes and chin, exposes an inverse for clicks
and rasterizes the ordered face oval. `rendering.py` warps a canonical intensity
texture and clips alpha after interpolation to prevent background leakage.
`app.py` owns capture, UI, cleanup and controls. Tests use synthetic landmarks;
they require neither the model nor a webcam. `scripts/download_model.py` is the
only model download step.

The current texture is a static soft patch. On tracking loss the overlay hides
immediately. Simulation state and timeout clearing arrive in stage 2. There is
no identity recognition when a face returns.

## Verification and limitations

Run `.\.venv\Scripts\python.exe -m pytest` and then the manual checklist.
Tests cover anchor correspondence, inverse mapping, motion/rotation/scale,
invalid geometry, off-frame clipping, zero intensity, unchanged input and exact
unchanged pixels outside the mask. Synthetic tests cannot establish webcam
tracking quality or camera performance.

Target one face with moderate movement and good lighting. The affine mapping
does not model depth, strong yaw/pitch or facial expression deformation. The oval
is a landmark polygon, not skin segmentation: it includes eyes, lips, facial
hair and potentially occluding objects. No frames or footage are saved.
Inference and rendering run locally after setup; application code does not send
camera data over the network.

## Upcoming native build and performance work

Stage 2 adds CMake and pybind11 with a portable serial C++ kernel. The planned
Windows toolchain is Visual Studio 2022 Build Tools with **Desktop development
with C++**, CMake and matching x64 Python. Stage 3 requires compiler OpenMP
support (MSVC or GCC; Clang may need libomp). No native build is needed yet.
MPI, CUDA, custom training and photorealistic flames are outside the baseline.

Stage 4 measures tracking, simulation, rendering and total processing separately,
with FPS and processing latency. Simulation-only benchmarks exclude capture and
rendering, warm up, reset identical initial conditions, repeat runs and report
medians across grid sizes/thread counts. Updates/s is `cells * steps / seconds`;
speedup is `serial_time / parallel_time`; efficiency is `speedup / threads`.
No performance results are claimed yet; small grids may be slower with OpenMP.
