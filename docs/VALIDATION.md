# Validation checkpoints

## Stage 2 — October 6, 2026

Environment: Windows x64, Python 3.11, MSVC 19.39.33523 / Visual Studio 2022
Community, CMake 3.31.10 / Visual Studio 17 2022 generator, Release build,
scikit-build-core 0.11.6,
pybind11 3.0.1. Existing tracking/rendering dependencies remain as below.

- Editable package compiled and installed successfully through CMake.
- `python -m pytest -q`: **52 passed**, including all eight stage 1 tests.
- Native output agrees with independent NumPy updates across rectangular grids
  over 40 ticks within float32 tolerances; repeat/batched runs agree exactly.
- Checks cover four-neighbor/no-wrap boundaries, synchronous double buffering,
  reset after odd buffer swaps, cooling, bounded intensities, input rejection,
  copies across the binding, fixed tick counts at 30/144 frame rates, user pause,
  short/long tracking loss, clearing on reacquisition, bounded stall recovery and
  native propagation mapped through the face mask with unchanged outside pixels.
- Cygwin Bash `./.venv/Scripts/python.exe -m facial_fire.app --help`: passed;
  native extension imports and all new CLI options are available.
- Live stage 2 webcam acceptance: **pending user verification**. No webcam was
  opened or saved during automated checks.
- Performance/parallel equivalence: deferred to stages 3–4; no speedup claim.

## Stage 1 — October 6, 2026

Environment: Windows x64, Python 3.11 virtual environment, MediaPipe 0.10.32,
OpenCV contrib 4.11.0.86, NumPy 2.2.6, pytest 8.4.2.

- `python -m pytest -q`: **8 passed**.
- Official model loaded successfully in FaceLandmarker VIDEO mode.
- Two synthetic blank frames correctly returned no detected face; repeated input
  timestamps were converted to strictly increasing task timestamps.
- `git diff --check`: passed.
- Live webcam visual acceptance: **pending user verification**. No webcam was
  opened and no footage was saved during these checks.
- Performance: not measured at this stage; no FPS or speedup claim.

Model source:
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`

Downloaded model SHA256:
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`

The user accepted stage 1 before authorizing stage 2.
