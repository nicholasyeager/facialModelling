# Stage 1 checkpoint — October 6, 2026

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

Stage 2 should begin after the stage 1 manual checklist in `STAGES.md` passes.
