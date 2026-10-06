# Incremental delivery and verification

Stop after each stage for local verification and a separate GitHub commit/push.
Automated checks and a human webcam check gate advancement. Stages 2–4 remain
planned work, not implemented functionality.

| Stage | Scope | Acceptance gate | Suggested commit |
| --- | --- | --- | --- |
| 1 | Capture, tracking, canonical mapping, clipped color patch | Tests pass; patch follows moderate movement; no background tint or overlay on tracking loss | `feat: add face-attached webcam color overlay` |
| 2 | CMake + pybind11 serial deterministic double-buffered grid; fixed timestep; ignition/reset/pause/speed; loss timeout | Kernel boundary/reset tests; controls; loss pauses updates and long loss clears before reacquisition | `feat: add deterministic serial face propagation` |
| 3 | OpenMP, execution switch and thread configuration | Equivalent grids from identical initial conditions and step counts; no shared writes | `feat: add verified OpenMP propagation` |
| 4 | Component timings, FPS/latency, repeated headless benchmarks, documentation/polish | Median timings, updates/s, speedup and efficiency; simulation-only measurements; webcam acceptance | `perf: add reproducible benchmarks and demo metrics` |

## Stage 1 manual acceptance

1. Start in good lighting. Confirm a warm patch appears on the face.
2. Translate your head, move closer/farther and tilt gently. Confirm the patch
   follows the same facial region. Strong yaw/pitch are outside this baseline.
3. Click several places inside the face; confirm the patch moves and stays
   attached. Click outside the face; confirm the location does not change.
4. Press D to inspect the outline. Confirm tint stays inside it.
5. Press O twice, then R. Confirm toggle and reset work.
6. Leave the frame or cover your face. Confirm the overlay disappears immediately.
   Return and confirm tracking resumes. Identity continuity is not guaranteed.
7. Close the window or press Q. Confirm the camera is released.

Record camera, CPU, Python/dependency versions, lighting and observed problems
when reporting a pass. Do not record or commit webcam footage.

## Planned simulation contract

Stage 2 will define a documented deterministic four-neighbor rule, reading only
`current` and writing each `next` cell once. Outside-grid neighbors contribute
zero. A fixed timestep accumulator advances independently of camera FPS; lost
tracking discards lost elapsed time rather than catching up. OpenMP parallelizes
the same cell rule with static work sharing. Benchmarks reset identical seeds
and execute identical step counts. These are constraints, not stage 1 behavior.
