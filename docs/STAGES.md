# Incremental delivery and verification

Stop after each stage for local verification and a separate GitHub commit/push.
Automated checks and a human webcam check gate advancement. Stage 1 was accepted
by the user; stage 2 is the current checkpoint. Stages 3–4 remain planned work.

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

## Stage 2 manual acceptance

1. Start the rebuilt demo. The effect is initially empty. Click inside your face
   or press I to ignite. Confirm a tint expands gradually from the seed.
2. Move and tilt moderately. Confirm propagation stays attached and D shows no
   tint outside the face outline. Click outside the face: no new ignition.
3. Press Space. The pattern should freeze but still follow your moving face.
   Press Space again: propagation resumes without a sudden catch-up jump.
4. Press R: all tint clears. Wait several seconds: no spontaneous reappearance.
   Click to ignite again, including while paused.
5. Press +/= and - to change speed. With speed 0, cooling should fade the effect.
   O only hides/shows the overlay; it does not pause simulation.
6. Leave the view briefly (less than two seconds by default), then return. The
   effect should resume from the held state. Leave for longer than the timeout:
   return to an empty effect until you ignite again.
7. Try `--grid-size 64` and `--loss-timeout 1` to verify configuration. The apparent
   propagation rate varies with resolution; this is documented behavior.
8. Close the window or press Q. Confirm the camera is released.

## Implemented serial simulation contract

Stage 2 implements the documented four-neighbor rule in `README.md`, reading
only `current` and writing each `next` cell once. Outside-grid neighbors
contribute zero. A fixed timestep accumulator advances independently of camera
FPS under normal load; lost/paused time is discarded. Intervals beyond 0.25s
drop excess time to bound work. OpenMP will parallelize the same rule in stage 3.
Stage 4 benchmarks will reset identical seeds and execute identical tick counts.
