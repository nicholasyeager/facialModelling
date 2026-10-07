# Simulation performance measurements

Measured on 2026-10-07T00:02:26.339227+00:00 (UTC).

CPU: AMD Ryzen 9 5900HX (8 cores, 16 logical CPUs). OS: Windows-10-10.0.26300-SP0.
Python 3.11.9; NumPy 2.2.6; facial-fire 0.3.0. Build: Windows x64; MSVC 19.39.33523; CMake 3.31.10; Release; OpenMP.

Each trial executes 300 ticks; 7 trials per variant; 30 untimed warmup ticks per variant.
Seed 1; dt=0.01666667; spread=12.0; cooling=0.4.

Initial state: three intensity-1 UV disks (radius 0.025) at (0.25, 0.35), (0.75, 0.65), (0.5, 0.5).
Variants are interleaved in seeded shuffled order. Every trial resets the same grid and random tick before timing. Outputs are checked bitwise against serial after each measured trial.

Timing covers one native batch call, including per-tick OpenMP team coordination and buffer swaps. It excludes capture, tracking, rendering, grid initialization/reset, snapshots and correctness checks.
Runtime is the median for the whole batch; ms/tick is its amortized average. Speedup = serial median / variant median. Efficiency = speedup / requested threads. Actual workers are shown separately; efficiency can be misleading if the runtime supplies fewer workers.

| Mode | Grid | Execution | Requested / actual workers | Median ms | ms/tick | M cell updates/s | Speedup | Efficiency |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| perimeter | 64² | serial | 1 / 1 | 6.293 | 0.0210 | 195.27 | 1.00x | 100.0% |
| perimeter | 64² | parallel | 1 / 1 | 6.545 | 0.0218 | 187.74 | 0.96x | 96.1% |
| perimeter | 64² | parallel | 2 / 2 | 3.959 | 0.0132 | 310.40 | 1.59x | 79.5% |
| perimeter | 64² | parallel | 4 / 4 | 2.604 | 0.0087 | 471.89 | 2.42x | 60.4% |
| perimeter | 64² | parallel | 8 / 8 | 3.266 | 0.0109 | 376.19 | 1.93x | 24.1% |
| perimeter | 128² | serial | 1 / 1 | 25.894 | 0.0863 | 189.82 | 1.00x | 100.0% |
| perimeter | 128² | parallel | 1 / 1 | 26.828 | 0.0894 | 183.21 | 0.97x | 96.5% |
| perimeter | 128² | parallel | 2 / 2 | 14.912 | 0.0497 | 329.61 | 1.74x | 86.8% |
| perimeter | 128² | parallel | 4 / 4 | 9.654 | 0.0322 | 509.15 | 2.68x | 67.1% |
| perimeter | 128² | parallel | 8 / 8 | 9.031 | 0.0301 | 544.28 | 2.87x | 35.8% |
| perimeter | 256² | serial | 1 / 1 | 103.680 | 0.3456 | 189.63 | 1.00x | 100.0% |
| perimeter | 256² | parallel | 1 / 1 | 106.638 | 0.3555 | 184.37 | 0.97x | 97.2% |
| perimeter | 256² | parallel | 2 / 2 | 56.055 | 0.1868 | 350.74 | 1.85x | 92.5% |
| perimeter | 256² | parallel | 4 / 4 | 36.048 | 0.1202 | 545.41 | 2.88x | 71.9% |
| perimeter | 256² | parallel | 8 / 8 | 30.284 | 0.1009 | 649.22 | 3.42x | 42.8% |
| perimeter | 512² | serial | 1 / 1 | 403.832 | 1.3461 | 194.74 | 1.00x | 100.0% |
| perimeter | 512² | parallel | 1 / 1 | 420.191 | 1.4006 | 187.16 | 0.96x | 96.1% |
| perimeter | 512² | parallel | 2 / 2 | 217.240 | 0.7241 | 362.01 | 1.86x | 92.9% |
| perimeter | 512² | parallel | 4 / 4 | 127.228 | 0.4241 | 618.13 | 3.17x | 79.4% |
| perimeter | 512² | parallel | 8 / 8 | 111.171 | 0.3706 | 707.41 | 3.63x | 45.4% |
| perimeter | 1024² | serial | 1 / 1 | 1563.122 | 5.2104 | 201.25 | 1.00x | 100.0% |
| perimeter | 1024² | parallel | 1 / 1 | 1659.198 | 5.5307 | 189.59 | 0.94x | 94.2% |
| perimeter | 1024² | parallel | 2 / 2 | 853.676 | 2.8456 | 368.49 | 1.83x | 91.6% |
| perimeter | 1024² | parallel | 4 / 4 | 489.155 | 1.6305 | 643.09 | 3.20x | 79.9% |
| perimeter | 1024² | parallel | 8 / 8 | 430.386 | 1.4346 | 730.91 | 3.63x | 45.4% |
| smooth | 64² | serial | 1 / 1 | 3.429 | 0.0114 | 358.34 | 1.00x | 100.0% |
| smooth | 64² | parallel | 1 / 1 | 3.827 | 0.0128 | 321.07 | 0.90x | 89.6% |
| smooth | 64² | parallel | 2 / 2 | 2.601 | 0.0087 | 472.47 | 1.32x | 65.9% |
| smooth | 64² | parallel | 4 / 4 | 1.754 | 0.0058 | 700.45 | 1.95x | 48.9% |
| smooth | 64² | parallel | 8 / 8 | 2.102 | 0.0070 | 584.50 | 1.63x | 20.4% |
| smooth | 128² | serial | 1 / 1 | 14.473 | 0.0482 | 339.60 | 1.00x | 100.0% |
| smooth | 128² | parallel | 1 / 1 | 15.193 | 0.0506 | 323.51 | 0.95x | 95.3% |
| smooth | 128² | parallel | 2 / 2 | 8.386 | 0.0280 | 586.11 | 1.73x | 86.3% |
| smooth | 128² | parallel | 4 / 4 | 5.065 | 0.0169 | 970.46 | 2.86x | 71.4% |
| smooth | 128² | parallel | 8 / 8 | 4.893 | 0.0163 | 1004.64 | 2.96x | 37.0% |
| smooth | 256² | serial | 1 / 1 | 61.852 | 0.2062 | 317.87 | 1.00x | 100.0% |
| smooth | 256² | parallel | 1 / 1 | 63.648 | 0.2122 | 308.90 | 0.97x | 97.2% |
| smooth | 256² | parallel | 2 / 2 | 33.856 | 0.1129 | 580.71 | 1.83x | 91.3% |
| smooth | 256² | parallel | 4 / 4 | 18.128 | 0.0604 | 1084.54 | 3.41x | 85.3% |
| smooth | 256² | parallel | 8 / 8 | 13.048 | 0.0435 | 1506.75 | 4.74x | 59.3% |
| smooth | 512² | serial | 1 / 1 | 307.528 | 1.0251 | 255.73 | 1.00x | 100.0% |
| smooth | 512² | parallel | 1 / 1 | 321.804 | 1.0727 | 244.38 | 0.96x | 95.6% |
| smooth | 512² | parallel | 2 / 2 | 213.133 | 0.7104 | 368.99 | 1.44x | 72.1% |
| smooth | 512² | parallel | 4 / 4 | 121.237 | 0.4041 | 648.67 | 2.54x | 63.4% |
| smooth | 512² | parallel | 8 / 8 | 73.906 | 0.2464 | 1064.09 | 4.16x | 52.0% |
| smooth | 1024² | serial | 1 / 1 | 1030.685 | 3.4356 | 305.21 | 1.00x | 100.0% |
| smooth | 1024² | parallel | 1 / 1 | 1084.489 | 3.6150 | 290.07 | 0.95x | 95.0% |
| smooth | 1024² | parallel | 2 / 2 | 797.831 | 2.6594 | 394.29 | 1.29x | 64.6% |
| smooth | 1024² | parallel | 4 / 4 | 455.829 | 1.5194 | 690.11 | 2.26x | 56.5% |
| smooth | 1024² | parallel | 8 / 8 | 259.797 | 0.8660 | 1210.84 | 3.97x | 49.6% |

These are measurements of this workload on this machine, not webcam FPS or end-to-end latency. Larger grids retain more cold cells during the fixed-duration growth, so the active fraction differs by grid size. Power mode, thermals, scheduling and background load affect results. JSON retains every sample and min/max timings; repeat locally before drawing general performance conclusions.
