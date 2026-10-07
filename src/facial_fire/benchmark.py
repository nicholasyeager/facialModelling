"""Headless, repeated simulation-only serial/OpenMP comparisons."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time

import numpy as np

from ._native import Simulation, openmp_available


def initial_grid(size, mode, seed):
    kernel = Simulation(size, size, mode=mode, seed=seed)
    for u, v in ((0.25, 0.35), (0.75, 0.65), (0.5, 0.5)):
        kernel.ignite(u, v, radius=0.025)
    return kernel.snapshot()


def summarize(samples, cells, steps, serial_seconds, threads):
    seconds = statistics.median(samples)
    speedup = serial_seconds / seconds
    return {
        "median_seconds": seconds,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "milliseconds_per_tick": seconds * 1000 / steps,
        "cell_updates_per_second": cells * steps / seconds,
        "speedup": speedup,
        "parallel_efficiency": speedup / threads,
    }


def benchmark_case(size, mode, threads, steps, repeats, warmup, seed, dt, speed, cooling):
    initial = initial_grid(size, mode, seed)
    reference = Simulation(size, size, mode=mode, seed=seed)
    reference.set_grid(initial)
    reference.step(dt, speed, cooling, steps=steps)
    expected = reference.snapshot()
    variants = [("serial", 1)] + [("parallel", count) for count in threads]
    kernels, samples, actual = {}, {}, {}
    for execution, count in variants:
        key = (execution, count)
        kernel = Simulation(size, size, mode=mode, seed=seed)
        kernel.set_execution(execution == "parallel", count)
        kernel.set_grid(initial)
        kernel.step(dt, speed, cooling, steps=warmup)
        kernels[key], samples[key], actual[key] = kernel, [], []
    # Deterministic interleaving avoids timing all serial trials first and all
    # parallel trials later. Each sample begins from the same grid/RNG tick.
    ordering = random.Random(seed)
    for _ in range(repeats):
        order = variants.copy()
        ordering.shuffle(order)
        for key in order:
            kernel = kernels[key]
            kernel.set_grid(initial)  # Reset and copying are outside measurement.
            started = time.perf_counter_ns()
            kernel.step(dt, speed, cooling, steps=steps)
            seconds = (time.perf_counter_ns() - started) / 1e9
            samples[key].append(seconds)
            actual[key].append(kernel.last_threads)
            # Every timed trial is checked, after stopping the clock.
            if not np.array_equal(kernel.snapshot(), expected):
                raise RuntimeError(f"Incorrect output for {mode}, {size}, {key}")
    serial_seconds = statistics.median(samples[("serial", 1)])
    rows = []
    for execution, count in variants:
        key = execution, count
        rows.append({
            "mode": mode, "grid_size": size, "execution": execution,
            "requested_threads": count, "actual_threads": sorted(set(actual[key])),
            "samples_seconds": samples[key], "correctness": "bitwise_equal",
            **summarize(samples[key], size * size, steps, serial_seconds, count),
        })
    return rows


def markdown_report(report):
    configuration = report["configuration"]
    environment = report["environment"]
    lines = [
        "# Simulation performance measurements", "",
        f"Measured on {report['measured_at_utc']} (UTC).", "",
        f"CPU: {environment['cpu_label']}. OS: {environment['platform']}.",
        f"Python {environment['python']}; NumPy {environment['numpy']}; "
        f"facial-fire {environment['package_version']}. Build: {environment['build_label']}.", "",
        f"Each trial executes {configuration['steps']} ticks; "
        f"{configuration['repeats']} trials per variant; "
        f"{configuration['warmup']} untimed warmup ticks per variant.",
        f"Seed {configuration['seed']}; dt={configuration['dt']:.8f}; "
        f"spread={configuration['spread_speed']}; cooling={configuration['cooling']}.", "",
        "Initial state: three intensity-1 UV disks (radius 0.025) at "
        "(0.25, 0.35), (0.75, 0.65), (0.5, 0.5).",
        "Variants are interleaved in seeded shuffled order. Every trial resets "
        "the same grid and random tick before timing. Outputs are checked bitwise "
        "against serial after each measured trial.", "",
        "Timing covers one native batch call, including per-tick OpenMP team "
        "coordination and buffer swaps. It excludes capture, tracking, rendering, "
        "grid initialization/reset, snapshots and correctness checks.",
        "Runtime is the median for the whole batch; ms/tick is its amortized average. "
        "Speedup = serial median / variant median. Efficiency = speedup / requested "
        "threads. Actual workers are shown separately; efficiency can be misleading "
        "if the runtime supplies fewer workers.", "",
        "| Mode | Grid | Execution | Requested / actual workers | Median ms | ms/tick | M cell updates/s | Speedup | Efficiency |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        actual = ",".join(map(str, row["actual_threads"]))
        lines.append(
            f"| {row['mode']} | {row['grid_size']}² | {row['execution']} | "
            f"{row['requested_threads']} / {actual} | {row['median_seconds'] * 1000:.3f} | "
            f"{row['milliseconds_per_tick']:.4f} | {row['cell_updates_per_second'] / 1e6:.2f} | "
            f"{row['speedup']:.2f}x | {100 * row['parallel_efficiency']:.1f}% |"
        )
    lines.extend(["", "These are measurements of this workload on this machine, not "
                  "webcam FPS or end-to-end latency. Larger grids retain more cold cells "
                  "during the fixed-duration growth, so the active fraction differs by "
                  "grid size. Power mode, thermals, scheduling and background load affect "
                  "results. JSON retains every sample and min/max timings; repeat locally "
                  "before drawing general performance conclusions.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--modes", choices=("perimeter", "smooth"), nargs="+", default=["perimeter", "smooth"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/latest.json"))
    parser.add_argument("--cpu-label", default=platform.processor() or "unknown")
    parser.add_argument("--build-label", default="not supplied")
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1 or args.warmup < 1:
        parser.error("steps, repeats and warmup must be positive")
    if any(not 2 <= size <= 2048 for size in args.sizes):
        parser.error("grid sizes must be in [2, 2048]")
    if any(not 1 <= count <= 256 for count in args.threads):
        parser.error("thread counts must be in [1, 256]")
    if not 0 <= args.seed < 2**64:
        parser.error("seed must be an unsigned 64-bit integer")
    if args.threads and not openmp_available:
        parser.error("OpenMP unavailable; rebuild with OpenMP for performance comparisons")
    from . import _native
    report = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "sizes": list(dict.fromkeys(args.sizes)), "threads": list(dict.fromkeys(args.threads)),
            "modes": list(dict.fromkeys(args.modes)), "steps": args.steps,
            "repeats": args.repeats, "warmup": args.warmup, "seed": args.seed,
            "dt": 1 / 60, "spread_speed": 12.0, "cooling": 0.4,
        },
        "environment": {
            "platform": platform.platform(), "cpu_label": args.cpu_label,
            "logical_cpus": os.cpu_count(), "python": platform.python_version(),
            "numpy": np.__version__, "package_version": version("facial-fire"),
            "build_label": args.build_label, "openmp_available": openmp_available,
            "native_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
            "openmp_environment": {name: os.environ[name] for name in (
                "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_THREAD_LIMIT", "OMP_PROC_BIND", "OMP_PLACES",
            ) if name in os.environ},
        },
        "results": [],
    }
    config = report["configuration"]
    for mode in config["modes"]:
        for size in config["sizes"]:
            rows = benchmark_case(size, mode, config["threads"], args.steps, args.repeats,
                                  args.warmup, args.seed, config["dt"], 12.0, 0.4)
            report["results"].extend(rows)
            print(f"{mode} {size}x{size}: " + " | ".join(
                f"{row['execution']} {row['requested_threads']}t: "
                f"{row['median_seconds'] * 1000:.2f} ms ({row['speedup']:.2f}x)" for row in rows
            ), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    print(f"Saved {args.output} and {args.output.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
