import pytest

from facial_fire.benchmark import benchmark_case, summarize
from facial_fire._native import openmp_available


def test_report_math_uses_median_and_requested_thread_efficiency():
    result = summarize([0.3, 0.1, 0.2], cells=100, steps=10, serial_seconds=0.4, threads=4)
    assert result["median_seconds"] == pytest.approx(0.2)
    assert result["cell_updates_per_second"] == pytest.approx(5000)
    assert result["speedup"] == pytest.approx(2)
    assert result["parallel_efficiency"] == pytest.approx(0.5)
    assert result["milliseconds_per_tick"] == pytest.approx(20)


@pytest.mark.skipif(not openmp_available, reason="Requires OpenMP comparison")
def test_benchmark_checks_equivalence_and_keeps_all_repeated_samples():
    rows = benchmark_case(8, "perimeter", [1, 2], steps=5, repeats=3, warmup=2,
                          seed=1, dt=1 / 60, speed=12, cooling=0.4)
    assert len(rows) == 3
    assert rows[0]["execution"] == "serial"
    assert rows[0]["speedup"] == 1
    for row in rows:
        assert row["correctness"] == "bitwise_equal"
        assert len(row["samples_seconds"]) == 3
        assert all(seconds > 0 for seconds in row["samples_seconds"])
