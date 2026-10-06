import numpy as np
import pytest

from facial_fire._native import Simulation
from facial_fire.simulation import Propagation


def seeded_grid(seed, steps=30):
    kernel = Simulation(41, 41, seed=seed)
    kernel.ignite(0.5, 0.5, radius=0.07)
    kernel.step(1 / 60, 12, 0.4, steps=steps)
    return kernel.snapshot()


def test_same_seed_reproduces_and_different_seeds_change_front():
    first = seeded_grid(123)
    np.testing.assert_array_equal(first, seeded_grid(123))
    assert not np.array_equal(first, seeded_grid(456))


def test_single_tick_selects_only_current_surface_without_cascades():
    kernel = Simulation(9, 9, seed=42)
    kernel.ignite(0.5, 0.5, radius=0)
    # Saturated probability activates all eight adjacent perimeter cells.
    kernel.step(1, 60, 0)
    grid = kernel.snapshot()
    expected_support = np.zeros((9, 9), bool)
    expected_support[3:6, 3:6] = True
    np.testing.assert_array_equal(grid >= 0.25, expected_support)
    assert grid[4, 4] == 1
    assert grid[3, 4] == pytest.approx(0.75)
    assert grid[3, 3] == pytest.approx(0.75 / np.sqrt(2), abs=1e-6)


def test_random_front_is_irregular_and_has_no_disconnected_islands():
    grid = seeded_grid(7, steps=40)
    active = grid >= 0.25
    assert active.sum() > 25  # Growth beyond the ignition disk.
    assert not np.array_equal(active, active[:, ::-1])
    # Independent flood fill: every active cell connects to the center seed.
    pending = [(20, 20)]
    visited = set(pending)
    while pending:
        row, col = pending.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbor = (row + dr, col + dc)
                if (0 <= neighbor[0] < 41 and 0 <= neighbor[1] < 41
                        and active[neighbor] and neighbor not in visited):
                    visited.add(neighbor)
                    pending.append(neighbor)
    assert len(visited) == active.sum()


def test_multiple_ignitions_grow_independently_until_contact():
    kernel = Simulation(21, 21)
    kernel.ignite(0.2, 0.2, radius=0)
    kernel.ignite(0.8, 0.8, radius=0)
    kernel.step(1, 60, 0)
    expected = np.zeros((21, 21), bool)
    expected[3:6, 3:6] = True
    expected[15:18, 15:18] = True
    np.testing.assert_array_equal(kernel.snapshot() >= 0.25, expected)


def test_perimeter_corner_clips_without_wrapping():
    kernel = Simulation(4, 5)
    kernel.ignite(1, 0, radius=0)
    kernel.step(1, 60, 0)
    expected = np.zeros((4, 5), bool)
    expected[:2, 3:] = True
    np.testing.assert_array_equal(kernel.snapshot() > 0, expected)


def test_reset_restores_random_sequence_and_zero_dt_does_not_consume_it():
    kernel = Simulation(21, 21, seed=321)
    kernel.ignite(0.5, 0.5)
    kernel.step(1 / 60, 12, 0.4, steps=50)
    original = kernel.snapshot()
    kernel.reset()
    kernel.ignite(0.5, 0.5)
    kernel.step(0, 12, 0.4, steps=100)
    kernel.step(1 / 60, 12, 0.4, steps=50)
    np.testing.assert_array_equal(kernel.snapshot(), original)


def test_set_grid_restores_tick_counter_and_empty_grid_cannot_ignite_itself():
    kernel = Simulation(21, 21, seed=321)
    kernel.ignite(0.5, 0.5)
    initial = kernel.snapshot()
    kernel.step(1 / 60, 12, 0.4, steps=50)
    expected = kernel.snapshot()
    kernel.set_grid(initial)
    kernel.step(1 / 60, 12, 0.4, steps=50)
    np.testing.assert_array_equal(kernel.snapshot(), expected)
    kernel.reset()
    kernel.step(1 / 60, 12, 0.4, steps=100)
    assert not kernel.snapshot().any()


@pytest.mark.parametrize("seed", [-1, 2**64, 0.5])
def test_seed_validation(seed):
    with pytest.raises(ValueError):
        Propagation(seed=seed)


def test_mode_validation():
    with pytest.raises(ValueError):
        Simulation(8, 8, mode="unknown")
