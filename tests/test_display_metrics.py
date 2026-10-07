import numpy as np
import pytest

from facial_fire.display import fit_frame, draw_hud, wrap_text
from facial_fire.metrics import FrameMetrics, FrameTiming


@pytest.mark.parametrize("width,height", [(1920, 1080), (600, 900), (640, 480)])
def test_resized_and_fullscreen_canvas_preserves_geometry_and_inverse_clicks(width, height):
    source = np.full((480, 640, 3), 100, np.uint8)
    canvas, viewport = fit_frame(source, width, height)
    assert canvas.shape == (height, width, 3)
    assert abs(viewport.width / viewport.height - 640 / 480) < 0.005
    for source_x, source_y in [(0, 0), (100, 200), (320, 240), (600, 400)]:
        displayed_x = viewport.x + source_x * viewport.width / 640
        displayed_y = viewport.y + source_y * viewport.height / 480
        mapped_x, mapped_y = viewport.frame_point(displayed_x, displayed_y)
        assert abs(mapped_x - source_x) <= 1
        assert abs(mapped_y - source_y) <= 1
    if viewport.x:
        assert viewport.frame_point(viewport.x - 1, height // 2) is None
        assert not canvas[:, :viewport.x].any()
    if viewport.y:
        assert viewport.frame_point(width // 2, viewport.y - 1) is None
        assert not canvas[:viewport.y].any()
    assert viewport.frame_point(viewport.x + viewport.width, viewport.y) is None


def test_letterboxed_ignition_lands_in_correct_native_grid_cell():
    from facial_fire._native import Simulation

    _, viewport = fit_frame(np.zeros((480, 640, 3), np.uint8), 1920, 1080)
    x, y = viewport.frame_point(960, 540)
    assert (x, y) == (320, 240)
    kernel = Simulation(17, 17)
    kernel.ignite(x / 640, y / 480, radius=0)
    assert kernel.snapshot()[8, 8] == 1
    assert np.count_nonzero(kernel.snapshot()) == 1


def test_text_wraps_for_narrow_views_without_losing_words():
    lines = ["Click/I: ignite | Space: pause | R: clear | F: fullscreen"]
    wrapped = wrap_text(lines, 220, 0.5)
    assert len(wrapped) > 1
    assert " ".join(wrapped) == lines[0]
    canvas = np.zeros((240, 320, 3), np.uint8)
    draw_hud(canvas, wrapped)
    assert canvas[:120].any()
    assert not canvas[120:].any()


def test_frame_metrics_use_completed_samples_and_actual_frame_intervals():
    metrics = FrameMetrics(window=2)
    assert "warming up" in metrics.lines()[0]
    metrics.record(FrameTiming(100, 100, 100, 100, 400), 0)
    metrics.record(FrameTiming(10, 2, 1, 3, 17), 0.1)
    metrics.record(FrameTiming(20, 4, 3, 5, 33), 0.2)
    # Old timing evicted; FPS includes the full 0.2s interval, not 1/latency.
    lines = metrics.lines()
    assert "FPS 10.0" in lines[0]
    assert "Processing 25.00 ms" in lines[0]
    assert "Tracking 15.00" in lines[1]
    assert "Mapping 3.00" in lines[1]
    assert "Simulation 2.000" in lines[1]
    assert "Rendering 4.00" in lines[1]
