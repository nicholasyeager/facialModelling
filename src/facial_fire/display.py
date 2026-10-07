"""Aspect-preserving presentation and inverse mapping for HighGUI clicks."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Viewport:
    x: int
    y: int
    width: int
    height: int
    source_width: int
    source_height: int

    def frame_point(self, x, y):
        if not (self.x <= x < self.x + self.width and self.y <= y < self.y + self.height):
            return None
        return (int((x - self.x) * self.source_width / self.width),
                int((y - self.y) * self.source_height / self.height))


def fit_frame(frame, width, height):
    """Letterbox the camera image rather than stretching a face in fullscreen."""
    if width < 1 or height < 1:
        raise ValueError("Display dimensions must be positive")
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    fitted_width = min(width, max(1, round(source_width * scale)))
    fitted_height = min(height, max(1, round(source_height * scale)))
    x, y = (width - fitted_width) // 2, (height - fitted_height) // 2
    canvas = np.zeros((height, width, 3), np.uint8)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    canvas[y:y + fitted_height, x:x + fitted_width] = cv2.resize(
        frame, (fitted_width, fitted_height), interpolation=interpolation,
    )
    return canvas, Viewport(x, y, fitted_width, fitted_height, source_width, source_height)


def wrap_text(lines, width, scale):
    wrapped = []
    for line in lines:
        current = ""
        for word in line.split():
            candidate = f"{current} {word}".strip()
            if current and cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        if current:
            wrapped.append(current)
    return wrapped


def draw_hud(canvas, lines):
    height, width = canvas.shape[:2]
    scale = min(0.7, max(0.3, min(width / 1600, height / 1000)))
    margin = max(4, round(12 * scale / 0.6))
    line_height = max(12, round(30 * scale))
    wrapped = wrap_text(lines, max(1, width - 2 * margin), scale)
    max_lines = max(1, (height // 2 - 2 * margin) // line_height)
    wrapped = wrapped[:max_lines]
    panel_height = min(height, (len(wrapped) + 1) * line_height)
    panel = canvas[:panel_height]
    cv2.addWeighted(panel, 0.18, np.full_like(panel, 245), 0.82, 0, dst=panel)
    for index, line in enumerate(wrapped):
        cv2.putText(canvas, line, (margin, (index + 1) * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


class DemoWindow:
    def __init__(self, name, width, height, fullscreen=False):
        self.name = name
        self.fullscreen = False
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, width, height)
        if fullscreen:
            self.toggle_fullscreen()

    def toggle_fullscreen(self):
        target = not self.fullscreen
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if target else cv2.WINDOW_NORMAL)
        self.fullscreen = target

    def size(self, fallback_width, fallback_height):
        try:
            _, _, width, height = cv2.getWindowImageRect(self.name)
            if width > 0 and height > 0:
                return width, height
        except cv2.error:
            pass  # Some HighGUI backends do not expose window geometry.
        return fallback_width, fallback_height

    def is_open(self):
        try:
            return cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False  # Closing a native window can remove its HighGUI record.
