"""Stage 1: one webcam, one face, one movable canonical color patch."""

import argparse
from pathlib import Path
import time

import cv2

from .mapping import FaceMapping
from .rendering import blend_effect, make_preview
from .tracking import FaceTracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()
    if min(args.width, args.height) < 1 or args.grid_size < 2:
        parser.error("Capture dimensions must be positive and grid size >= 2")

    tracker = None
    camera = None
    window = "Facial Fire - Stage 1"
    mapping = None
    preview = make_preview(args.grid_size)
    enabled = True
    debug = False

    def on_mouse(event, x, y, flags, userdata):
        nonlocal preview
        if event == cv2.EVENT_LBUTTONDOWN and mapping is not None:
            if 0 <= y < mapping.mask.shape[0] and 0 <= x < mapping.mask.shape[1]:
                if mapping.mask[y, x]:
                    uv = mapping.canonical_point(x, y)
                    if (uv >= 0).all() and (uv <= 1).all():
                        preview = make_preview(args.grid_size, center=uv)

    try:
        tracker = FaceTracker(args.model)
        camera = cv2.VideoCapture(args.camera)
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera}; check Windows camera permissions.")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, on_mouse)
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Webcam frame capture failed; camera may be disconnected.")
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            points = tracker.detect(frame, time.perf_counter_ns() // 1_000_000)
            mapping = None if points is None else FaceMapping.from_landmarks(points, frame.shape)
            display = blend_effect(frame, preview, mapping) if mapping is not None and enabled else frame.copy()
            if debug and mapping is not None:
                contours, _ = cv2.findContours(mapping.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, contours, -1, (0, 255, 0), 2)
            status = "Face tracked" if mapping is not None else "No face - overlay hidden"
            cv2.putText(display, status, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(display, "Click: move tint | O: toggle | R: reset | D: outline | Q: quit",
                        (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("o"):
                enabled = not enabled
            elif key == ord("r"):
                preview = make_preview(args.grid_size)
            elif key == ord("d"):
                debug = not debug
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    finally:
        if camera is not None:
            camera.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
