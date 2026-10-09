"""Synchronous tracking keeps each result paired with its input frame."""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision


class FaceTracker:
    def __init__(self, model: Path):
        if not model.is_file():
            raise FileNotFoundError(
                f"Model missing: {model}. Run python scripts/download_model.py."
            )
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model.resolve())),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._tracker = vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = -1

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        # Millisecond rounding must never produce duplicate VIDEO timestamps.
        self._timestamp_ms = max(timestamp_ms, self._timestamp_ms + 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._tracker.detect_for_video(image, self._timestamp_ms)
        if not result.face_landmarks:
            return None
        height, width = frame.shape[:2]
        return np.array(
            [(p.x * width, p.y * height) for p in result.face_landmarks[0]],
            dtype=np.float32,
        )

    def close(self) -> None:
        self._tracker.close()


class HandTracker:
    """Pixel landmarks from the same mirrored/unmirrored frame as the face."""

    def __init__(self, model: Path):
        if not model.is_file():
            raise FileNotFoundError(
                f"Hand model missing: {model}. Run python scripts/download_model.py --kind hand."
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model.resolve())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._tracker = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = -1
        self.handedness: list[str | None] = []

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> list[np.ndarray]:
        self._timestamp_ms = max(timestamp_ms, self._timestamp_ms + 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._tracker.detect_for_video(image, self._timestamp_ms)
        labels = getattr(result, "handedness", [])
        self.handedness = [
            label[0].category_name if label and label[0].score >= 0.7 else None
            for label in labels
        ]
        height, width = frame.shape[:2]
        return [np.array([(p.x * width, p.y * height) for p in hand], dtype=np.float32)
                for hand in result.hand_landmarks]

    def close(self) -> None:
        self._tracker.close()
