"""Explicit model download; never called automatically by the webcam app."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("face", "hand"), default="face")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"models/{args.kind}_landmarker.task")
    model_url = MODEL_URL if args.kind == "face" else HAND_MODEL_URL
    if args.output.exists():
        parser.exit(0, f"Already exists: {args.output}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part")
    try:
        with urllib.request.urlopen(model_url, timeout=60) as response:
            temporary.write_bytes(response.read())
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded {args.output}")
    print(f"SHA256: {hashlib.sha256(args.output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
