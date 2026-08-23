"""Extract sampled frames from a video file for downstream analysis."""
from __future__ import annotations

import cv2
import numpy as np


def extract_frames(video_path: str, max_frames: int = 30) -> list[np.ndarray]:
    """Uniformly sample up to `max_frames` frames from a video (BGR numpy arrays)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, total // max_frames)

    frames = []
    idx = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img
