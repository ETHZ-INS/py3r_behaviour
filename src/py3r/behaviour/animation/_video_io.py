from __future__ import annotations

import cv2
import numpy as np


def _open_video_capture(video_path: str, *, start_frame: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    return cap


def _resolve_video_fps(cap: cv2.VideoCapture, *, fallback: float) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if np.isfinite(fps) and fps > 0:
        return fps
    return float(fallback)


def _make_video_writer(
    out_path: str, *, width: int, height: int, fps: float, codec: str
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer for: {out_path}")
    return writer
