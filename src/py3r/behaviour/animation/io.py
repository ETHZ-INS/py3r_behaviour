from __future__ import annotations

import cv2

from ._video_io import _make_video_writer, _open_video_capture, _resolve_video_fps


def play_stream(
    stream,
    *,
    fps: float | None = None,
    frame_step: int = 1,
    speed: float = 1.0,
    window_name: str = "geometry_animation",
    loop: bool = False,
    video_path: str | None = None,
    align_to_frame_ids: bool = True,
) -> None:
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    if speed <= 0:
        raise ValueError("speed must be > 0")
    playback_fps = float(stream.fps if fps is None else fps)
    delay_ms = max(1, int(round(1000.0 / (playback_fps * float(speed)))))
    idx = 0
    cap = None
    start_frame = int(stream._frame_ids[0]) if align_to_frame_ids else 0
    if video_path is not None:
        cap = _open_video_capture(video_path, start_frame=start_frame)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            if idx >= stream.frame_count:
                if not loop:
                    break
                idx = 0
                if cap is not None:
                    cap.release()
                    cap = _open_video_capture(video_path, start_frame=start_frame)
            if cap is not None:
                ok, base = cap.read()
                if not ok:
                    break
                frame = stream.render_into(base, frame_idx=idx, copy=False)
                for _ in range(frame_step - 1):
                    if not cap.grab():
                        break
            else:
                frame = stream.get_frame(idx)
            cv2.imshow(window_name, frame)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKeyEx(delay_ms)
            if key in (27, ord("q"), ord("Q")):
                break
            idx += frame_step
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def save_stream(
    stream,
    out_path: str,
    *,
    fps: float | None = None,
    frame_step: int = 1,
    video_path: str | None = None,
    align_to_frame_ids: bool = True,
    codec: str = "mp4v",
) -> None:
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    cap = None
    writer = None
    try:
        if video_path is not None:
            start_frame = int(stream._frame_ids[0]) if align_to_frame_ids else 0
            cap = _open_video_capture(video_path, start_frame=start_frame)
            ok, first = cap.read()
            if not ok:
                raise ValueError("Could not read first video frame from video_path")
            h, w = first.shape[:2]
            out_fps = _resolve_video_fps(cap, fallback=(stream.fps if fps is None else float(fps)))
            writer = _make_video_writer(out_path, width=w, height=h, fps=out_fps, codec=codec)
            idx = 0
            current = first
            while idx < stream.frame_count:
                writer.write(stream.render_into(current, frame_idx=idx, copy=True))
                idx += frame_step
                if idx >= stream.frame_count:
                    break
                for _ in range(frame_step):
                    ok, current = cap.read()
                    if not ok:
                        return
        else:
            w, h = stream._canvas_size
            out_fps = float(stream.fps if fps is None else fps)
            writer = _make_video_writer(out_path, width=w, height=h, fps=out_fps, codec=codec)
            for idx in range(0, stream.frame_count, frame_step):
                writer.write(stream.get_frame(idx))
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
