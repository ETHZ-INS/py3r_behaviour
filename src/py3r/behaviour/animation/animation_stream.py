from __future__ import annotations

import cv2
import numpy as np

from ._projection import (
    _clip_axis_to_canvas,
    _compute_bounds,
    _coords_to_pixels,
    _data_to_pixel_float,
    _is_valid,
    _make_projector,
    _project_boundary_arrays_3d_to_2d,
    _project_xyz_with_projector,
)
from ._style import (
    _apply_dynamic_overrides,
    _compute_dynamic_array,
    _format_overlay_value,
    _is_dynamic_spec,
    _resolve_text_color_for_frame,
    _style_for_axis,
    _style_for_boundary,
    _style_for_line,
    _style_for_point,
    _style_for_text,
    _style_raw_for_axis,
    _style_raw_for_boundary,
    _style_raw_for_line,
    _style_raw_for_point,
    _style_raw_for_text,
)
from ._video_io import _make_video_writer, _open_video_capture, _resolve_video_fps


class AnimationStream:
    """
    OpenCV-backed frame stream for points, lines, and boundaries.

    Construct this class through `Tracking.animation_stream` and
    `Features.animation_stream`.
    Frames are generated lazily on demand, supporting both random-access
    rendering (``get_frame(i)``) and sequential playback
    (``read()``, iteration, ``play()``, ``save()``).
    """

    def __init__(
        self,
        *,
        points_xy: np.ndarray,
        point_names: list[str],
        draw_point_indices: list[int],
        frame_ids: np.ndarray,
        lines_idx: list[tuple[int, int]],
        line_keys: list[tuple[str, str]],
        boundary_arrays: list[tuple[str, np.ndarray]],
        axis_arrays: list[tuple[str, np.ndarray]] | None = None,
        text_overlays: list[tuple[str, np.ndarray | None]] | None,
        canvas_size: tuple[int, int],
        fps: float,
        bg_color: tuple[int, int, int],
        style: dict | None,
        style_sources: dict[str, np.ndarray] | None,
        pixel_coords: bool,
        bounds_pad: float = 0.05,
    ) -> None:
        if points_xy.ndim != 3 or points_xy.shape[2] != 2:
            raise ValueError("points_xy must be shape (n_frames, n_points, 2)")
        if len(frame_ids) != points_xy.shape[0]:
            raise ValueError("frame_ids length must match n_frames")
        for _, arr in boundary_arrays:
            barr = np.asarray(arr)
            if barr.ndim != 3 or barr.shape[0] != points_xy.shape[0] or barr.shape[2] != 2:
                raise ValueError("Boundary arrays must have shape (n_frames, n_vertices, 2)")
        for _, arr in axis_arrays or []:
            aarr = np.asarray(arr)
            n = points_xy.shape[0]
            if aarr.ndim != 3 or aarr.shape[0] != n or aarr.shape[1] != 2 or aarr.shape[2] != 2:
                raise ValueError("Axis arrays must have shape (n_frames, 2, 2)")
        if text_overlays is None:
            text_overlays = []
        for label, values in text_overlays:
            if values is None:
                continue
            if len(values) != points_xy.shape[0]:
                raise ValueError(
                    f"text overlay '{label}' length must match n_frames ({points_xy.shape[0]})"
                )
        self._points_xy = points_xy
        self._point_names = list(point_names)
        self._draw_point_indices = list(draw_point_indices)
        self._frame_ids = np.asarray(frame_ids)
        self._lines_idx = lines_idx
        self._line_keys = line_keys
        self._boundary_arrays = [
            (str(name), np.asarray(arr, dtype=float)) for name, arr in boundary_arrays
        ]
        self._axis_arrays = [
            (str(name), np.asarray(arr, dtype=float)) for name, arr in (axis_arrays or [])
        ]
        self._style = style or {}
        self._style_sources = style_sources or {}
        self._text_overlays = []
        for label, values in text_overlays:
            if values is None:
                self._text_overlays.append((str(label), None))
            else:
                self._text_overlays.append((str(label), np.asarray(values)))
        self._dynamic_styles: dict[str, dict[object, dict[str, np.ndarray]]] = {
            "points": {},
            "lines": {},
            "boundaries": {},
            "axes": {},
            "text": {},
        }
        n_frames = points_xy.shape[0]

        def _populate(section_name: str, item_key, merged: dict):
            item_dyn: dict[str, np.ndarray] = {}
            for prop, value in merged.items():
                if not _is_dynamic_spec(value):
                    continue
                source_name = str(value["from"])
                if source_name not in self._style_sources:
                    raise ValueError(
                        f"Dynamic style for {section_name}.{item_key}.{prop} references "
                        f"unknown source '{source_name}'"
                    )
                item_dyn[prop] = _compute_dynamic_array(
                    value,
                    np.asarray(self._style_sources[source_name]),
                    n_frames,
                    prop_name=prop,
                    style_path=f"{section_name}.{item_key}.{prop}",
                )
            if item_dyn:
                self._dynamic_styles[section_name][item_key] = item_dyn

        for p in self._point_names:
            _populate("points", p, _style_raw_for_point(self._style, p))
        for lk in self._line_keys:
            _populate("lines", lk, _style_raw_for_line(self._style, lk))
        boundary_names = {name for name, _ in self._boundary_arrays}
        for b in boundary_names:
            _populate("boundaries", b, _style_raw_for_boundary(self._style, b))
        for ax_name, _ in self._axis_arrays:
            _populate("axes", ax_name, _style_raw_for_axis(self._style, ax_name))
        for label, values in self._text_overlays:
            if values is None:
                continue
            _populate("text", label, _style_raw_for_text(self._style, label))

        self._canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        self.fps = float(fps)
        self._bg_color = tuple(map(int, bg_color))
        self._pixel_coords = bool(pixel_coords)
        self._cursor = 0
        self._bounds = _compute_bounds(
            points_xy, self._boundary_arrays, self._axis_arrays, pad=bounds_pad
        )

    @property
    def frame_count(self) -> int:
        """Number of renderable frames."""
        return int(self._points_xy.shape[0])

    @property
    def frame_ids(self) -> np.ndarray:
        """Copy of source frame identifiers aligned to stream indices."""
        return self._frame_ids.copy()

    def reset(self) -> None:
        """
        Reset the internal sequential cursor.

        Examples
        --------
            ```pycon
            >>> from py3r.behaviour.util.docdata import data_path
            >>> from py3r.behaviour.tracking.tracking import Tracking
            >>> with data_path("py3r.behaviour.tracking._data", "dlc_single.csv") as p:
            ...     t = Tracking.from_dlc(str(p), handle="ex", fps=30)
            >>> s = t.animation_stream(points=["p1"], pixel_coords=True, canvas_size=(64, 48))
            >>> _ = s.read()
            >>> s.reset()
            >>> ok, _ = s.read()
            >>> ok
            True

            ```
        """
        self._cursor = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Return the next rendered frame using VideoCapture-style semantics.

        Returns
        -------
            tuple[bool, np.ndarray | None]: ``(True, frame)`` while frames remain;
            otherwise ``(False, None)``.

        Examples
        --------
            ```pycon
            >>> from py3r.behaviour.util.docdata import data_path
            >>> from py3r.behaviour.tracking.tracking import Tracking
            >>> with data_path("py3r.behaviour.tracking._data", "dlc_single.csv") as p:
            ...     t = Tracking.from_dlc(str(p), handle="ex", fps=30)
            >>> s = t.animation_stream(points=["p1"], pixel_coords=True, canvas_size=(64, 48))
            >>> ok, frame = s.read()
            >>> ok and frame is not None
            True

            ```
        """
        if self._cursor >= self.frame_count:
            return False, None
        frame = self.get_frame(self._cursor)
        self._cursor += 1
        return True, frame

    def __iter__(self) -> AnimationStream:
        return self

    def __next__(self) -> np.ndarray:
        """Iterate frames sequentially; raises ``StopIteration`` at end."""
        ok, frame = self.read()
        if not ok or frame is None:
            raise StopIteration
        return frame

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Render and return one frame by stream index.

        Args:
            frame_idx: Zero-based frame index.

        Returns
        -------
            np.ndarray: Rendered BGR image with shape ``(H, W, 3)``.

        Raises
        ------
            IndexError: If ``frame_idx`` is out of range.

        Examples
        --------
            ```pycon
            >>> from py3r.behaviour.util.docdata import data_path
            >>> from py3r.behaviour.tracking.tracking import Tracking
            >>> with data_path("py3r.behaviour.tracking._data", "dlc_single.csv") as p:
            ...     t = Tracking.from_dlc(str(p), handle="ex", fps=30)
            >>> s = t.animation_stream(points=["p1"], pixel_coords=True, canvas_size=(64, 48))
            >>> frame0 = s.get_frame(0)
            >>> frame0.ndim
            3

            ```
        """
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(f"frame_idx {frame_idx} out of range")
        w, h = self._canvas_size
        canvas = np.full((h, w, 3), self._bg_color, dtype=np.uint8)
        return self.render_into(canvas, frame_idx=frame_idx, copy=False)

    def render_into(self, frame: np.ndarray, *, frame_idx: int, copy: bool = True) -> np.ndarray:
        """
        Draw stream geometry into an existing frame buffer.

        Args:
            frame: Base BGR image buffer with shape ``(H, W, 3)``.
            frame_idx: Stream frame index to render.
            copy: If ``True``, draw into a copy. If ``False``, draw in-place.

        Returns
        -------
            np.ndarray: Rendered frame buffer.

        Raises
        ------
            IndexError: If ``frame_idx`` is out of range.
            ValueError: If ``frame`` does not have shape ``(H, W, 3)``.

        Examples
        --------
            ```pycon
            >>> import numpy as np
            >>> from py3r.behaviour.util.docdata import data_path
            >>> from py3r.behaviour.tracking.tracking import Tracking
            >>> with data_path("py3r.behaviour.tracking._data", "dlc_single.csv") as p:
            ...     t = Tracking.from_dlc(str(p), handle="ex", fps=30)
            >>> s = t.animation_stream(points=["p1"], pixel_coords=True, canvas_size=(32, 24))
            >>> base = np.zeros((24, 32, 3), dtype=np.uint8)
            >>> out = s.render_into(base, frame_idx=0, copy=True)
            >>> out.shape
            (24, 32, 3)

            ```
        """
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(f"frame_idx {frame_idx} out of range")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (H, W, 3)")
        target = frame.copy() if copy else frame
        # Snapshot of original underlay, used by boundary fill_mode="erase".
        underlay = target.copy()

        valid_polys: list[tuple[np.ndarray, dict]] = []
        for boundary_name, arr in self._boundary_arrays:
            poly = arr[frame_idx]
            pix = _coords_to_pixels(
                poly, target.shape[1], target.shape[0], self._bounds, self._pixel_coords
            )
            if pix is None or len(pix) < 3:
                continue
            if np.any(pix[:, 0] < 0) or np.any(pix[:, 1] < 0):
                continue
            bstyle = _style_for_boundary(self._style, boundary_name)
            bstyle = _apply_dynamic_overrides(
                bstyle, self._dynamic_styles["boundaries"].get(boundary_name), frame_idx
            )
            valid_polys.append((pix, bstyle))
            alpha = float(bstyle["fill_alpha"])
            if alpha > 0 and bstyle["fill_mode"] == "normal" and bstyle["fill_color"] is not None:
                fill_overlay = target.copy()
                cv2.fillPoly(fill_overlay, [pix], color=bstyle["fill_color"])
                cv2.addWeighted(fill_overlay, alpha, target, 1.0 - alpha, 0.0, dst=target)
            elif bstyle["fill_mode"] == "erase":
                # Hard erase using ROI + OpenCV masked copy (no NumPy boolean indexing).
                pts = np.asarray(pix, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                if w <= 0 or h <= 0:
                    continue
                img_h, img_w = target.shape[:2]
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(img_w, x + w)
                y1 = min(img_h, y + h)
                if x1 <= x0 or y1 <= y0:
                    continue
                local_pts = pts - np.array([x, y], dtype=np.int32)
                mask_full = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_full, [local_pts], color=255)
                dx0 = x0 - x
                dy0 = y0 - y
                dx1 = dx0 + (x1 - x0)
                dy1 = dy0 + (y1 - y0)
                mask_roi = mask_full[dy0:dy1, dx0:dx1]
                if mask_roi.size == 0:
                    continue
                underlay_roi = underlay[y0:y1, x0:x1]
                target_roi = target[y0:y1, x0:x1]
                cv2.copyTo(underlay_roi, mask_roi, target_roi)
        for pix, bstyle in valid_polys:
            edge_width = int(bstyle["edge_width"])
            edge_color = bstyle["edge_color"]
            if edge_width > 0 and edge_color is not None:
                cv2.polylines(
                    target,
                    [pix],
                    isClosed=True,
                    color=edge_color,
                    thickness=edge_width,
                )

        w, h = target.shape[1], target.shape[0]
        for ax_name, ax_arr in self._axis_arrays:
            seg = ax_arr[frame_idx]  # (2, 2)
            if not np.all(np.isfinite(seg)):
                continue
            pix_float = _data_to_pixel_float(seg, w, h, self._bounds, self._pixel_coords)
            clipped = _clip_axis_to_canvas(pix_float[0], pix_float[1], w, h)
            if clipped is None:
                continue
            cp1, cp2 = clipped
            axstyle = _style_for_axis(self._style, ax_name)
            axstyle = _apply_dynamic_overrides(
                axstyle, self._dynamic_styles["axes"].get(ax_name), frame_idx
            )
            edge_width = int(axstyle["edge_width"])
            edge_color = axstyle["edge_color"]
            if edge_width > 0 and edge_color is not None:
                cv2.line(target, tuple(cp1), tuple(cp2), edge_color, edge_width)

        pts = self._points_xy[frame_idx]
        pix_pts = _coords_to_pixels(
            pts, target.shape[1], target.shape[0], self._bounds, self._pixel_coords
        )
        if pix_pts is not None:
            for line_i, (i1, i2) in enumerate(self._lines_idx):
                if _is_valid(pix_pts, i1) and _is_valid(pix_pts, i2):
                    lstyle = _style_for_line(self._style, self._line_keys[line_i])
                    lstyle = _apply_dynamic_overrides(
                        lstyle,
                        self._dynamic_styles["lines"].get(self._line_keys[line_i]),
                        frame_idx,
                    )
                    cv2.line(
                        target,
                        tuple(pix_pts[i1]),
                        tuple(pix_pts[i2]),
                        lstyle["color"],
                        lstyle["width"],
                    )
            for point_i in self._draw_point_indices:
                p = pix_pts[point_i]
                if p[0] >= 0:
                    pstyle = _style_for_point(self._style, self._point_names[point_i])
                    pstyle = _apply_dynamic_overrides(
                        pstyle,
                        self._dynamic_styles["points"].get(self._point_names[point_i]),
                        frame_idx,
                    )
                    cv2.circle(target, tuple(p), pstyle["radius"], pstyle["color"], thickness=-1)

        if self._text_overlays:
            text_section = self._style.get("text", {})
            ox, oy = text_section.get("origin", (10, 20))
            default_fmt = str(text_section.get("format", ".3f"))
            y = int(oy)
            entries: list[dict] = []
            max_text_w = 0
            for label, values in self._text_overlays:
                tstyle = _style_for_text(self._style, label)
                tstyle = _apply_dynamic_overrides(
                    tstyle,
                    self._dynamic_styles["text"].get(label),
                    frame_idx,
                )
                if y < 0:
                    y += tstyle["line_height"]
                if values is None:
                    entries.append({"y": y, "line_height": tstyle["line_height"], "spacer": True})
                    y += tstyle["line_height"]
                    continue
                fmt = default_fmt if tstyle.get("format") is None else str(tstyle["format"])
                color = _resolve_text_color_for_frame(tstyle, np.asarray(values), frame_idx)
                txt = (
                    f"{label}: "
                    f"{_format_overlay_value(values[frame_idx], fmt, as_bool=tstyle['as_bool'])}"
                )
                (tw, _), _ = cv2.getTextSize(
                    txt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    tstyle["font_scale"],
                    tstyle["thickness"],
                )
                max_text_w = max(max_text_w, int(tw))
                entries.append(
                    {
                        "y": y,
                        "line_height": tstyle["line_height"],
                        "spacer": False,
                        "txt": txt,
                        "style": tstyle,
                        "color": color,
                    }
                )
                y += tstyle["line_height"]

            panel_cfg = text_section.get("panel", {})
            panel_enabled = (
                bool(panel_cfg.get("enabled", False))
                if isinstance(panel_cfg, dict)
                else bool(panel_cfg)
            )
            if panel_enabled and entries and max_text_w > 0:
                pad = int(panel_cfg.get("padding", 6)) if isinstance(panel_cfg, dict) else 6
                alpha = float(panel_cfg.get("alpha", 0.45)) if isinstance(panel_cfg, dict) else 0.45
                alpha = float(np.clip(alpha, 0.0, 1.0))
                panel_color = (
                    tuple(map(int, panel_cfg.get("color", (0, 0, 0))))
                    if isinstance(panel_cfg, dict)
                    else (0, 0, 0)
                )
                top = min(int(e["y"] - e["line_height"] + 4) for e in entries)
                bottom = max(int(e["y"] + 6) for e in entries)
                left = int(ox) - pad
                right = int(ox) + max_text_w + pad
                left = max(0, left)
                right = min(target.shape[1] - 1, right)
                top = max(0, top)
                bottom = min(target.shape[0] - 1, bottom)
                if right > left and bottom > top:
                    if alpha >= 1.0:
                        cv2.rectangle(
                            target,
                            (left, top),
                            (right, bottom),
                            panel_color,
                            thickness=-1,
                        )
                    elif alpha > 0:
                        panel_overlay = target.copy()
                        cv2.rectangle(
                            panel_overlay,
                            (left, top),
                            (right, bottom),
                            panel_color,
                            thickness=-1,
                        )
                        cv2.addWeighted(panel_overlay, alpha, target, 1.0 - alpha, 0.0, dst=target)

            for e in entries:
                if e["spacer"]:
                    continue
                tstyle = e["style"]
                if tstyle["outline_thickness"] > 0:
                    cv2.putText(
                        target,
                        e["txt"],
                        (int(ox), int(e["y"])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        tstyle["font_scale"],
                        tstyle["outline_color"],
                        tstyle["outline_thickness"],
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    target,
                    e["txt"],
                    (int(ox), int(e["y"])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    tstyle["font_scale"],
                    e["color"],
                    tstyle["thickness"],
                    cv2.LINE_AA,
                )
        return target

    def play(
        self,
        *,
        fps: float | None = None,
        frame_step: int = 1,
        speed: float = 1.0,
        window_name: str = "geometry_animation",
        loop: bool = False,
        video_path: str | None = None,
        align_to_frame_ids: bool = True,
    ) -> None:
        """
        Play stream in an OpenCV window.

        Press ``q`` or ``Esc`` to exit playback.

        Args:
            fps: Playback FPS. Uses stream FPS when ``None``.
            frame_step: Number of stream frames to advance per displayed frame.
            speed: Playback speed multiplier.
            window_name: OpenCV window name.
            loop: If ``True``, restart at the end.
            video_path: Optional source video to draw overlays on.
            align_to_frame_ids: If ``True``, seek source video to first ``frame_id``.

        Raises
        ------
            ValueError: If ``frame_step < 1`` or ``speed <= 0``.

        Examples
        --------
            ```pycon
            >>> from py3r.behaviour.util.docdata import data_path
            >>> from py3r.behaviour.tracking.tracking import Tracking
            >>> with data_path("py3r.behaviour.tracking._data", "dlc_single.csv") as p:
            ...     t = Tracking.from_dlc(str(p), handle="ex", fps=30)
            >>> s = t.animation_stream(points=["p1"], pixel_coords=True, canvas_size=(64, 48))
            >>> s.play(loop=False, speed=1.0)  # xdoctest: +SKIP

            ```
        """
        if frame_step < 1:
            raise ValueError("frame_step must be >= 1")
        if speed <= 0:
            raise ValueError("speed must be > 0")
        playback_fps = float(self.fps if fps is None else fps)
        delay_ms = max(1, int(round(1000.0 / (playback_fps * float(speed)))))
        idx = 0
        cap = None
        start_frame = int(self._frame_ids[0]) if align_to_frame_ids else 0
        if video_path is not None:
            cap = _open_video_capture(video_path, start_frame=start_frame)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            while True:
                if idx >= self.frame_count:
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
                    frame = self.render_into(base, frame_idx=idx, copy=False)
                    for _ in range(frame_step - 1):
                        if not cap.grab():
                            break
                else:
                    frame = self.get_frame(idx)
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

    def save(
        self,
        out_path: str,
        *,
        fps: float | None = None,
        frame_step: int = 1,
        video_path: str | None = None,
        align_to_frame_ids: bool = True,
        codec: str = "mp4v",
    ) -> None:
        """
        Render stream to a video file.

        If ``video_path`` is provided, geometry is composited onto decoded frames.

        Args:
            out_path: Output video path.
            fps: Output FPS when rendering without ``video_path``.
            frame_step: Number of stream frames to skip per output frame.
            video_path: Optional source video to draw overlays on.
            align_to_frame_ids: If ``True``, seek source video to first ``frame_id``.
            codec: FourCC codec string (for example ``"mp4v"``).

        Raises
        ------
            ValueError: If ``frame_step < 1`` or writer/capture cannot be opened.

        Examples
        --------
            ```pycon
            >>> import tempfile
            >>> from py3r.behaviour.util.docdata import data_path
            >>> from py3r.behaviour.tracking.tracking import Tracking
            >>> with data_path("py3r.behaviour.tracking._data", "dlc_single.csv") as p:
            ...     t = Tracking.from_dlc(str(p), handle="ex", fps=30)
            >>> s = t.animation_stream(points=["p1"], pixel_coords=True, canvas_size=(64, 48))
            >>> with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            ...     s.save(f.name)  # xdoctest: +SKIP

            ```
        """
        if frame_step < 1:
            raise ValueError("frame_step must be >= 1")
        cap = None
        writer = None
        try:
            if video_path is not None:
                start_frame = int(self._frame_ids[0]) if align_to_frame_ids else 0
                cap = _open_video_capture(video_path, start_frame=start_frame)
                ok, first = cap.read()
                if not ok:
                    raise ValueError("Could not read first video frame from video_path")
                h, w = first.shape[:2]
                out_fps = _resolve_video_fps(
                    cap, fallback=(self.fps if fps is None else float(fps))
                )
                writer = _make_video_writer(out_path, width=w, height=h, fps=out_fps, codec=codec)
                idx = 0
                current = first
                while idx < self.frame_count:
                    writer.write(self.render_into(current, frame_idx=idx, copy=True))
                    idx += frame_step
                    if idx >= self.frame_count:
                        break
                    for _ in range(frame_step):
                        ok, current = cap.read()
                        if not ok:
                            return
            else:
                w, h = self._canvas_size
                out_fps = float(self.fps if fps is None else fps)
                writer = _make_video_writer(out_path, width=w, height=h, fps=out_fps, codec=codec)
                for idx in range(0, self.frame_count, frame_step):
                    writer.write(self.get_frame(idx))
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()


def build_animation_stream(
    *,
    points: np.ndarray,
    point_names: list[str],
    draw_points: list[str] | None = None,
    lines: list[tuple[str, str]] | None = None,
    view: dict | None = None,
    boundary_z: float | dict[str, float] | None = 0.0,
    frame_ids: np.ndarray,
    fps: float = 30.0,
    boundary_arrays: list[tuple[str, np.ndarray]] | None = None,
    axis_arrays: list[tuple[str, np.ndarray]] | None = None,
    canvas_size: tuple[int, int] = (800, 800),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    style: dict | None = None,
    style_sources: dict[str, np.ndarray] | None = None,
    text_overlays: list[tuple[str, np.ndarray | None]] | None = None,
    pixel_coords: bool = False,
    bounds_pad: float = 0.05,
) -> AnimationStream:
    """
    Build stream from precomputed point arrays.

    Parameters
    ----------
    points : np.ndarray
        Shape ``(n_frames, n_points, 2|3)``. If 3D, points are projected using
        ``view`` and optional ``boundary_z``.
    point_names : list[str]
        Names for the second axis in ``points``.
    draw_points : list[str], optional
        Subset of ``point_names`` to draw as circles.
    lines : list[tuple[str, str]], optional
        Point-pair line segments.
    frame_ids : np.ndarray
        Source frame identifiers aligned to stream rows.

    Returns
    -------
    AnimationStream
        Lazy stream object supporting ``read()``, ``get_frame()``, ``play()``,
        and ``save()``.

    Examples
    --------
    ```pycon
    >>> from py3r.behaviour.util.docdata import data_path
    >>> from py3r.behaviour.tracking.tracking import Tracking
    >>> with data_path("py3r.behaviour.tracking._data", "dlc_single.csv") as p:
    ...     t = Tracking.from_dlc(str(p), handle="ex", fps=30)
    >>> point_names, points = t.points_to_numpy(["p1", "p2"], dims=("x", "y"))
    >>> stream = build_animation_stream(
    ...     points=points,
    ...     point_names=point_names,
    ...     draw_points=["p1"],
    ...     lines=[("p1", "p2")],
    ...     frame_ids=t.data.index.to_numpy(copy=True),
    ...     pixel_coords=True,
    ... )
    >>> stream.frame_count
    5

    ```
    """
    if lines is None:
        lines = []
    if points.ndim != 3 or points.shape[2] not in (2, 3):
        raise ValueError("points must have shape (n_frames, n_points, 2|3)")
    if points.shape[1] != len(point_names):
        raise ValueError("point_names length must match points.shape[1]")
    if len(frame_ids) != points.shape[0]:
        raise ValueError("frame_ids length must match points.shape[0]")
    if boundary_arrays is None:
        boundary_arrays = []
    for _, arr in boundary_arrays:
        barr = np.asarray(arr)
        if barr.ndim != 3 or barr.shape[0] != points.shape[0] or barr.shape[2] not in (2, 3):
            raise ValueError("Boundary arrays must have shape (n_frames, n_vertices, 2|3)")
    if axis_arrays is None:
        axis_arrays = []
    for _, arr in axis_arrays:
        aarr = np.asarray(arr)
        n = points.shape[0]
        if aarr.ndim != 3 or aarr.shape[0] != n or aarr.shape[1] != 2 or aarr.shape[2] != 2:
            raise ValueError("Axis arrays must have shape (n_frames, 2, 2)")
    if text_overlays is None:
        text_overlays = []

    if points.shape[2] == 3:
        projector = _make_projector(points, view)
        points_xy = _project_xyz_with_projector(points, projector)
        if boundary_arrays:
            boundary_arrays = _project_boundary_arrays_3d_to_2d(
                boundary_arrays,
                projector,
                boundary_z,
                points.shape[0],
            )
    else:
        points_xy = points.astype(float, copy=True)
        boundary_arrays = [
            (str(name), np.asarray(arr, dtype=float)) for name, arr in boundary_arrays
        ]

    point_idx = {name: i for i, name in enumerate(point_names)}
    draw_points = point_names if draw_points is None else draw_points
    draw_point_indices = [point_idx[name] for name in draw_points]
    lines_idx: list[tuple[int, int]] = []
    line_keys: list[tuple[str, str]] = []
    for p1, p2 in lines:
        if p1 not in point_idx or p2 not in point_idx:
            raise ValueError(f"Unknown point in line ({p1}, {p2})")
        lines_idx.append((point_idx[p1], point_idx[p2]))
        line_keys.append((p1, p2))

    return AnimationStream(
        points_xy=points_xy,
        point_names=point_names,
        draw_point_indices=draw_point_indices,
        frame_ids=np.asarray(frame_ids),
        lines_idx=lines_idx,
        line_keys=line_keys,
        boundary_arrays=boundary_arrays,
        axis_arrays=axis_arrays,
        text_overlays=text_overlays,
        canvas_size=canvas_size,
        fps=fps,
        bg_color=bg_color,
        style=style,
        style_sources=style_sources,
        pixel_coords=pixel_coords,
        bounds_pad=bounds_pad,
    )
