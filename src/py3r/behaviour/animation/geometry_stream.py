from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def undo_meta_scaling_for_geometry(
    df: pd.DataFrame, meta: dict, dims: tuple[str, ...] = ("x", "y")
) -> pd.DataFrame:
    """Return a copy with aspect-ratio and rescale-factor metadata inverted."""
    out = df.copy()
    dims_set = set(dims)
    correction = float(meta.get("aspectratio_correction", 1.0) or 1.0)
    if correction not in (0.0, 1.0):
        x_cols = [c for c in out.columns if c.endswith(".x")]
        if x_cols:
            out.loc[:, x_cols] = out.loc[:, x_cols] / correction
    factors = meta.get("rescale_factor")
    if isinstance(factors, dict):
        for dim, factor in factors.items():
            if dim not in dims_set:
                continue
            factor = float(factor)
            if factor in (0.0, 1.0):
                continue
            cols = [c for c in out.columns if c.endswith(f".{dim}")]
            if cols:
                out.loc[:, cols] = out.loc[:, cols] / factor
    return out


def _style_for_point(style: dict, point_name: str) -> dict:
    section = style.get("points", {})
    merged = {"color": (0, 255, 255), "radius": 3}
    merged.update(section.get("default", {}))
    merged.update(section.get(point_name, {}))
    merged["color"] = tuple(map(int, merged["color"]))
    merged["radius"] = int(merged["radius"])
    return merged


def _style_for_line(style: dict, line_key: tuple[str, str]) -> dict:
    section = style.get("lines", {})
    merged = {"color": (255, 255, 255), "width": 1}
    merged.update(section.get("default", {}))
    if line_key in section:
        merged.update(section[line_key])
    else:
        rev = (line_key[1], line_key[0])
        if rev in section:
            merged.update(section[rev])
    merged["color"] = tuple(map(int, merged["color"]))
    merged["width"] = int(merged["width"])
    return merged


def _style_for_boundary(style: dict, boundary_name: str) -> dict:
    section = style.get("boundaries", {})
    merged = {
        "edge_color": (0, 255, 0),
        "edge_width": 1,
        "fill_color": None,
        "fill_alpha": 0.0,
        "fill_mode": "normal",
    }
    merged.update(section.get("default", {}))
    merged.update(section.get(boundary_name, {}))
    merged["edge_color"] = (
        None if merged["edge_color"] is None else tuple(map(int, merged["edge_color"]))
    )
    merged["fill_color"] = (
        None if merged["fill_color"] is None else tuple(map(int, merged["fill_color"]))
    )
    merged["edge_width"] = int(merged["edge_width"])
    merged["fill_alpha"] = float(np.clip(merged["fill_alpha"], 0.0, 1.0))
    fill_mode = str(merged.get("fill_mode", "normal")).lower()
    if fill_mode not in {"normal", "erase"}:
        raise ValueError("boundary fill_mode must be 'normal' or 'erase'")
    merged["fill_mode"] = fill_mode
    return merged


def _is_valid(pix: np.ndarray, idx: int) -> bool:
    return idx < len(pix) and pix[idx, 0] >= 0 and pix[idx, 1] >= 0


def _compute_bounds(
    points_xy: np.ndarray, polygons_per_frame: list[list[tuple[str, np.ndarray]]], pad: float = 0.05
) -> tuple[float, float, float, float]:
    flat = points_xy.reshape(-1, 2)
    valid = np.isfinite(flat[:, 0]) & np.isfinite(flat[:, 1])
    xs = [flat[valid, 0]] if np.any(valid) else []
    ys = [flat[valid, 1]] if np.any(valid) else []
    for polys in polygons_per_frame:
        for _, poly in polys:
            if len(poly) == 0:
                continue
            ok = np.isfinite(poly[:, 0]) & np.isfinite(poly[:, 1])
            if np.any(ok):
                xs.append(poly[ok, 0])
                ys.append(poly[ok, 1])
    if not xs:
        return 0.0, 1.0, 0.0, 1.0
    xmin, xmax = float(np.min(np.concatenate(xs))), float(np.max(np.concatenate(xs)))
    ymin, ymax = float(np.min(np.concatenate(ys))), float(np.max(np.concatenate(ys)))
    if xmin == xmax:
        xmax = xmin + 1.0
    if ymin == ymax:
        ymax = ymin + 1.0
    pad = float(np.clip(pad, 0.0, 0.45))
    if pad > 0:
        xspan = xmax - xmin
        yspan = ymax - ymin
        xmin -= xspan * pad
        xmax += xspan * pad
        ymin -= yspan * pad
        ymax += yspan * pad
    return xmin, xmax, ymin, ymax


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


def _coords_to_pixels(
    coords: np.ndarray,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
    pixel_coords: bool,
) -> np.ndarray | None:
    if coords.size == 0:
        return None
    arr = np.asarray(coords, dtype=float)
    out = np.full((arr.shape[0], 2), -1, dtype=np.int32)
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    if not np.any(valid):
        return out
    x = arr[valid, 0]
    y = arr[valid, 1]
    if pixel_coords:
        xi = np.rint(x).astype(np.int32)
        yi = np.rint(y).astype(np.int32)
    else:
        xmin, xmax, ymin, ymax = bounds
        sx = max(width - 1, 1) / (xmax - xmin)
        sy = max(height - 1, 1) / (ymax - ymin)
        xi = np.rint((x - xmin) * sx).astype(np.int32)
        yi = np.rint((height - 1) - ((y - ymin) * sy)).astype(np.int32)
    inb = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    idx = np.where(valid)[0][inb]
    out[idx, 0] = xi[inb]
    out[idx, 1] = yi[inb]
    return out


def _make_projector(points_xyz: np.ndarray, view: dict | None) -> dict:
    """
    Project points with shape (n_frames, n_points, 3) to (n_frames, n_points, 2).

    Supported projections:
    - ortho: orthographic
    - persp: simple perspective camera
    """
    v = view or {}
    azim_deg = float(v.get("azim", 45.0))
    elev_deg = float(v.get("elev", 30.0))
    proj = str(v.get("proj", "ortho")).lower()
    if proj not in {"ortho", "persp"}:
        raise ValueError("view['proj'] must be 'ortho' or 'persp'")

    valid = np.isfinite(points_xyz).all(axis=2)
    if np.any(valid):
        center = np.nanmean(points_xyz[valid], axis=0)
    else:
        center = np.array([0.0, 0.0, 0.0], dtype=float)

    centered = points_xyz - center
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    cz, sz = np.cos(az), np.sin(az)
    cx, sx = np.cos(el), np.sin(el)

    # Rotate around z by azimuth, then around x by elevation.
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    if proj == "ortho":
        return {"center": center, "rotm": rz.T @ rx.T, "proj": proj}

    finite_vals = centered[np.isfinite(centered)]
    max_abs = float(np.nanmax(np.abs(finite_vals))) if finite_vals.size > 0 else 1.0
    camera_distance = float(v.get("camera_distance", max(1.0, 4.0 * max_abs)))
    focal_length = float(v.get("focal_length", camera_distance))
    return {
        "center": center,
        "rotm": rz.T @ rx.T,
        "proj": proj,
        "camera_distance": camera_distance,
        "focal_length": focal_length,
    }


def _project_xyz_with_projector(points_xyz: np.ndarray, projector: dict) -> np.ndarray:
    valid = np.isfinite(points_xyz).all(axis=2)
    centered = points_xyz - projector["center"]
    rot = centered @ projector["rotm"]
    x = rot[:, :, 0]
    y = rot[:, :, 1]
    z = rot[:, :, 2]
    if projector["proj"] == "ortho":
        xy = np.stack((x, y), axis=2)
        xy[~valid] = np.nan
        return xy
    denom = projector["camera_distance"] - z
    good = valid & np.isfinite(denom) & (denom > 1e-9)
    xp = np.full_like(x, np.nan, dtype=float)
    yp = np.full_like(y, np.nan, dtype=float)
    xp[good] = projector["focal_length"] * x[good] / denom[good]
    yp[good] = projector["focal_length"] * y[good] / denom[good]
    return np.stack((xp, yp), axis=2)


def _project_points_3d_to_2d(points_xyz: np.ndarray, view: dict | None) -> np.ndarray:
    projector = _make_projector(points_xyz, view)
    return _project_xyz_with_projector(points_xyz, projector)


def _resolve_boundary_z(name: str, boundary_z) -> float:
    if isinstance(boundary_z, dict):
        return float(boundary_z.get(name, 0.0))
    if boundary_z is None:
        return 0.0
    return float(boundary_z)


def _project_polygons_3d_to_2d(
    polygons_per_frame: list[list[tuple[str, np.ndarray]]], projector: dict, boundary_z
) -> list[list[tuple[str, np.ndarray]]]:
    projected: list[list[tuple[str, np.ndarray]]] = []
    for frame_polys in polygons_per_frame:
        frame_out: list[tuple[str, np.ndarray]] = []
        for name, poly in frame_polys:
            arr = np.asarray(poly, dtype=float)
            if arr.ndim != 2 or arr.shape[0] == 0:
                continue
            if arr.shape[1] == 2:
                z = _resolve_boundary_z(name, boundary_z)
                xyz = np.column_stack((arr[:, 0], arr[:, 1], np.full(len(arr), z, dtype=float)))
            elif arr.shape[1] == 3:
                xyz = arr
            else:
                raise ValueError("Boundary polygon must have 2 or 3 columns for projection")
            poly_xy = _project_xyz_with_projector(xyz[None, :, :], projector)[0]
            frame_out.append((name, poly_xy))
        projected.append(frame_out)
    return projected


class GeometryAnimationStream:
    """Tiny OpenCV stream: points/lines/boundaries with style dict."""

    def __init__(
        self,
        *,
        points_xy: np.ndarray,
        point_names: list[str],
        draw_point_indices: list[int],
        frame_ids: np.ndarray,
        lines_idx: list[tuple[int, int]],
        line_keys: list[tuple[str, str]],
        polygons_per_frame: list[list[tuple[str, np.ndarray]]],
        canvas_size: tuple[int, int],
        fps: float,
        bg_color: tuple[int, int, int],
        style: dict | None,
        pixel_coords: bool,
        bounds_pad: float = 0.05,
    ) -> None:
        if points_xy.ndim != 3 or points_xy.shape[2] != 2:
            raise ValueError("points_xy must be shape (n_frames, n_points, 2)")
        if len(frame_ids) != points_xy.shape[0]:
            raise ValueError("frame_ids length must match n_frames")
        if len(polygons_per_frame) != points_xy.shape[0]:
            raise ValueError("polygons_per_frame length must match n_frames")
        self._points_xy = points_xy
        self._point_names = list(point_names)
        self._draw_point_indices = list(draw_point_indices)
        self._frame_ids = np.asarray(frame_ids)
        self._lines_idx = lines_idx
        self._line_keys = line_keys
        self._polygons_per_frame = polygons_per_frame
        self._canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        self.fps = float(fps)
        self._bg_color = tuple(map(int, bg_color))
        self._style = style or {}
        self._pixel_coords = bool(pixel_coords)
        self._cursor = 0
        self._bounds = _compute_bounds(points_xy, polygons_per_frame, pad=bounds_pad)

    @property
    def frame_count(self) -> int:
        return int(self._points_xy.shape[0])

    @property
    def frame_ids(self) -> np.ndarray:
        return self._frame_ids.copy()

    def reset(self) -> None:
        self._cursor = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cursor >= self.frame_count:
            return False, None
        frame = self.get_frame(self._cursor)
        self._cursor += 1
        return True, frame

    def __iter__(self) -> GeometryAnimationStream:
        return self

    def __next__(self) -> np.ndarray:
        ok, frame = self.read()
        if not ok or frame is None:
            raise StopIteration
        return frame

    def get_frame(self, frame_idx: int) -> np.ndarray:
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(f"frame_idx {frame_idx} out of range")
        w, h = self._canvas_size
        canvas = np.full((h, w, 3), self._bg_color, dtype=np.uint8)
        return self.render_into(canvas, frame_idx=frame_idx, copy=False)

    def render_into(self, frame: np.ndarray, *, frame_idx: int, copy: bool = True) -> np.ndarray:
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(f"frame_idx {frame_idx} out of range")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (H, W, 3)")
        target = frame.copy() if copy else frame
        # Snapshot of original underlay, used by boundary fill_mode="erase".
        underlay = target.copy()

        valid_polys: list[tuple[np.ndarray, dict]] = []
        for boundary_name, poly in self._polygons_per_frame[frame_idx]:
            pix = _coords_to_pixels(
                poly, target.shape[1], target.shape[0], self._bounds, self._pixel_coords
            )
            if pix is None or len(pix) < 3:
                continue
            if np.any(pix[:, 0] < 0) or np.any(pix[:, 1] < 0):
                continue
            bstyle = _style_for_boundary(self._style, boundary_name)
            valid_polys.append((pix, bstyle))
            alpha = float(bstyle["fill_alpha"])
            if alpha > 0 and bstyle["fill_mode"] == "normal" and bstyle["fill_color"] is not None:
                fill_overlay = target.copy()
                cv2.fillPoly(fill_overlay, [pix], color=bstyle["fill_color"])
                cv2.addWeighted(fill_overlay, alpha, target, 1.0 - alpha, 0.0, dst=target)
            elif alpha > 0 and bstyle["fill_mode"] == "erase":
                # Blend back toward the original underlay only inside this polygon.
                mask = np.zeros(target.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pix], color=255)
                m = mask.astype(bool)
                target[m] = (
                    (1.0 - alpha) * target[m].astype(np.float32)
                    + alpha * underlay[m].astype(np.float32)
                ).astype(np.uint8)
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

        pts = self._points_xy[frame_idx]
        pix_pts = _coords_to_pixels(
            pts, target.shape[1], target.shape[0], self._bounds, self._pixel_coords
        )
        if pix_pts is None:
            return target
        for line_i, (i1, i2) in enumerate(self._lines_idx):
            if _is_valid(pix_pts, i1) and _is_valid(pix_pts, i2):
                lstyle = _style_for_line(self._style, self._line_keys[line_i])
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
                cv2.circle(target, tuple(p), pstyle["radius"], pstyle["color"], thickness=-1)
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


def build_geometry_stream(
    df: pd.DataFrame,
    *,
    point_names: list[str],
    lines: list[tuple[str, str]] | None = None,
    dims: tuple[str, ...] = ("x", "y"),
    view: dict | None = None,
    boundary_z: float | dict[str, float] | None = 0.0,
    frame_ids: np.ndarray | None = None,
    fps: float = 30.0,
    polygons_per_frame: list[list[tuple[str, np.ndarray]]] | None = None,
    canvas_size: tuple[int, int] = (800, 800),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    style: dict | None = None,
    pixel_coords: bool = False,
    bounds_pad: float = 0.05,
) -> GeometryAnimationStream:
    if len(dims) not in (2, 3):
        raise ValueError("dims must be length 2 or 3")
    if lines is None:
        lines = []
    if frame_ids is None:
        frame_ids = df.index.to_numpy(copy=True)
    if polygons_per_frame is None:
        polygons_per_frame = [[] for _ in range(len(df))]
    if len(polygons_per_frame) != len(df):
        raise ValueError("polygons_per_frame length must match number of frames")
    all_point_names = list(point_names)
    for p1, p2 in lines:
        if p1 not in all_point_names:
            all_point_names.append(p1)
        if p2 not in all_point_names:
            all_point_names.append(p2)

    for point in all_point_names:
        for dim in dims:
            col = f"{point}.{dim}"
            if col not in df.columns:
                raise ValueError(f"Column {col} not found")
    if len(dims) == 2:
        point_arrays = []
        for point in all_point_names:
            x = df[f"{point}.{dims[0]}"].to_numpy(dtype=float, copy=True)
            y = df[f"{point}.{dims[1]}"].to_numpy(dtype=float, copy=True)
            point_arrays.append(np.column_stack((x, y)))
        points_xy = np.stack(point_arrays, axis=1)
    else:
        point_arrays3 = []
        for point in all_point_names:
            x = df[f"{point}.{dims[0]}"].to_numpy(dtype=float, copy=True)
            y = df[f"{point}.{dims[1]}"].to_numpy(dtype=float, copy=True)
            z = df[f"{point}.{dims[2]}"].to_numpy(dtype=float, copy=True)
            point_arrays3.append(np.column_stack((x, y, z)))
        points_xyz = np.stack(point_arrays3, axis=1)
        projector = _make_projector(points_xyz, view)
        points_xy = _project_xyz_with_projector(points_xyz, projector)
        if polygons_per_frame and any(len(p) > 0 for p in polygons_per_frame):
            polygons_per_frame = _project_polygons_3d_to_2d(
                polygons_per_frame,
                projector,
                boundary_z,
            )
    point_idx = {name: i for i, name in enumerate(all_point_names)}
    draw_point_indices = [point_idx[name] for name in point_names]
    lines_idx: list[tuple[int, int]] = []
    line_keys: list[tuple[str, str]] = []
    for p1, p2 in lines:
        lines_idx.append((point_idx[p1], point_idx[p2]))
        line_keys.append((p1, p2))
    return GeometryAnimationStream(
        points_xy=points_xy,
        point_names=all_point_names,
        draw_point_indices=draw_point_indices,
        frame_ids=np.asarray(frame_ids),
        lines_idx=lines_idx,
        line_keys=line_keys,
        polygons_per_frame=polygons_per_frame,
        canvas_size=canvas_size,
        fps=fps,
        bg_color=bg_color,
        style=style,
        pixel_coords=pixel_coords,
        bounds_pad=bounds_pad,
    )
