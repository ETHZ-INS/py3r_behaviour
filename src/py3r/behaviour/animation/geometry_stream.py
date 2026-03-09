from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

try:
    import matplotlib as mpl
    import matplotlib.cm as mpl_cm
except Exception:  # pragma: no cover - optional dependency
    mpl_cm = None
    mpl = None


def _is_dynamic_spec(value) -> bool:
    return isinstance(value, dict) and "from" in value and ("map" in value or "cmap" in value)


def collect_dynamic_source_names_from_style(style: dict | None) -> set[str]:
    names: set[str] = set()

    def _walk(node):
        if _is_dynamic_spec(node):
            names.add(str(node["from"]))
            return
        if isinstance(node, dict):
            for v in node.values():
                _walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _walk(v)

    _walk(style or {})
    return names


def _style_raw_for_point(style: dict, point_name: str) -> dict:
    section = style.get("points", {})
    merged = {"color": (0, 255, 255), "radius": 3}
    merged.update(section.get("default", {}))
    merged.update(section.get(point_name, {}))
    return merged


def _style_for_point(style: dict, point_name: str) -> dict:
    merged = _style_raw_for_point(style, point_name)
    if _is_dynamic_spec(merged.get("color")):
        merged["color"] = (0, 255, 255)
    if _is_dynamic_spec(merged.get("radius")):
        merged["radius"] = 3
    merged["color"] = tuple(map(int, merged["color"]))
    merged["radius"] = int(merged["radius"])
    return merged


def _style_raw_for_line(style: dict, line_key: tuple[str, str]) -> dict:
    section = style.get("lines", {})
    merged = {"color": (255, 255, 255), "width": 1}
    merged.update(section.get("default", {}))
    if line_key in section:
        merged.update(section[line_key])
    else:
        rev = (line_key[1], line_key[0])
        if rev in section:
            merged.update(section[rev])
    return merged


def _style_for_line(style: dict, line_key: tuple[str, str]) -> dict:
    merged = _style_raw_for_line(style, line_key)
    if _is_dynamic_spec(merged.get("color")):
        merged["color"] = (255, 255, 255)
    if _is_dynamic_spec(merged.get("width")):
        merged["width"] = 1
    merged["color"] = tuple(map(int, merged["color"]))
    merged["width"] = int(merged["width"])
    return merged


def _style_raw_for_boundary(style: dict, boundary_name: str) -> dict:
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
    return merged


def _style_for_boundary(style: dict, boundary_name: str) -> dict:
    merged = _style_raw_for_boundary(style, boundary_name)
    if _is_dynamic_spec(merged.get("edge_color")):
        merged["edge_color"] = (0, 255, 0)
    if _is_dynamic_spec(merged.get("fill_color")):
        merged["fill_color"] = None
    if _is_dynamic_spec(merged.get("edge_width")):
        merged["edge_width"] = 1
    if _is_dynamic_spec(merged.get("fill_alpha")):
        merged["fill_alpha"] = 0.0
    if _is_dynamic_spec(merged.get("fill_mode")):
        merged["fill_mode"] = "normal"
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


def _style_raw_for_text(style: dict, label: str) -> dict:
    section = style.get("text", {})
    merged = {
        "color": (255, 255, 255),
        "font_scale": 0.5,
        "thickness": 1,
        "outline_color": (0, 0, 0),
        "outline_thickness": 2,
        "line_height": 18,
        "format": None,
        "as_bool": False,
        "cmap": None,
        "vmin": None,
        "vmax": None,
        "nan_color": None,
    }
    merged.update(section.get("default", {}))
    merged.update(section.get(label, {}))
    return merged


def _style_for_text(style: dict, label: str) -> dict:
    merged = _style_raw_for_text(style, label)
    if _is_dynamic_spec(merged.get("color")):
        merged["color"] = (255, 255, 255)
    if _is_dynamic_spec(merged.get("outline_color")):
        merged["outline_color"] = (0, 0, 0)
    if _is_dynamic_spec(merged.get("font_scale")):
        merged["font_scale"] = 0.5
    if _is_dynamic_spec(merged.get("thickness")):
        merged["thickness"] = 1
    if _is_dynamic_spec(merged.get("outline_thickness")):
        merged["outline_thickness"] = 2
    if _is_dynamic_spec(merged.get("line_height")):
        merged["line_height"] = 18
    if _is_dynamic_spec(merged.get("format")):
        merged["format"] = None
    if _is_dynamic_spec(merged.get("as_bool")):
        merged["as_bool"] = False
    if _is_dynamic_spec(merged.get("nan_color")):
        merged["nan_color"] = None
    if _is_dynamic_spec(merged.get("cmap")):
        merged["cmap"] = None
    if _is_dynamic_spec(merged.get("vmin")):
        merged["vmin"] = None
    if _is_dynamic_spec(merged.get("vmax")):
        merged["vmax"] = None
    merged["color"] = tuple(map(int, merged["color"]))
    merged["outline_color"] = tuple(map(int, merged["outline_color"]))
    merged["font_scale"] = float(merged["font_scale"])
    merged["thickness"] = int(merged["thickness"])
    merged["outline_thickness"] = int(merged["outline_thickness"])
    merged["line_height"] = int(merged["line_height"])
    merged["as_bool"] = bool(merged["as_bool"])
    if merged["nan_color"] is not None:
        merged["nan_color"] = tuple(map(int, merged["nan_color"]))
    return merged


def _format_overlay_value(value, fmt: str, *, as_bool: bool = False) -> str:
    if value is None:
        return "NA"
    if value is pd.NA:
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(fval):
        return "nan"
    if as_bool and np.isfinite(fval):
        if fval == 0.0:
            return "False"
        if fval == 1.0:
            return "True"
    return format(fval, fmt)


def _resolve_text_color_arrays(
    text_overlays: list[tuple[str, np.ndarray | None]],
    style: dict,
    n_frames: int,
) -> list[np.ndarray | None]:
    """
    Precompute per-frame BGR text colors from optional cmap settings.
    """
    out: list[np.ndarray | None] = []
    if not text_overlays:
        return out
    for label, values in text_overlays:
        if values is None:
            out.append(None)
            continue
        tstyle = _style_for_text(style, label)
        cmap_name = tstyle.get("cmap")
        if cmap_name in (None, ""):
            base = np.array(tstyle["color"], dtype=np.uint8)
            out.append(np.tile(base[None, :], (n_frames, 1)))
            continue
        if mpl_cm is None:
            raise ValueError(
                f"text style for '{label}' requests cmap='{cmap_name}' "
                "but matplotlib is not available"
            )
        # Support nullable/object inputs (e.g. pd.NA) by coercing to float with NaN.
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float, copy=False)
        finite = np.isfinite(arr)
        vmin = tstyle.get("vmin")
        vmax = tstyle.get("vmax")
        if vmin is None:
            vmin = float(np.nanmin(arr)) if np.any(finite) else 0.0
        else:
            vmin = float(vmin)
        if vmax is None:
            vmax = float(np.nanmax(arr)) if np.any(finite) else (vmin + 1.0)
        else:
            vmax = float(vmax)
        if vmax <= vmin:
            vmax = vmin + 1e-12
        norm = (arr - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0.0, 1.0)
        if mpl is not None and hasattr(mpl, "colormaps"):
            cmap = mpl.colormaps[str(cmap_name)]
        else:  # pragma: no cover - for older matplotlib
            cmap = mpl_cm.get_cmap(str(cmap_name))
        rgba = cmap(norm)
        colors = np.rint(rgba[:, :3][:, ::-1] * 255.0).astype(np.uint8)  # RGB->BGR
        nan_color = tstyle.get("nan_color")
        if nan_color is None:
            nan_color = tstyle["color"]
        colors[~finite] = np.asarray(nan_color, dtype=np.uint8)
        out.append(colors)
    return out


def _compute_dynamic_array(
    spec: dict,
    source: np.ndarray,
    n_frames: int,
    *,
    prop_name: str,
) -> np.ndarray:
    if len(source) != n_frames:
        raise ValueError(f"Dynamic source '{spec['from']}' length must match n_frames ({n_frames})")
    if "map" in spec:
        mapping = spec["map"]
        if not isinstance(mapping, dict):
            raise ValueError(f"Dynamic map for '{prop_name}' must be a dict")
        out = []
        for raw in source:
            if pd.isna(raw):
                key = None
            elif isinstance(raw, np.generic):
                key = raw.item()
            else:
                key = raw
            if key in mapping:
                out.append(mapping[key])
            elif isinstance(key, float) and int(key) == key and int(key) in mapping:
                out.append(mapping[int(key)])
            elif str(key) in mapping:
                out.append(mapping[str(key)])
            elif "default" in mapping:
                out.append(mapping["default"])
            elif None in mapping:
                out.append(mapping[None])
            else:
                out.append(key)
        return np.asarray(out, dtype=object)
    if "cmap" in spec:
        if mpl_cm is None:
            raise ValueError(
                f"Dynamic style for '{prop_name}' requests cmap='{spec['cmap']}' "
                "but matplotlib is not available"
            )
        arr = pd.to_numeric(pd.Series(source), errors="coerce").to_numpy(dtype=float, copy=False)
        finite = np.isfinite(arr)
        vmin = float(spec.get("vmin", np.nanmin(arr) if np.any(finite) else 0.0))
        vmax = float(spec.get("vmax", np.nanmax(arr) if np.any(finite) else (vmin + 1.0)))
        if vmax <= vmin:
            vmax = vmin + 1e-12
        norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        if mpl is not None and hasattr(mpl, "colormaps"):
            cmap = mpl.colormaps[str(spec["cmap"])]
        else:  # pragma: no cover
            cmap = mpl_cm.get_cmap(str(spec["cmap"]))
        rgba = cmap(norm)
        colors = np.rint(rgba[:, :3][:, ::-1] * 255.0).astype(np.uint8)  # RGB->BGR
        nan_color = spec.get("nan_color")
        if nan_color is not None:
            colors[~finite] = np.asarray(nan_color, dtype=np.uint8)
        return colors
    raise ValueError("Dynamic spec must include 'map' or 'cmap'")


def _normalize_dynamic_value(value, prop_name: str):
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if prop_name in {"color", "edge_color", "fill_color", "outline_color", "nan_color"}:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return tuple(map(int, value))
    if prop_name in {
        "radius",
        "width",
        "edge_width",
        "thickness",
        "outline_thickness",
        "line_height",
    }:
        return int(value)
    if prop_name in {"fill_alpha", "font_scale", "vmin", "vmax"}:
        return float(value)
    if prop_name in {"as_bool"}:
        return bool(value)
    if prop_name in {"fill_mode", "format", "cmap"}:
        return str(value)
    return value


def _apply_dynamic_overrides(
    base_style: dict,
    dyn_props: dict[str, np.ndarray] | None,
    frame_idx: int,
) -> dict:
    if not dyn_props:
        return base_style
    out = dict(base_style)
    for prop, arr in dyn_props.items():
        if frame_idx >= len(arr):
            continue
        out[prop] = _normalize_dynamic_value(arr[frame_idx], prop)
    return out


def _is_valid(pix: np.ndarray, idx: int) -> bool:
    return idx < len(pix) and pix[idx, 0] >= 0 and pix[idx, 1] >= 0


def _compute_bounds(
    points_xy: np.ndarray, boundary_arrays: list[tuple[str, np.ndarray]], pad: float = 0.05
) -> tuple[float, float, float, float]:
    flat = points_xy.reshape(-1, 2)
    valid = np.isfinite(flat[:, 0]) & np.isfinite(flat[:, 1])
    xs = [flat[valid, 0]] if np.any(valid) else []
    ys = [flat[valid, 1]] if np.any(valid) else []
    for _, arr in boundary_arrays:
        poly = np.asarray(arr, dtype=float)
        if poly.ndim != 3 or poly.shape[2] != 2 or poly.shape[0] == 0 or poly.shape[1] == 0:
            continue
        flat_poly = poly.reshape(-1, 2)
        ok = np.isfinite(flat_poly[:, 0]) & np.isfinite(flat_poly[:, 1])
        if np.any(ok):
            xs.append(flat_poly[ok, 0])
            ys.append(flat_poly[ok, 1])
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


def _resolve_boundary_z(name: str, boundary_z) -> float:
    if isinstance(boundary_z, dict):
        return float(boundary_z.get(name, 0.0))
    if boundary_z is None:
        return 0.0
    return float(boundary_z)


def _project_boundary_arrays_3d_to_2d(
    boundary_arrays: list[tuple[str, np.ndarray]],
    projector: dict,
    boundary_z,
    n_frames: int,
) -> list[tuple[str, np.ndarray]]:
    projected: list[tuple[str, np.ndarray]] = []
    for name, poly in boundary_arrays:
        arr = np.asarray(poly, dtype=float)
        if arr.ndim != 3 or arr.shape[0] != n_frames:
            raise ValueError("Boundary array must have shape (n_frames, n_vertices, 2|3)")
        if arr.shape[2] == 2:
            z = _resolve_boundary_z(name, boundary_z)
            xyz = np.concatenate(
                (arr, np.full((arr.shape[0], arr.shape[1], 1), z, dtype=float)),
                axis=2,
            )
        elif arr.shape[2] == 3:
            xyz = arr
        else:
            raise ValueError("Boundary array must have 2 or 3 dimensions on last axis")
        projected.append((str(name), _project_xyz_with_projector(xyz, projector)))
    return projected


class GeometryAnimationStream:
    """
    OpenCV-backed frame stream for points, lines, and boundaries.

    Frames are generated lazily on demand, so callers can use this object both
    for random-access rendering (``get_frame(i)``) and sequential playback
    (``read()``, iteration, ``play()``, ``save()``).

    Examples
    --------
    ```pycon
    >>> import numpy as np
    >>> points = np.array(
    ...     [
    ...         [[10.0, 10.0], [20.0, 20.0]],
    ...         [[11.0, 11.0], [21.0, 21.0]],
    ...     ],
    ...     dtype=float,
    ... )
    >>> stream = build_geometry_stream_from_points(
    ...     points=points,
    ...     point_names=["nose", "tail"],
    ...     draw_points=["nose"],
    ...     lines=[("nose", "tail")],
    ...     frame_ids=np.array([0, 1]),
    ...     pixel_coords=True,
    ...     canvas_size=(64, 48),
    ... )
    >>> stream.frame_count
    2
    >>> stream.get_frame(0).shape
    (48, 64, 3)

    ```
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
        self._style = style or {}
        self._style_sources = style_sources or {}
        self._text_overlays = []
        for label, values in text_overlays:
            if values is None:
                self._text_overlays.append((str(label), None))
            else:
                self._text_overlays.append((str(label), np.asarray(values)))
        self._text_colors = _resolve_text_color_arrays(
            self._text_overlays,
            self._style,
            points_xy.shape[0],
        )
        self._dynamic_styles: dict[str, dict[object, dict[str, np.ndarray]]] = {
            "points": {},
            "lines": {},
            "boundaries": {},
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
        for label, values in self._text_overlays:
            if values is None:
                continue
            _populate("text", label, _style_raw_for_text(self._style, label))

        self._canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        self.fps = float(fps)
        self._bg_color = tuple(map(int, bg_color))
        self._pixel_coords = bool(pixel_coords)
        self._cursor = 0
        self._bounds = _compute_bounds(points_xy, self._boundary_arrays, pad=bounds_pad)

    @property
    def frame_count(self) -> int:
        """Number of renderable frames."""
        return int(self._points_xy.shape[0])

    @property
    def frame_ids(self) -> np.ndarray:
        """Copy of source frame identifiers aligned to stream indices."""
        return self._frame_ids.copy()

    def reset(self) -> None:
        """Reset sequential cursor used by ``read()`` / iteration."""
        self._cursor = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Return next rendered frame using VideoCapture-style semantics.

        Returns
        -------
        tuple[bool, np.ndarray | None]
            ``(True, frame)`` while frames remain, otherwise ``(False, None)``.

        Examples
        --------
        ```pycon
        >>> import numpy as np
        >>> s = build_geometry_stream_from_points(
        ...     points=np.array([[[1.0, 2.0]]], dtype=float),
        ...     point_names=["p1"],
        ...     frame_ids=np.array([0]),
        ...     pixel_coords=True,
        ... )
        >>> ok, frame = s.read()
        >>> ok
        True

        ```
        """
        if self._cursor >= self.frame_count:
            return False, None
        frame = self.get_frame(self._cursor)
        self._cursor += 1
        return True, frame

    def __iter__(self) -> GeometryAnimationStream:
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

        Examples
        --------
        ```pycon
        >>> import numpy as np
        >>> s = build_geometry_stream_from_points(
        ...     points=np.array([[[1.0, 2.0]]], dtype=float),
        ...     point_names=["p1"],
        ...     frame_ids=np.array([0]),
        ...     pixel_coords=True,
        ... )
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

        Parameters
        ----------
        frame : np.ndarray
            Base image buffer with shape ``(H, W, 3)`` in BGR.
        frame_idx : int
            Stream frame index to render.
        copy : bool, default True
            If True, draw into a copy and return it. If False, draw in-place.
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
            for overlay_i, (label, values) in enumerate(self._text_overlays):
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
                color_arr = self._text_colors[overlay_i]
                dyn_text = self._dynamic_styles["text"].get(label)
                if dyn_text is not None and "color" in dyn_text:
                    color = tstyle["color"]
                elif color_arr is None:
                    color = tstyle["color"]
                else:
                    color = tuple(map(int, color_arr[frame_idx]))
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
    boundary_arrays: list[tuple[str, np.ndarray]] | None = None,
    canvas_size: tuple[int, int] = (800, 800),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    style: dict | None = None,
    style_sources: dict[str, np.ndarray] | None = None,
    text_overlays: list[tuple[str, np.ndarray | None]] | None = None,
    pixel_coords: bool = False,
    bounds_pad: float = 0.05,
) -> GeometryAnimationStream:
    """
    Build a stream from tracking dataframe columns.

    This is a convenience wrapper that extracts the requested columns from
    ``df`` and delegates to :func:`build_geometry_stream_from_points`.

    Examples
    --------
    ```pycon
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {"a.x": [0.0], "a.y": [1.0], "b.x": [2.0], "b.y": [3.0]},
    ...     index=pd.Index([7], name="frame"),
    ... )
    >>> stream = build_geometry_stream(
    ...     df,
    ...     point_names=["a"],
    ...     lines=[("a", "b")],
    ...     frame_ids=np.array([7]),
    ...     pixel_coords=True,
    ... )
    >>> stream.frame_ids.tolist()
    [7]

    ```
    """
    if len(dims) not in (2, 3):
        raise ValueError("dims must be length 2 or 3")
    if lines is None:
        lines = []
    if frame_ids is None:
        frame_ids = df.index.to_numpy(copy=True)
    if boundary_arrays is None:
        boundary_arrays = []
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
    point_arrays = []
    for point in all_point_names:
        cols = [df[f"{point}.{dim}"].to_numpy(dtype=float, copy=True) for dim in dims]
        point_arrays.append(np.column_stack(cols))
    points_arr = np.stack(point_arrays, axis=1)
    return build_geometry_stream_from_points(
        points=points_arr,
        point_names=all_point_names,
        draw_points=point_names,
        lines=lines,
        view=view,
        boundary_z=boundary_z,
        frame_ids=np.asarray(frame_ids),
        fps=fps,
        boundary_arrays=boundary_arrays,
        canvas_size=canvas_size,
        bg_color=bg_color,
        style=style,
        style_sources=style_sources,
        text_overlays=text_overlays,
        pixel_coords=pixel_coords,
        bounds_pad=bounds_pad,
    )


def build_geometry_stream_from_points(
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
    canvas_size: tuple[int, int] = (800, 800),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    style: dict | None = None,
    style_sources: dict[str, np.ndarray] | None = None,
    text_overlays: list[tuple[str, np.ndarray | None]] | None = None,
    pixel_coords: bool = False,
    bounds_pad: float = 0.05,
) -> GeometryAnimationStream:
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
    GeometryAnimationStream
        Lazy stream object supporting ``read()``, ``get_frame()``, ``play()``,
        and ``save()``.

    Examples
    --------
    ```pycon
    >>> import numpy as np
    >>> points = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=float)
    >>> stream = build_geometry_stream_from_points(
    ...     points=points,
    ...     point_names=["a", "b"],
    ...     draw_points=["a"],
    ...     lines=[("a", "b")],
    ...     frame_ids=np.array([42]),
    ...     pixel_coords=True,
    ... )
    >>> stream.frame_ids.tolist()
    [42]

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

    return GeometryAnimationStream(
        points_xy=points_xy,
        point_names=point_names,
        draw_point_indices=draw_point_indices,
        frame_ids=np.asarray(frame_ids),
        lines_idx=lines_idx,
        line_keys=line_keys,
        boundary_arrays=boundary_arrays,
        text_overlays=text_overlays,
        canvas_size=canvas_size,
        fps=fps,
        bg_color=bg_color,
        style=style,
        style_sources=style_sources,
        pixel_coords=pixel_coords,
        bounds_pad=bounds_pad,
    )
