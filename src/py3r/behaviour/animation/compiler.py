from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import matplotlib as mpl
    import matplotlib.cm as mpl_cm
except Exception:  # pragma: no cover - optional dependency
    mpl_cm = None
    mpl = None

from .models import CompiledScene, SceneInput

_STYLE_DEFAULTS = {
    "points": {"color": (0, 255, 255), "radius": 3},
    "lines": {"color": (255, 255, 255), "width": 1},
    "boundaries": {
        "edge_color": (0, 255, 0),
        "edge_width": 1,
        "fill_color": None,
        "fill_alpha": 0.0,
        "fill_mode": "normal",
    },
    "text": {
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
    },
}


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


def _to_numeric_float_array(values: np.ndarray) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float, copy=False)


def _resolve_numeric_range(
    arr: np.ndarray,
    *,
    vmin: float | None,
    vmax: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(arr)
    lo = float(np.nanmin(arr)) if vmin is None and np.any(finite) else float(vmin or 0.0)
    hi = float(np.nanmax(arr)) if vmax is None and np.any(finite) else float(vmax or (lo + 1.0))
    if hi <= lo:
        hi = lo + 1e-12
    return finite, np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _resolve_matplotlib_cmap(cmap_name: str):
    if mpl_cm is None:
        raise ValueError(f"cmap='{cmap_name}' requested but matplotlib is not available")
    if mpl is not None and hasattr(mpl, "colormaps"):
        return mpl.colormaps[str(cmap_name)]
    return mpl_cm.get_cmap(str(cmap_name))  # pragma: no cover


def _map_norm_to_bgr(norm: np.ndarray, cmap_name: str) -> np.ndarray:
    cmap = _resolve_matplotlib_cmap(str(cmap_name))
    rgba = cmap(norm)
    return np.rint(rgba[:, :3][:, ::-1] * 255.0).astype(np.uint8)


def _as_int_tuple(value) -> tuple[int, ...]:
    return tuple(map(int, value))


def _as_optional_int_tuple(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    return _as_int_tuple(value)


def _merge_style(style: dict, section: str, item_key, *, allow_rev_line_key: bool = False) -> dict:
    section_cfg = style.get(section, {})
    merged = dict(_STYLE_DEFAULTS[section])
    merged.update(section_cfg.get("default", {}))
    if allow_rev_line_key and isinstance(item_key, tuple) and len(item_key) == 2:
        if item_key in section_cfg:
            merged.update(section_cfg[item_key])
        elif (item_key[1], item_key[0]) in section_cfg:
            merged.update(section_cfg[(item_key[1], item_key[0])])
    else:
        merged.update(section_cfg.get(item_key, {}))
    return merged


def _replace_dynamic_props(style_dict: dict, section: str) -> dict:
    out = dict(style_dict)
    for k, default in _STYLE_DEFAULTS[section].items():
        if _is_dynamic_spec(out.get(k)):
            out[k] = default
    return out


def _coerce_style(section: str, style_dict: dict) -> dict:
    out = dict(style_dict)
    if section == "points":
        out["color"] = _as_int_tuple(out["color"])
        out["radius"] = int(out["radius"])
    elif section == "lines":
        out["color"] = _as_int_tuple(out["color"])
        out["width"] = int(out["width"])
    elif section == "boundaries":
        out["edge_color"] = _as_optional_int_tuple(out["edge_color"])
        out["fill_color"] = _as_optional_int_tuple(out["fill_color"])
        out["edge_width"] = int(out["edge_width"])
        out["fill_alpha"] = float(np.clip(out["fill_alpha"], 0.0, 1.0))
        out["fill_mode"] = str(out.get("fill_mode", "normal")).lower()
        if out["fill_mode"] not in {"normal", "erase"}:
            raise ValueError("boundary fill_mode must be 'normal' or 'erase'")
    elif section == "text":
        out["color"] = _as_int_tuple(out["color"])
        out["outline_color"] = _as_int_tuple(out["outline_color"])
        out["font_scale"] = float(out["font_scale"])
        out["thickness"] = int(out["thickness"])
        out["outline_thickness"] = int(out["outline_thickness"])
        out["line_height"] = int(out["line_height"])
        out["as_bool"] = bool(out["as_bool"])
        if out["nan_color"] is not None:
            out["nan_color"] = _as_int_tuple(out["nan_color"])
    return out


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
    if prop_name == "as_bool":
        return bool(value)
    if prop_name in {"fill_mode", "format", "cmap"}:
        return str(value)
    return value


def _compute_dynamic_array(spec: dict, source: np.ndarray, *, n_frames: int) -> np.ndarray:
    if len(source) != n_frames:
        raise ValueError(f"Dynamic source '{spec['from']}' length must match n_frames ({n_frames})")
    if "map" in spec:
        mapping = spec["map"]
        if not isinstance(mapping, dict):
            raise ValueError("Dynamic map must be a dict")
        out = []
        for raw in source:
            key = None if pd.isna(raw) else (raw.item() if isinstance(raw, np.generic) else raw)
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
        arr = _to_numeric_float_array(source)
        finite, norm = _resolve_numeric_range(
            arr,
            vmin=(None if spec.get("vmin") is None else float(spec["vmin"])),
            vmax=(None if spec.get("vmax") is None else float(spec["vmax"])),
        )
        out = _map_norm_to_bgr(norm, str(spec["cmap"]))
        if spec.get("nan_color") is not None:
            out[~finite] = np.asarray(spec["nan_color"], dtype=np.uint8)
        return out
    raise ValueError("Dynamic spec must include 'map' or 'cmap'")


def _format_overlay_value(value, fmt: str, *, as_bool: bool = False) -> str:
    if value is None or value is pd.NA:
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


def _is_valid(pix: np.ndarray, idx: int) -> bool:
    return idx < len(pix) and pix[idx, 0] >= 0 and pix[idx, 1] >= 0


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


def normalize_input(scene: SceneInput) -> SceneInput:
    points = np.asarray(scene.points, dtype=float)
    if points.ndim != 3 or points.shape[2] not in (2, 3):
        raise ValueError("points must have shape (n_frames, n_points, 2|3)")
    if points.shape[1] != len(scene.point_names):
        raise ValueError("point_names length must match points.shape[1]")
    frame_ids = np.asarray(scene.frame_ids)
    if len(frame_ids) != points.shape[0]:
        raise ValueError("frame_ids length must match points.shape[0]")

    boundaries: dict[str, np.ndarray] = {}
    for name, arr in scene.boundaries.items():
        barr = np.asarray(arr, dtype=float)
        if barr.ndim != 3 or barr.shape[0] != points.shape[0] or barr.shape[2] not in (2, 3):
            raise ValueError("Boundary arrays must have shape (n_frames, n_vertices, 2|3)")
        boundaries[str(name)] = barr

    features: dict[str, np.ndarray] = {}
    for name, arr in scene.features.items():
        farr = np.asarray(arr)
        if len(farr) != points.shape[0]:
            raise ValueError(f"Feature '{name}' length must match n_frames ({points.shape[0]})")
        features[str(name)] = farr

    overlays: list[tuple[str, np.ndarray | None]] = []
    for label, values in scene.text_overlays:
        if values is None:
            overlays.append((str(label), None))
            continue
        arr = np.asarray(values)
        if len(arr) != points.shape[0]:
            raise ValueError(
                f"text overlay '{label}' length must match n_frames ({points.shape[0]})"
            )
        overlays.append((str(label), arr))

    lines = [] if scene.lines is None else list(scene.lines)
    draw_points = scene.point_names if not scene.draw_points else list(scene.draw_points)
    return SceneInput(
        points=points,
        point_names=list(scene.point_names),
        draw_points=draw_points,
        lines=lines,
        boundaries=boundaries,
        features=features,
        text_overlays=overlays,
        style=scene.style or {},
        frame_ids=frame_ids,
        fps=float(scene.fps),
        canvas_size=(int(scene.canvas_size[0]), int(scene.canvas_size[1])),
        bg_color=tuple(map(int, scene.bg_color)),
        pixel_coords=bool(scene.pixel_coords),
        bounds_pad=float(scene.bounds_pad),
        view=scene.view or {},
        boundary_z=scene.boundary_z,
    )


def _make_projector(points_xyz: np.ndarray, view: dict) -> dict:
    azim_deg = float(view.get("azim", 45.0))
    elev_deg = float(view.get("elev", 30.0))
    proj = str(view.get("proj", "ortho")).lower()
    if proj not in {"ortho", "persp"}:
        raise ValueError("view['proj'] must be 'ortho' or 'persp'")
    valid = np.isfinite(points_xyz).all(axis=2)
    center = np.nanmean(points_xyz[valid], axis=0) if np.any(valid) else np.array([0.0, 0.0, 0.0])
    centered = points_xyz - center
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    rz = np.array(
        [[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(el), -np.sin(el)], [0.0, np.sin(el), np.cos(el)]],
        dtype=float,
    )
    projector = {"center": center, "rotm": rz.T @ rx.T, "proj": proj}
    if proj == "persp":
        finite_vals = centered[np.isfinite(centered)]
        max_abs = float(np.nanmax(np.abs(finite_vals))) if finite_vals.size else 1.0
        projector["camera_distance"] = float(view.get("camera_distance", max(1.0, 4.0 * max_abs)))
        projector["focal_length"] = float(view.get("focal_length", projector["camera_distance"]))
    return projector


def _project_xyz(points_xyz: np.ndarray, projector: dict) -> np.ndarray:
    valid = np.isfinite(points_xyz).all(axis=2)
    rot = (points_xyz - projector["center"]) @ projector["rotm"]
    x, y, z = rot[:, :, 0], rot[:, :, 1], rot[:, :, 2]
    if projector["proj"] == "ortho":
        xy = np.stack((x, y), axis=2)
        xy[~valid] = np.nan
        return xy
    denom = projector["camera_distance"] - z
    good = valid & np.isfinite(denom) & (denom > 1e-9)
    xp = np.full_like(x, np.nan)
    yp = np.full_like(y, np.nan)
    xp[good] = projector["focal_length"] * x[good] / denom[good]
    yp[good] = projector["focal_length"] * y[good] / denom[good]
    return np.stack((xp, yp), axis=2)


def project_geometry_if_needed(scene: SceneInput) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if scene.points.shape[2] == 2:
        return scene.points.copy(), {
            name: (arr[:, :, :2].copy() if arr.shape[2] == 3 else arr.copy())
            for name, arr in scene.boundaries.items()
        }
    projector = _make_projector(scene.points, scene.view)
    points_xy = _project_xyz(scene.points, projector)
    boundaries_xy: dict[str, np.ndarray] = {}
    for name, arr in scene.boundaries.items():
        if arr.shape[2] == 3:
            boundaries_xy[name] = _project_xyz(arr, projector)
        else:
            if isinstance(scene.boundary_z, dict):
                z = float(scene.boundary_z.get(name, 0.0))
            elif scene.boundary_z is None:
                z = 0.0
            else:
                z = float(scene.boundary_z)
            xyz = np.concatenate(
                (arr, np.full((arr.shape[0], arr.shape[1], 1), z, dtype=float)), axis=2
            )
            boundaries_xy[name] = _project_xyz(xyz, projector)
    return points_xy, boundaries_xy


def compute_bounds(
    points_xy: np.ndarray,
    boundaries_xy: dict[str, np.ndarray],
    *,
    pad: float,
) -> tuple[float, float, float, float]:
    flat = points_xy.reshape(-1, 2)
    valid = np.isfinite(flat[:, 0]) & np.isfinite(flat[:, 1])
    xs = [flat[valid, 0]] if np.any(valid) else []
    ys = [flat[valid, 1]] if np.any(valid) else []
    for arr in boundaries_xy.values():
        poly = arr.reshape(-1, 2)
        ok = np.isfinite(poly[:, 0]) & np.isfinite(poly[:, 1])
        if np.any(ok):
            xs.append(poly[ok, 0])
            ys.append(poly[ok, 1])
    if not xs:
        return (0.0, 1.0, 0.0, 1.0)
    xmin, xmax = float(np.min(np.concatenate(xs))), float(np.max(np.concatenate(xs)))
    ymin, ymax = float(np.min(np.concatenate(ys))), float(np.max(np.concatenate(ys)))
    if xmin == xmax:
        xmax = xmin + 1.0
    if ymin == ymax:
        ymax = ymin + 1.0
    p = float(np.clip(pad, 0.0, 0.45))
    if p > 0:
        xspan, yspan = xmax - xmin, ymax - ymin
        xmin, xmax = xmin - xspan * p, xmax + xspan * p
        ymin, ymax = ymin - yspan * p, ymax + yspan * p
    return xmin, xmax, ymin, ymax


def _compile_section_styles(
    *,
    scene: SceneInput,
    section: str,
    item_keys: list,
    n_frames: int,
    allow_dynamic: bool = True,
) -> list[dict]:
    out = [{} for _ in range(n_frames)]
    for item_key in item_keys:
        raw = _merge_style(scene.style, section, item_key, allow_rev_line_key=(section == "lines"))
        base = _coerce_style(section, _replace_dynamic_props(raw, section))
        dyn_arrays: dict[str, np.ndarray] = {}
        if allow_dynamic:
            for prop, val in raw.items():
                if not _is_dynamic_spec(val):
                    continue
                source_name = str(val["from"])
                if source_name not in scene.features:
                    raise ValueError(
                        f"Dynamic style for {section}.{item_key}.{prop} references "
                        f"unknown source '{source_name}'"
                    )
                dyn_arrays[prop] = _compute_dynamic_array(
                    val, scene.features[source_name], n_frames=n_frames
                )
        for frame_idx in range(n_frames):
            frame_style = dict(base)
            for prop, arr in dyn_arrays.items():
                frame_style[prop] = _normalize_dynamic_value(arr[frame_idx], prop)
            out[frame_idx][item_key] = frame_style
    return out


def compile_dynamic_styles(
    scene: SceneInput,
    *,
    n_frames: int,
    line_keys: list[tuple[str, str]],
    boundary_names: list[str],
    text_labels: list[str],
) -> list[dict]:
    points_frames = _compile_section_styles(
        scene=scene,
        section="points",
        item_keys=scene.point_names,
        n_frames=n_frames,
    )
    lines_frames = _compile_section_styles(
        scene=scene,
        section="lines",
        item_keys=line_keys,
        n_frames=n_frames,
    )
    boundaries_frames = _compile_section_styles(
        scene=scene,
        section="boundaries",
        item_keys=boundary_names,
        n_frames=n_frames,
    )
    text_frames = _compile_section_styles(
        scene=scene,
        section="text",
        item_keys=text_labels,
        n_frames=n_frames,
        allow_dynamic=True,
    )
    return [
        {
            "points": points_frames[i],
            "lines": lines_frames[i],
            "boundaries": boundaries_frames[i],
            "text": text_frames[i],
        }
        for i in range(n_frames)
    ]


def compile_text_overlays(
    scene: SceneInput, styles_by_frame: list[dict]
) -> tuple[list[list[dict]], dict]:
    n_frames = len(styles_by_frame)
    text_cfg = scene.style.get("text", {})
    default_fmt = str(text_cfg.get("format", ".3f"))
    text_by_frame: list[list[dict]] = [[] for _ in range(n_frames)]
    for label, values in scene.text_overlays:
        if values is None:
            for i in range(n_frames):
                style = styles_by_frame[i]["text"].get(
                    label, _coerce_style("text", _STYLE_DEFAULTS["text"])
                )
                text_by_frame[i].append({"spacer": True, "style": style})
            continue
        arr = np.asarray(values)
        static_style = _merge_style(scene.style, "text", label)
        use_cmap = static_style.get("cmap") not in (None, "") and not _is_dynamic_spec(
            static_style.get("color")
        )
        cmap_colors = None
        if use_cmap:
            finite, norm = _resolve_numeric_range(
                _to_numeric_float_array(arr),
                vmin=(None if static_style.get("vmin") is None else float(static_style["vmin"])),
                vmax=(None if static_style.get("vmax") is None else float(static_style["vmax"])),
            )
            cmap_colors = _map_norm_to_bgr(norm, str(static_style["cmap"]))
            nan_color = static_style.get("nan_color")
            if nan_color is not None:
                cmap_colors[~finite] = np.asarray(nan_color, dtype=np.uint8)
        for i in range(n_frames):
            st = styles_by_frame[i]["text"][label]
            fmt = default_fmt if st.get("format") is None else str(st["format"])
            color = st["color"] if cmap_colors is None else tuple(map(int, cmap_colors[i]))
            text_by_frame[i].append(
                {
                    "spacer": False,
                    "text": f"{label}: {_format_overlay_value(arr[i], fmt, as_bool=st['as_bool'])}",
                    "style": st,
                    "color": color,
                }
            )
    return text_by_frame, text_cfg


def compile_scene(scene_input: SceneInput) -> CompiledScene:
    scene = normalize_input(scene_input)
    n_frames = scene.points.shape[0]
    points_xy, boundaries_xy = project_geometry_if_needed(scene)
    bounds = compute_bounds(points_xy, boundaries_xy, pad=scene.bounds_pad)
    point_idx = {name: i for i, name in enumerate(scene.point_names)}
    draw_point_indices = [point_idx[name] for name in scene.draw_points]
    lines_idx = [(point_idx[a], point_idx[b]) for a, b in scene.lines]
    styles_by_frame = compile_dynamic_styles(
        scene,
        n_frames=n_frames,
        line_keys=scene.lines,
        boundary_names=list(boundaries_xy.keys()),
        text_labels=[label for label, _ in scene.text_overlays],
    )
    text_by_frame, text_cfg = compile_text_overlays(scene, styles_by_frame)
    return CompiledScene(
        points_xy=points_xy,
        boundaries_xy=boundaries_xy,
        point_names=scene.point_names,
        draw_point_indices=draw_point_indices,
        lines_idx=lines_idx,
        line_keys=scene.lines,
        frame_ids=scene.frame_ids,
        fps=scene.fps,
        canvas_size=scene.canvas_size,
        bg_color=scene.bg_color,
        pixel_coords=scene.pixel_coords,
        bounds=bounds,
        styles_by_frame=styles_by_frame,
        text_by_frame=text_by_frame,
        text_config=text_cfg,
    )
