from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import matplotlib as mpl
    import matplotlib.cm as mpl_cm
except Exception:  # pragma: no cover - optional dependency
    mpl_cm = None
    mpl = None

from .models import StyleProgram


def _is_dynamic_spec(value) -> bool:
    return isinstance(value, dict) and "from" in value and ("map" in value or "cmap" in value)


def collect_dynamic_source_names_from_style(style: dict | None) -> set[str]:
    names: set[str] = set()

    def _walk(node):
        if _is_dynamic_spec(node):
            names.add(str(node["from"]))
            return
        if isinstance(node, dict):
            for val in node.values():
                _walk(val)
        elif isinstance(node, (list, tuple)):
            for val in node:
                _walk(val)

    _walk(style or {})
    return names


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


def _replace_dynamic_specs(merged: dict, defaults: dict) -> dict:
    for key, default in defaults.items():
        if _is_dynamic_spec(merged.get(key)):
            merged[key] = default
    return merged


def _merge_style_section(
    style: dict,
    section_name: str,
    item_key,
    *,
    defaults: dict,
    allow_reversed_tuple_key: bool = False,
) -> dict:
    section = style.get(section_name, {})
    merged = dict(defaults)
    merged.update(section.get("default", {}))
    if allow_reversed_tuple_key and isinstance(item_key, tuple) and len(item_key) == 2:
        if item_key in section:
            merged.update(section[item_key])
        else:
            rev = (item_key[1], item_key[0])
            if rev in section:
                merged.update(section[rev])
        return merged
    merged.update(section.get(item_key, {}))
    return merged


def _as_int_tuple(value) -> tuple[int, ...]:
    return tuple(map(int, value))


def _as_optional_int_tuple(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    return _as_int_tuple(value)


def _style_raw(style: dict, section_name: str, item_key) -> dict:
    return _merge_style_section(
        style,
        section_name,
        item_key,
        defaults=_STYLE_DEFAULTS[section_name],
        allow_reversed_tuple_key=(section_name == "lines"),
    )


def _coerce_style(section_name: str, merged: dict) -> dict:
    if section_name == "points":
        merged["color"] = _as_int_tuple(merged["color"])
        merged["radius"] = int(merged["radius"])
        return merged
    if section_name == "lines":
        merged["color"] = _as_int_tuple(merged["color"])
        merged["width"] = int(merged["width"])
        return merged
    if section_name == "boundaries":
        merged["edge_color"] = _as_optional_int_tuple(merged["edge_color"])
        merged["fill_color"] = _as_optional_int_tuple(merged["fill_color"])
        merged["edge_width"] = int(merged["edge_width"])
        merged["fill_alpha"] = float(np.clip(merged["fill_alpha"], 0.0, 1.0))
        fill_mode = str(merged.get("fill_mode", "normal")).lower()
        if fill_mode not in {"normal", "erase"}:
            raise ValueError("boundary fill_mode must be 'normal' or 'erase'")
        merged["fill_mode"] = fill_mode
        return merged
    if section_name == "text":
        merged["color"] = _as_int_tuple(merged["color"])
        merged["outline_color"] = _as_int_tuple(merged["outline_color"])
        merged["font_scale"] = float(merged["font_scale"])
        merged["thickness"] = int(merged["thickness"])
        merged["outline_thickness"] = int(merged["outline_thickness"])
        merged["line_height"] = int(merged["line_height"])
        merged["as_bool"] = bool(merged["as_bool"])
        if merged["nan_color"] is not None:
            merged["nan_color"] = _as_int_tuple(merged["nan_color"])
        return merged
    raise ValueError(f"Unknown style section: {section_name}")


def _style_resolved(style: dict, section_name: str, item_key) -> dict:
    merged = _style_raw(style, section_name, item_key)
    _replace_dynamic_specs(merged, _STYLE_DEFAULTS[section_name])
    return _coerce_style(section_name, merged)


def _resolve_text_color_arrays(
    text_overlays: list[tuple[str, np.ndarray | None]],
    style: dict,
    n_frames: int,
) -> list[np.ndarray | None]:
    out: list[np.ndarray | None] = []
    if not text_overlays:
        return out
    for label, values in text_overlays:
        if values is None:
            out.append(None)
            continue
        tstyle = _style_resolved(style, "text", label)
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
        norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        if mpl is not None and hasattr(mpl, "colormaps"):
            cmap = mpl.colormaps[str(cmap_name)]
        else:  # pragma: no cover - older matplotlib
            cmap = mpl_cm.get_cmap(str(cmap_name))
        rgba = cmap(norm)
        colors = np.rint(rgba[:, :3][:, ::-1] * 255.0).astype(np.uint8)
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
        colors = np.rint(rgba[:, :3][:, ::-1] * 255.0).astype(np.uint8)
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
    if prop_name == "as_bool":
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


def _compute_bounds(
    points_xy: np.ndarray,
    boundary_arrays: list[tuple[str, np.ndarray]],
    pad: float = 0.05,
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


def normalize_text_overlays(
    text_overlays: list[tuple[str, np.ndarray | None]] | None,
    *,
    n_frames: int,
) -> list[tuple[str, np.ndarray | None]]:
    overlays = [] if text_overlays is None else list(text_overlays)
    out: list[tuple[str, np.ndarray | None]] = []
    for label, values in overlays:
        if values is None:
            out.append((str(label), None))
            continue
        arr = np.asarray(values)
        if len(arr) != n_frames:
            raise ValueError(f"text overlay '{label}' length must match n_frames ({n_frames})")
        out.append((str(label), arr))
    return out


def compile_styles(
    *,
    point_names: list[str],
    line_keys: list[tuple[str, str]],
    boundary_arrays: list[tuple[str, np.ndarray]],
    text_overlays: list[tuple[str, np.ndarray | None]],
    style: dict | None,
    style_sources: dict[str, np.ndarray] | None,
    n_frames: int,
) -> StyleProgram:
    style_dict = style or {}
    sources = style_sources or {}
    overlays = normalize_text_overlays(text_overlays, n_frames=n_frames)
    text_colors = _resolve_text_color_arrays(overlays, style_dict, n_frames)
    compiled_styles: dict[str, dict[object, dict[str, object]]] = {
        "points": {},
        "lines": {},
        "boundaries": {},
        "text": {},
    }

    def _compile(
        section_name: str,
        item_key,
        *,
        allow_dynamic: bool = True,
    ) -> None:
        raw_style = _style_raw(style_dict, section_name, item_key)
        resolved_style = _style_resolved(style_dict, section_name, item_key)
        item_dyn: dict[str, np.ndarray] = {}
        if allow_dynamic:
            for prop, value in raw_style.items():
                if not _is_dynamic_spec(value):
                    continue
                source_name = str(value["from"])
                if source_name not in sources:
                    raise ValueError(
                        f"Dynamic style for {section_name}.{item_key}.{prop} references "
                        f"unknown source '{source_name}'"
                    )
                item_dyn[prop] = _compute_dynamic_array(
                    value,
                    np.asarray(sources[source_name]),
                    n_frames,
                    prop_name=prop,
                )
        compiled_styles[section_name][item_key] = {"base": resolved_style, "dyn": item_dyn or None}

    for point_name in point_names:
        _compile("points", point_name)
    for line_key in line_keys:
        _compile("lines", line_key)
    for boundary_name in {name for name, _ in boundary_arrays}:
        _compile("boundaries", boundary_name)
    for label, values in overlays:
        _compile("text", label, allow_dynamic=(values is not None))
    return StyleProgram(
        style=style_dict,
        text_overlays=overlays,
        text_colors=text_colors,
        compiled_styles=compiled_styles,
    )


def prepare_points_and_boundaries(
    *,
    points: np.ndarray,
    view: dict | None,
    boundary_arrays: list[tuple[str, np.ndarray]],
    boundary_z: float | dict[str, float] | None,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    if points.shape[2] == 3:
        projector = _make_projector(points, view)
        points_xy = _project_xyz_with_projector(points, projector)
        if boundary_arrays:
            boundaries_xy = _project_boundary_arrays_3d_to_2d(
                boundary_arrays,
                projector,
                boundary_z,
                points.shape[0],
            )
        else:
            boundaries_xy = []
        return points_xy, boundaries_xy
    return points.astype(float, copy=True), [
        (str(name), np.asarray(arr, dtype=float)) for name, arr in boundary_arrays
    ]


def prepare_indices(
    *,
    point_names: list[str],
    draw_points: list[str] | None,
    lines: list[tuple[str, str]],
) -> tuple[list[int], list[tuple[int, int]], list[tuple[str, str]]]:
    point_idx = {name: i for i, name in enumerate(point_names)}
    selected = point_names if draw_points is None else draw_points
    draw_point_indices = [point_idx[name] for name in selected]

    lines_idx: list[tuple[int, int]] = []
    line_keys: list[tuple[str, str]] = []
    for p1, p2 in lines:
        if p1 not in point_idx or p2 not in point_idx:
            raise ValueError(f"Unknown point in line ({p1}, {p2})")
        lines_idx.append((point_idx[p1], point_idx[p2]))
        line_keys.append((p1, p2))
    return draw_point_indices, lines_idx, line_keys
