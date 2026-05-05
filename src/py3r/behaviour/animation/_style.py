from __future__ import annotations

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


def _replace_dynamic_specs(merged: dict, defaults: dict) -> dict:
    for key, default in defaults.items():
        if _is_dynamic_spec(merged.get(key)):
            merged[key] = default
    return merged


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
    _replace_dynamic_specs(merged, {"color": (0, 255, 255), "radius": 3})
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
    _replace_dynamic_specs(merged, {"color": (255, 255, 255), "width": 1})
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
    _replace_dynamic_specs(
        merged,
        {
            "edge_color": (0, 255, 0),
            "fill_color": None,
            "edge_width": 1,
            "fill_alpha": 0.0,
            "fill_mode": "normal",
        },
    )
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


def _style_raw_for_axis(style: dict, name: str) -> dict:
    section = style.get("axes", {})
    merged = {"edge_color": (0, 255, 0), "edge_width": 1}
    merged.update(section.get("default", {}))
    merged.update(section.get(name, {}))
    return merged


def _style_for_axis(style: dict, name: str) -> dict:
    merged = _style_raw_for_axis(style, name)
    _replace_dynamic_specs(merged, {"edge_color": (0, 255, 0), "edge_width": 1})
    merged["edge_color"] = (
        None if merged["edge_color"] is None else tuple(map(int, merged["edge_color"]))
    )
    merged["edge_width"] = int(merged["edge_width"])
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
    _replace_dynamic_specs(
        merged,
        {
            "color": (255, 255, 255),
            "outline_color": (0, 0, 0),
            "font_scale": 0.5,
            "thickness": 1,
            "outline_thickness": 2,
            "line_height": 18,
            "format": None,
            "as_bool": False,
            "nan_color": None,
            "cmap": None,
            "vmin": None,
            "vmax": None,
        },
    )
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


def _resolve_text_color_for_frame(
    text_style: dict,
    values: np.ndarray,
    frame_idx: int,
) -> tuple[int, int, int]:
    """
    Resolve BGR text color for one overlay at one frame.

    If ``cmap`` is set, the frame's value is mapped through the colormap using
    ``vmin``/``vmax`` from the same (already-dynamic-resolved) style dict.
    """
    cmap_name = text_style.get("cmap")
    if cmap_name in (None, ""):
        return tuple(map(int, text_style["color"]))

    if mpl_cm is None:
        raise ValueError(f"text style requests cmap='{cmap_name}' but matplotlib is not available")

    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float, copy=False)
    finite = np.isfinite(arr)
    current = arr[frame_idx] if frame_idx < len(arr) else np.nan
    if not np.isfinite(current):
        nan_color = text_style.get("nan_color")
        if nan_color is None:
            nan_color = text_style["color"]
        return tuple(map(int, nan_color))

    vmin = text_style.get("vmin")
    vmax = text_style.get("vmax")
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

    norm = float(np.clip((current - vmin) / (vmax - vmin), 0.0, 1.0))
    if mpl is not None and hasattr(mpl, "colormaps"):
        cmap = mpl.colormaps[str(cmap_name)]
    else:  # pragma: no cover - for older matplotlib
        cmap = mpl_cm.get_cmap(str(cmap_name))
    rgba = cmap(norm)
    bgr = np.rint(np.asarray(rgba[:3])[::-1] * 255.0).astype(np.uint8)
    return tuple(map(int, bgr))


def _compute_dynamic_array(
    spec: dict,
    source: np.ndarray,
    n_frames: int,
    *,
    prop_name: str,
    style_path: str | None = None,
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
                context = style_path or prop_name
                source_name = str(spec["from"])
                missing_value = "NA/None" if key is None else repr(key)
                raise ValueError(
                    f"Dynamic style {context} from source {source_name}, "
                    f"resolved to value {missing_value} not specified in map. "
                    "Hint: you can add map key 'default' to automatically catch all "
                    "non-specified values."
                )
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
