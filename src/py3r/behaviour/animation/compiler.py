from __future__ import annotations

import numpy as np

from ._projection import (
    _make_projector,
    _project_boundary_arrays_3d_to_2d,
    _project_xyz_with_projector,
)
from ._style import (
    _compute_dynamic_array,
    _is_dynamic_spec,
    _resolve_text_color_arrays,
    _style_raw,
    _style_resolved,
)
from .models import StyleProgram


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
