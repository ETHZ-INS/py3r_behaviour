from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometryData:
    points_xy: np.ndarray
    point_names: list[str]
    draw_point_indices: list[int]
    lines_idx: list[tuple[int, int]]
    line_keys: list[tuple[str, str]]
    boundary_arrays: list[tuple[str, np.ndarray]]
    bounds: tuple[float, float, float, float]
    pixel_coords: bool


@dataclass(frozen=True)
class StyleProgram:
    style: dict
    text_overlays: list[tuple[str, np.ndarray | None]]
    text_colors: list[np.ndarray | None]
    compiled_styles: dict[str, dict[object, dict[str, object]]]
