from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SceneInput:
    points: np.ndarray
    point_names: list[str]
    draw_points: list[str]
    lines: list[tuple[str, str]]
    boundaries: dict[str, np.ndarray]
    features: dict[str, np.ndarray]
    text_overlays: list[tuple[str, np.ndarray | None]]
    style: dict
    frame_ids: np.ndarray
    fps: float
    canvas_size: tuple[int, int]
    bg_color: tuple[int, int, int]
    pixel_coords: bool
    bounds_pad: float
    view: dict
    boundary_z: float | dict[str, float] | None


@dataclass(frozen=True)
class CompiledScene:
    points_xy: np.ndarray
    boundaries_xy: dict[str, np.ndarray]
    point_names: list[str]
    draw_point_indices: list[int]
    lines_idx: list[tuple[int, int]]
    line_keys: list[tuple[str, str]]
    frame_ids: np.ndarray
    fps: float
    canvas_size: tuple[int, int]
    bg_color: tuple[int, int, int]
    pixel_coords: bool
    bounds: tuple[float, float, float, float]
    styles_by_frame: list[dict]
    text_by_frame: list[list[dict]]
    text_config: dict


@dataclass
class AnimationStreamState:
    cursor: int = 0
