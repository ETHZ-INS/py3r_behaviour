from __future__ import annotations

import numpy as np

from .compiler import (
    _format_overlay_value,
    collect_dynamic_source_names_from_style,
    compile_scene,
)
from .io import play, save
from .models import AnimationStreamState, SceneInput
from .renderer import render_frame


class AnimationStream:
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
    >>> stream = build_animation_stream(
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
        scene,
    ) -> None:
        self._scene = scene
        self._state = AnimationStreamState()

    @property
    def frame_count(self) -> int:
        """Number of renderable frames."""
        return int(self._scene.points_xy.shape[0])

    @property
    def frame_ids(self) -> np.ndarray:
        """Copy of source frame identifiers aligned to stream indices."""
        return self._scene.frame_ids.copy()

    def reset(self) -> None:
        """
        Reset the internal sequential cursor.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> s = build_animation_stream(
            ...     points=np.array([[[1.0, 2.0]]], dtype=float),
            ...     point_names=["p1"],
            ...     frame_ids=np.array([0]),
            ...     pixel_coords=True,
            ... )
            >>> _ = s.read()
            >>> s.reset()
            >>> ok, _ = s.read()
            >>> ok
            True

            ```
        """
        self._state.cursor = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Return the next rendered frame using VideoCapture-style semantics.

        Returns:
            tuple[bool, np.ndarray | None]: ``(True, frame)`` while frames remain;
            otherwise ``(False, None)``.
        """
        if self._state.cursor >= self.frame_count:
            return False, None
        frame = self.get_frame(self._state.cursor)
        self._state.cursor += 1
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
        """
        return render_frame(self._scene, frame_idx)

    def render_into(self, frame: np.ndarray, *, frame_idx: int, copy: bool = True) -> np.ndarray:
        """
        Draw stream geometry into an existing frame buffer.
        """
        base = frame.copy() if copy else frame
        return render_frame(self._scene, frame_idx, frame=base)

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
        play(
            self._scene,
            fps=fps,
            frame_step=frame_step,
            speed=speed,
            window_name=window_name,
            loop=loop,
            video_path=video_path,
            align_to_frame_ids=align_to_frame_ids,
        )

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
        save(
            self._scene,
            out_path,
            fps=fps,
            frame_step=frame_step,
            video_path=video_path,
            align_to_frame_ids=align_to_frame_ids,
            codec=codec,
        )


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
    canvas_size: tuple[int, int] = (800, 800),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    style: dict | None = None,
    style_sources: dict[str, np.ndarray] | None = None,
    text_overlays: list[tuple[str, np.ndarray | None]] | None = None,
    pixel_coords: bool = False,
    bounds_pad: float = 0.05,
) -> AnimationStream:
    scene_input = SceneInput(
        points=np.asarray(points),
        point_names=list(point_names),
        draw_points=list(point_names if draw_points is None else draw_points),
        lines=[] if lines is None else list(lines),
        boundaries={}
        if boundary_arrays is None
        else {str(name): np.asarray(arr) for name, arr in boundary_arrays},
        features={}
        if style_sources is None
        else {str(k): np.asarray(v) for k, v in style_sources.items()},
        text_overlays=[] if text_overlays is None else list(text_overlays),
        style={} if style is None else dict(style),
        frame_ids=np.asarray(frame_ids),
        fps=float(fps),
        canvas_size=(int(canvas_size[0]), int(canvas_size[1])),
        bg_color=tuple(map(int, bg_color)),
        pixel_coords=bool(pixel_coords),
        bounds_pad=float(bounds_pad),
        view={} if view is None else dict(view),
        boundary_z=boundary_z,
    )
    return AnimationStream(scene=compile_scene(scene_input))


__all__ = [
    "AnimationStream",
    "build_animation_stream",
    "collect_dynamic_source_names_from_style",
    "_format_overlay_value",
]
