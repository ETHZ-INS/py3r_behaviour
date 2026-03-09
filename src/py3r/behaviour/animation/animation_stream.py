from __future__ import annotations

import numpy as np
import pandas as pd

from .compiler import (
    _compute_bounds,
    _format_overlay_value,
    collect_dynamic_source_names_from_style,
    compile_styles,
    prepare_indices,
    prepare_points_and_boundaries,
)
from .io import play_stream, save_stream
from .models import GeometryData
from .renderer import render_into_frame


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
    >>> stream = build_animation_stream_from_points(
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
        self._style_program = compile_styles(
            point_names=self._point_names,
            line_keys=self._line_keys,
            boundary_arrays=self._boundary_arrays,
            text_overlays=text_overlays,
            style=self._style,
            style_sources=self._style_sources,
            n_frames=points_xy.shape[0],
        )
        self._text_overlays = self._style_program.text_overlays

        self._canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        self.fps = float(fps)
        self._bg_color = tuple(map(int, bg_color))
        self._pixel_coords = bool(pixel_coords)
        self._cursor = 0
        self._bounds = _compute_bounds(points_xy, self._boundary_arrays, pad=bounds_pad)
        self._geometry = GeometryData(
            points_xy=self._points_xy,
            point_names=self._point_names,
            draw_point_indices=self._draw_point_indices,
            lines_idx=self._lines_idx,
            line_keys=self._line_keys,
            boundary_arrays=self._boundary_arrays,
            bounds=self._bounds,
            pixel_coords=self._pixel_coords,
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

        Examples:
            ```pycon
            >>> import numpy as np
            >>> s = build_animation_stream_from_points(
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
        self._cursor = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Return the next rendered frame using VideoCapture-style semantics.

        Returns:
            tuple[bool, np.ndarray | None]: ``(True, frame)`` while frames remain;
            otherwise ``(False, None)``.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> s = build_animation_stream_from_points(
            ...     points=np.array([[[1.0, 2.0]]], dtype=float),
            ...     point_names=["p1"],
            ...     frame_ids=np.array([0]),
            ...     pixel_coords=True,
            ... )
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

        Returns:
            np.ndarray: Rendered BGR image with shape ``(H, W, 3)``.

        Raises:
            IndexError: If ``frame_idx`` is out of range.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> s = build_animation_stream_from_points(
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

        Args:
            frame: Base BGR image buffer with shape ``(H, W, 3)``.
            frame_idx: Stream frame index to render.
            copy: If ``True``, draw into a copy. If ``False``, draw in-place.

        Returns:
            np.ndarray: Rendered frame buffer.

        Raises:
            IndexError: If ``frame_idx`` is out of range.
            ValueError: If ``frame`` does not have shape ``(H, W, 3)``.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> s = build_animation_stream_from_points(
            ...     points=np.array([[[5.0, 6.0]]], dtype=float),
            ...     point_names=["p1"],
            ...     frame_ids=np.array([0]),
            ...     pixel_coords=True,
            ...     canvas_size=(32, 24),
            ... )
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
        return render_into_frame(
            frame,
            frame_idx=frame_idx,
            geometry=self._geometry,
            styles=self._style_program,
            copy=copy,
        )

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

        Raises:
            ValueError: If ``frame_step < 1`` or ``speed <= 0``.

        Examples:
            ```pycon
            >>> import numpy as np
            >>> s = build_animation_stream_from_points(
            ...     points=np.array([[[1.0, 2.0]]], dtype=float),
            ...     point_names=["p1"],
            ...     frame_ids=np.array([0]),
            ...     pixel_coords=True,
            ... )
            >>> s.play(loop=False, speed=1.0)  # xdoctest: +SKIP

            ```
        """
        if frame_step < 1:
            raise ValueError("frame_step must be >= 1")
        if speed <= 0:
            raise ValueError("speed must be > 0")
        play_stream(
            self,
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

        Raises:
            ValueError: If ``frame_step < 1`` or writer/capture cannot be opened.

        Examples:
            ```pycon
            >>> import tempfile
            >>> import numpy as np
            >>> s = build_animation_stream_from_points(
            ...     points=np.array([[[1.0, 2.0]]], dtype=float),
            ...     point_names=["p1"],
            ...     frame_ids=np.array([0]),
            ...     pixel_coords=True,
            ... )
            >>> with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            ...     s.save(f.name)  # xdoctest: +SKIP

            ```
        """
        if frame_step < 1:
            raise ValueError("frame_step must be >= 1")
        save_stream(
            self,
            out_path,
            fps=fps,
            frame_step=frame_step,
            video_path=video_path,
            align_to_frame_ids=align_to_frame_ids,
            codec=codec,
        )


def build_animation_stream(
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
) -> AnimationStream:
    """
    Build a stream from tracking dataframe columns.

    This is a convenience wrapper that extracts the requested columns from
    ``df`` and delegates to :func:`build_animation_stream_from_points`.

    Examples
    --------
    ```pycon
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {"a.x": [0.0], "a.y": [1.0], "b.x": [2.0], "b.y": [3.0]},
    ...     index=pd.Index([7], name="frame"),
    ... )
    >>> stream = build_animation_stream(
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
    return build_animation_stream_from_points(
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


def build_animation_stream_from_points(
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
    >>> import numpy as np
    >>> points = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=float)
    >>> stream = build_animation_stream_from_points(
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

    points_xy, boundary_arrays = prepare_points_and_boundaries(
        points=points,
        view=view,
        boundary_arrays=boundary_arrays,
        boundary_z=boundary_z,
    )
    draw_point_indices, lines_idx, line_keys = prepare_indices(
        point_names=point_names,
        draw_points=draw_points,
        lines=lines,
    )

    return AnimationStream(
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


__all__ = [
    "AnimationStream",
    "build_animation_stream",
    "build_animation_stream_from_points",
    "collect_dynamic_source_names_from_style",
    "_format_overlay_value",
]
