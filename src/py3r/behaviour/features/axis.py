from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _transform_axis_endpoints(
    A: np.ndarray,
    B: np.ndarray,
    offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift static axis reference points perpendicularly by ``offset``.

    Parameters
    ----------
    A, B : np.ndarray
        Reference point coordinate vectors, shape ``(n_dims,)``.
    offset : float
        Perpendicular displacement in coordinate units.  Positive is to
        the right when facing from A to B.  Must be ``0.0`` for axes with
        more than 2 dimensions.
    """
    if offset == 0.0:
        return A.copy(), B.copy()
    n_dims = len(A)
    if n_dims != 2:
        raise ValueError(f"offset is only supported for 2-D axes; got {n_dims} dims.")
    d = B - A
    length = np.linalg.norm(d)
    if length == 0:
        raise ValueError("Cannot apply a non-zero offset to a zero-length axis (A == B).")
    d_norm = d / length
    perp = np.array([d_norm[1], -d_norm[0]])  # right-hand perpendicular
    shift = offset * perp
    return A + shift, B + shift


def _transform_axis_per_frame(
    arr: np.ndarray,
    offset: float,
) -> np.ndarray:
    """Shift per-frame axis reference points perpendicularly by ``offset``.

    Parameters
    ----------
    arr : np.ndarray
        Shape ``(n_frames, 2, n_dims)`` — raw resolved axis reference points.
    offset : float
        Perpendicular displacement.  Must be ``0.0`` for ``n_dims != 2``.

    Returns
    -------
    np.ndarray
        Shifted array, same shape ``(n_frames, 2, n_dims)``.
    """
    if offset == 0.0:
        return arr.copy()
    n_dims = arr.shape[2]
    if n_dims != 2:
        raise ValueError(f"offset is only supported for 2-D axes; got {n_dims} dims.")
    A = arr[:, 0, :]  # (n, 2)
    B = arr[:, 1, :]  # (n, 2)
    d = B - A  # (n, 2)
    length = np.linalg.norm(d, axis=1, keepdims=True)  # (n, 1)
    # Degenerate frames (A == B) get zero shift rather than NaN.
    with np.errstate(invalid="ignore", divide="ignore"):
        d_norm = np.where(length > 0, d / length, 0.0)  # (n, 2)
    perp = np.stack([d_norm[:, 1], -d_norm[:, 0]], axis=1)  # (n, 2)
    shift = offset * perp  # (n, 2)
    return np.stack([A + shift, B + shift], axis=1)  # (n, 2, 2)


@dataclass(frozen=True)
class StaticAxis:
    """A static infinite axis defined by two reference points in N-dimensional space.

    The axis passes through both stored reference points and extends infinitely
    in both directions.  Distance computations and rendering always treat it as
    infinite: the reference points define *position* and *direction*, not
    endpoints.

    Unlike boundaries, ``dims`` may contain any number of coordinate names
    (e.g. ``("x", "y")`` or ``("x", "y", "z")``).  The ``offset`` transform
    is baked into the stored vertices at definition time.
    """

    vertices: tuple[tuple[float, ...], ...]  # exactly 2, each N-dim
    dims: tuple[str, ...]
    source_points: tuple[str, ...] | None = None
    name: str | None = None

    def with_name(self, name: str) -> StaticAxis:
        return StaticAxis(
            vertices=self.vertices,
            dims=self.dims,
            source_points=self.source_points,
            name=name,
        )

    def to_dict(self) -> dict:
        return {
            "kind": "static_axis",
            "vertices": [list(v) for v in self.vertices],
            "dims": list(self.dims),
            "source_points": list(self.source_points) if self.source_points is not None else None,
            "name": self.name,
        }

    def to_numpy(self) -> np.ndarray:
        """Return reference points as a float array of shape ``(2, n_dims)``."""
        return np.array(self.vertices, dtype=float)

    @classmethod
    def from_dict(cls, payload: dict) -> StaticAxis:
        vertices = tuple(tuple(float(c) for c in v) for v in payload["vertices"])
        dims = tuple(str(d) for d in payload["dims"])
        source_points = payload.get("source_points")
        return cls(
            vertices=vertices,
            dims=dims,
            source_points=tuple(source_points) if source_points is not None else None,
            name=payload.get("name"),
        )


@dataclass(frozen=True)
class DynamicAxis:
    """A dynamic infinite axis defined by two keypoint names in N-dimensional space.

    Reference-point coordinates are resolved per frame from tracking data at
    compute time.  The axis is always treated as infinite in both distance
    computations and rendering.

    ``offset`` shifts the axis perpendicularly to its direction at each frame
    (positive = right when facing from ``points[0]`` to ``points[1]``).
    Only supported for 2-D axes.
    """

    points: tuple[str, str]
    dims: tuple[str, ...]
    offset: float = 0.0
    name: str | None = None

    def with_name(self, name: str) -> DynamicAxis:
        return DynamicAxis(
            points=self.points,
            dims=self.dims,
            offset=self.offset,
            name=name,
        )

    def to_dict(self) -> dict:
        return {
            "kind": "dynamic_axis",
            "points": list(self.points),
            "dims": list(self.dims),
            "offset": self.offset,
            "name": self.name,
        }

    def to_numpy_per_frame(self, tracking_df) -> np.ndarray:
        """Resolve axis point names against tracking data and apply offset.

        Returns
        -------
        np.ndarray
            Shape ``(n_frames, 2, n_dims)`` with offset applied.
        """
        cols_per_point = []
        for point in self.points:
            cols = [f"{point}.{dim}" for dim in self.dims]
            missing = [c for c in cols if c not in tracking_df.columns]
            if missing:
                raise ValueError(
                    f"Dynamic axis point {point!r} missing columns {missing} in tracking data."
                )
            cols_per_point.append(tracking_df[cols].to_numpy(dtype=float, copy=True))
        arr = np.stack(cols_per_point, axis=1)  # (n_frames, 2, n_dims)

        if self.offset != 0.0:
            arr = _transform_axis_per_frame(arr, offset=self.offset)

        return arr

    @classmethod
    def from_dict(cls, payload: dict) -> DynamicAxis:
        points_raw = payload["points"]
        if len(points_raw) != 2:
            raise ValueError(f"DynamicAxis requires exactly 2 points; got {len(points_raw)}.")
        dims = tuple(str(d) for d in payload["dims"])
        return cls(
            points=(str(points_raw[0]), str(points_raw[1])),
            dims=dims,
            offset=float(payload.get("offset", 0.0)),
            name=payload.get("name"),
        )
