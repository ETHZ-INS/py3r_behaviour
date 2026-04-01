from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_tuple_point(point) -> tuple[float, float]:
    x, y = point
    return (float(x), float(y))


@dataclass(frozen=True)
class StaticBoundary:
    """A static boundary resolved to numeric vertices in ``dims`` space."""

    vertices: tuple[tuple[float, float], ...]
    dims: tuple[str, str] = ("x", "y")
    source_points: tuple[str, ...] | None = None
    anchor_points: tuple[str, ...] | None = None
    scale_dim1: float = 1.0
    scale_dim2: float = 1.0
    name: str | None = None

    def with_name(self, name: str) -> StaticBoundary:
        return StaticBoundary(
            vertices=self.vertices,
            dims=self.dims,
            source_points=self.source_points,
            anchor_points=self.anchor_points,
            scale_dim1=self.scale_dim1,
            scale_dim2=self.scale_dim2,
            name=name,
        )

    def to_dict(self) -> dict:
        return {
            "kind": "static",
            "vertices": list(self.vertices),
            "dims": list(self.dims),
            "source_points": list(self.source_points) if self.source_points is not None else None,
            "anchor_points": list(self.anchor_points) if self.anchor_points is not None else None,
            "scale_dim1": self.scale_dim1,
            "scale_dim2": self.scale_dim2,
            "name": self.name,
        }

    def to_numpy(self) -> np.ndarray:
        """Return boundary vertices as a float numpy array of shape (n_vertices, 2)."""
        return np.asarray(self.vertices, dtype=float)

    @classmethod
    def from_dict(cls, payload: dict) -> StaticBoundary:
        vertices = tuple(_as_tuple_point(p) for p in payload.get("vertices", ()))
        dims = tuple(payload.get("dims", ("x", "y")))
        source_points = payload.get("source_points")
        anchor_points = payload.get("anchor_points")
        return cls(
            vertices=vertices,
            dims=(str(dims[0]), str(dims[1])),
            source_points=tuple(source_points) if source_points is not None else None,
            anchor_points=tuple(anchor_points) if anchor_points is not None else None,
            scale_dim1=float(payload.get("scale_dim1", 1.0)),
            scale_dim2=float(payload.get("scale_dim2", 1.0)),
            name=payload.get("name"),
        )


@dataclass(frozen=True)
class DynamicBoundary:
    """A dynamic boundary defined by ordered point names in ``dims`` space."""

    points: tuple[str, ...]
    dims: tuple[str, str] = ("x", "y")
    anchor_points: tuple[str, ...] | None = None
    scale_dim1: float = 1.0
    scale_dim2: float = 1.0
    name: str | None = None

    def with_name(self, name: str) -> DynamicBoundary:
        return DynamicBoundary(
            points=self.points,
            dims=self.dims,
            anchor_points=self.anchor_points,
            scale_dim1=self.scale_dim1,
            scale_dim2=self.scale_dim2,
            name=name,
        )

    def to_dict(self) -> dict:
        return {
            "kind": "dynamic",
            "points": list(self.points),
            "dims": list(self.dims),
            "anchor_points": list(self.anchor_points) if self.anchor_points is not None else None,
            "scale_dim1": self.scale_dim1,
            "scale_dim2": self.scale_dim2,
            "name": self.name,
        }

    def to_numpy_per_frame(self, tracking_df) -> np.ndarray:
        """
        Resolve dynamic boundary point names against tracking data.

        Returns
        -------
        np.ndarray
            Shape (n_frames, n_vertices, 2) in this boundary's ``dims``.
        """
        cols_per_point = []
        for point in self.points:
            col_x = f"{point}.{self.dims[0]}"
            col_y = f"{point}.{self.dims[1]}"
            if col_x not in tracking_df.columns or col_y not in tracking_df.columns:
                raise ValueError(
                    f"Dynamic boundary point {point} missing in tracking data for dims {self.dims}"
                )
            cols_per_point.append(
                np.column_stack(
                    (
                        tracking_df[col_x].to_numpy(dtype=float, copy=True),
                        tracking_df[col_y].to_numpy(dtype=float, copy=True),
                    )
                )
            )
        return np.stack(cols_per_point, axis=1)

    @classmethod
    def from_dict(cls, payload: dict) -> DynamicBoundary:
        dims = tuple(payload.get("dims", ("x", "y")))
        anchor_points = payload.get("anchor_points")
        return cls(
            points=tuple(payload.get("points", ())),
            dims=(str(dims[0]), str(dims[1])),
            anchor_points=tuple(anchor_points) if anchor_points is not None else None,
            scale_dim1=float(payload.get("scale_dim1", 1.0)),
            scale_dim2=float(payload.get("scale_dim2", 1.0)),
            name=payload.get("name"),
        )
