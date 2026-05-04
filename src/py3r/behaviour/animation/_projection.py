from __future__ import annotations

import numpy as np


def _is_valid(pix: np.ndarray, idx: int) -> bool:
    return idx < len(pix) and pix[idx, 0] >= 0 and pix[idx, 1] >= 0


def _compute_bounds(
    points_xy: np.ndarray,
    boundary_arrays: list[tuple[str, np.ndarray]],
    axis_arrays: list[tuple[str, np.ndarray]] | None = None,
    pad: float = 0.05,
) -> tuple[float, float, float, float]:
    flat = points_xy.reshape(-1, 2)
    valid = np.isfinite(flat[:, 0]) & np.isfinite(flat[:, 1])
    xs = [flat[valid, 0]] if np.any(valid) else []
    ys = [flat[valid, 1]] if np.any(valid) else []
    all_arrays = list(boundary_arrays) + list(axis_arrays or [])
    for _, arr in all_arrays:
        a = np.asarray(arr, dtype=float)
        if a.ndim != 3 or a.shape[2] != 2 or a.shape[0] == 0 or a.shape[1] == 0:
            continue
        flat_a = a.reshape(-1, 2)
        ok = np.isfinite(flat_a[:, 0]) & np.isfinite(flat_a[:, 1])
        if np.any(ok):
            xs.append(flat_a[ok, 0])
            ys.append(flat_a[ok, 1])
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
    """Build projection parameters for 3D->2D projection."""
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


def _data_to_pixel_float(
    data_xy: np.ndarray,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
    pixel_coords: bool,
) -> np.ndarray:
    """Convert data-space coordinates to float pixel coordinates.

    Unlike ``_coords_to_pixels``, this function does *not* clip to canvas
    bounds or substitute a sentinel for out-of-bounds values.  It is intended
    for computing axis line intersections with the canvas edge.

    Parameters
    ----------
    data_xy : np.ndarray
        Shape ``(n, 2)`` data coordinates.
    width, height : int
        Canvas dimensions in pixels.
    bounds : tuple[float, float, float, float]
        ``(xmin, xmax, ymin, ymax)`` data-space bounds.
    pixel_coords : bool
        If True, ``data_xy`` is already in pixel space and is returned
        unchanged (rounded to float).

    Returns
    -------
    np.ndarray
        Shape ``(n, 2)`` float pixel coordinates.
    """
    arr = np.asarray(data_xy, dtype=float)
    if pixel_coords:
        return arr.copy()
    xmin, xmax, ymin, ymax = bounds
    sx = max(width - 1, 1) / (xmax - xmin)
    sy = max(height - 1, 1) / (ymax - ymin)
    px = (arr[:, 0] - xmin) * sx
    py = (height - 1) - ((arr[:, 1] - ymin) * sy)
    return np.stack([px, py], axis=1)


def _clip_axis_to_canvas(
    p1: np.ndarray,
    p2: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find the two canvas-edge intersection points of an infinite axis.

    The axis is defined by two float pixel-space reference points ``p1`` and
    ``p2`` (which may lie outside the canvas).

    Returns ``None`` if the axis does not intersect the canvas, or if the two
    reference points are coincident.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    d = p2 - p1  # direction vector

    ts: list[float] = []
    eps = 1e-9

    if abs(d[0]) > eps:
        for x_edge in (0.0, float(width - 1)):
            t = (x_edge - p1[0]) / d[0]
            y = p1[1] + t * d[1]
            if -0.5 <= y <= height - 0.5:
                ts.append(t)

    if abs(d[1]) > eps:
        for y_edge in (0.0, float(height - 1)):
            t = (y_edge - p1[1]) / d[1]
            x = p1[0] + t * d[0]
            if -0.5 <= x <= width - 0.5:
                ts.append(t)

    if len(ts) < 2:
        return None

    t_min, t_max = min(ts), max(ts)
    if abs(t_max - t_min) < eps:
        return None

    cp1 = np.round(p1 + t_min * d).astype(np.int32)
    cp2 = np.round(p1 + t_max * d).astype(np.int32)
    return cp1, cp2
