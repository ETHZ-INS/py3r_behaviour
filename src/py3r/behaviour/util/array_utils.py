from __future__ import annotations

import numpy as np


def rescale_array_by_dim(
    arr: np.ndarray,
    *,
    dims: tuple[str, ...],
    factors: dict[str, float],
    dim_axis: int,
    copy: bool = True,
) -> np.ndarray:
    """
    Multiply selected dimensions of an array by per-dimension factors.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    dims : tuple[str, ...]
        Dimension labels aligned to ``dim_axis``.
    factors : dict[str, float]
        Scale factors keyed by dimension label.
    dim_axis : int
        Axis index corresponding to ``dims``.
    copy : bool, default True
        If True, return a copy. If False, modify input in place.
    """
    if dim_axis < 0:
        dim_axis = arr.ndim + dim_axis
    if dim_axis < 0 or dim_axis >= arr.ndim:
        raise ValueError("dim_axis out of range")
    if arr.shape[dim_axis] != len(dims):
        raise ValueError("len(dims) must match arr.shape[dim_axis]")

    out = np.array(arr, copy=True) if copy else arr
    for dim_i, dim in enumerate(dims):
        factor = float(factors.get(dim, 1.0) or 1.0)
        if factor == 1.0:
            continue
        index = [slice(None)] * out.ndim
        index[dim_axis] = dim_i
        out[tuple(index)] *= factor
    return out
