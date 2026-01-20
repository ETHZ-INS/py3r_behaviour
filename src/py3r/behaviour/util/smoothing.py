from __future__ import annotations

from typing import Tuple, Dict

import pandas as pd


def apply_smoothing(
    df: pd.DataFrame,
    specs: Dict[str, Dict],
    dims: Tuple[str, ...],
    *,
    smoother=None,
    smoother_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Pure smoothing engine used by Tracking and others.

    - df: DataFrame with columns like "point.x", "point.y", "point.z"
    - specs: mapping point -> {"method": 'median'|'mean', "window": int|None}
    - dims: which coordinate dims to smooth
    - smoother: optional callable(series, point, dim, window, method, df, **kwargs) -> Series
    - smoother_kwargs: optional dict passed to smoother
    """
    smoother_kwargs = smoother_kwargs or {}
    out = df.copy()
    for point, cfg in specs.items():
        w = cfg.get("window")
        if not w:
            continue
        m = cfg.get("method", "median")
        # Capture any extra kwargs (e.g., for Savitzky–Golay like polyorder, mode, etc.)
        extra_kwargs = {k: v for k, v in cfg.items() if k not in ("method", "window")}
        for d in dims:
            col = f"{point}.{d}"
            if col not in out.columns:
                continue
            if smoother is not None:
                out[col] = smoother(
                    out[col],
                    point=point,
                    dim=d,
                    window=w,
                    method=m,
                    df=out,
                    **smoother_kwargs,
                )
            else:
                out[col] = smooth_series(out[col], method=m, window=w, **extra_kwargs)
    return out


def smooth_series(
    series: pd.Series,
    *,
    method: str,
    window: int | None,
    **method_kwargs,
) -> pd.Series:
    """
    Smooth a single pandas Series by method with a given window.
    Supported methods: 'median', 'mean', 'savgol'.
    - For 'savgol', SciPy is required. Series must not contain NaNs.
    """
    if not window or window <= 0:
        return series
    if method == "median":
        return series.rolling(window=window, center=True).median()
    if method == "mean":
        return series.rolling(window=window, center=True).mean()
    if method == "savgol":
        return _smooth_series_savgol(series, window=window, **method_kwargs)
    raise ValueError(f"Unknown smoothing method '{method}'")


def _smooth_series_savgol(
    series: pd.Series,
    *,
    window: int = 11,
    polyorder: int = 3,
    mode: str = "interp",
    nan_policy: str = "segment",
) -> pd.Series:
    """
    Apply Savitzky–Golay filter using scipy.signal.savgol_filter.
    Requirements:
      - SciPy installed
      - window is odd and >= 1
      - polyorder < window
      - NaN handling controlled by nan_policy:
          'error'   -> raise if NaNs present (optional, conservative)
          'segment' -> smooth each contiguous finite segment independently,
                       leave short segments (< window) as-is; preserve NaNs elsewhere (default)
    """
    try:
        from scipy.signal import savgol_filter
    except ImportError as e:
        raise ImportError(
            "Savitzky–Golay requires SciPy. Install with: pip install scipy"
        ) from e
    # Basic validation
    w = int(window)
    if w % 2 == 0:
        w += 1  # ensure odd (minimal handling)
    if polyorder >= w:
        raise ValueError("For Savitzky–Golay, polyorder must be < window")
    has_nan = series.isna().any()
    if not has_nan:
        vals = series.to_numpy(dtype=float)
        smoothed = savgol_filter(
            vals, window_length=w, polyorder=int(polyorder), mode=mode
        )
        return pd.Series(smoothed, index=series.index)
    # NaN handling
    if nan_policy == "error":
        raise ValueError(
            "Savitzky–Golay cannot handle NaNs with nan_policy='error'. Interpolate first or use nan_policy='segment'."
        )
    if nan_policy != "segment":
        raise ValueError("nan_policy must be one of {'error','segment'}")
    # Segment-wise smoothing over finite runs
    out = series.copy()
    mask = series.notna()
    if not mask.any():
        return out  # all NaN
    # Identify contiguous finite segments
    # Use a group id that increments when mask changes or at NaN boundaries
    group_ids = (mask != mask.shift(fill_value=False)).cumsum()
    for gid, is_finite in mask.groupby(group_ids):
        if not is_finite.iloc[0]:
            continue  # this group is NaNs
        idx = is_finite[is_finite].index
        seg = series.loc[idx]
        if len(seg) < w:
            # too short for this window; leave as original (finite)
            continue
        vals = seg.to_numpy(dtype=float)
        seg_sm = savgol_filter(
            vals, window_length=w, polyorder=int(polyorder), mode=mode
        )
        out.loc[idx] = seg_sm
    return out
