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
                if m == "median":
                    out[col] = out[col].rolling(window=w).median()
                else:
                    out[col] = out[col].rolling(window=w).mean()
    return out
