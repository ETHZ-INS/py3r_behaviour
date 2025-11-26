from __future__ import annotations

import json
import os
from typing import Literal

import pandas as pd
import numpy as np


SchemaVersion = 1


def _ensure_dir(dirpath: str, overwrite: bool = False) -> None:
    dirpath = os.path.expanduser(dirpath)
    if os.path.isdir(dirpath):
        if not overwrite:
            raise FileExistsError(
                f"Directory already exists: {dirpath} (set overwrite=True to replace)"
            )
    os.makedirs(dirpath, exist_ok=True)


def write_manifest(dirpath: str, manifest: dict) -> None:
    path = os.path.join(dirpath, "manifest.json")
    def _json_safe(obj):
        # Recursively cast numpy/pandas scalars to builtin types and replace pd.NA with None
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        # numpy arrays -> lists (recursively converted)
        if isinstance(obj, np.ndarray):
            return _json_safe(obj.tolist())
        # numpy/pandas scalar types
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        try:
            # pandas NA sentinel
            import pandas as pd  # local import in case pandas not needed elsewhere
            if obj is pd.NA:
                return None
            # pandas Timestamp/Timedelta
            if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                return str(obj)
            # pandas Index -> list
            if isinstance(obj, pd.Index):
                return _json_safe(obj.tolist())
        except Exception:
            pass
        return obj
    with open(path, "w") as f:
        json.dump(_json_safe(manifest), f, indent=2, allow_nan=False)


def read_manifest(dirpath: str) -> dict:
    path = os.path.join(dirpath, "manifest.json")
    with open(path, "r") as f:
        return json.load(f)


def write_dataframe(
    dirpath: str,
    df: pd.DataFrame | pd.Series,
    *,
    filename: str = "data.parquet",
    format: Literal["parquet", "csv"] = "parquet",
) -> dict:
    os.makedirs(dirpath, exist_ok=True)
    fullpath = os.path.join(dirpath, filename)
    if isinstance(df, pd.Series):
        df_to_write = df.to_frame()
    else:
        df_to_write = df
    if format == "parquet":
        df_to_write.to_parquet(fullpath)
    elif format == "csv":
        df_to_write.to_csv(fullpath)
    else:
        raise ValueError("format must be 'parquet' or 'csv'")
    return {"format": format, "path": filename}


def read_dataframe(dirpath: str, spec: dict) -> pd.DataFrame:
    format = spec.get("format", "parquet")
    relpath = spec["path"]
    fullpath = os.path.join(dirpath, relpath)
    if format == "parquet":
        return pd.read_parquet(fullpath)
    elif format == "csv":
        return pd.read_csv(fullpath, index_col=0)
    else:
        raise ValueError(f"Unknown data format: {format}")


def begin_save(dirpath: str, overwrite: bool) -> str:
    _ensure_dir(dirpath, overwrite=overwrite)
    return os.path.expanduser(dirpath)
