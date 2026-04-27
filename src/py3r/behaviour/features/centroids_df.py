from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class CentroidsDf:
    """
    Lightweight, DataFrame-like wrapper for clustering centroids and their scaling recipe.

    This is a delegating wrapper (not a pd.DataFrame subclass). It behaves like a
    DataFrame for common read-only usage via attribute delegation, but additionally
    carries a ``scaling_recipe`` with everything needed to reproduce the normalisation
    and weighting applied during clustering, for use on future datasets.

    Treat instances as immutable. Most delegated DataFrame operations return a plain
    ``pd.DataFrame`` and will not preserve the recipe. Use ``.to_df()`` if you need
    the underlying DataFrame explicitly.

    The ``scaling_recipe`` schema (version 1)::

        {
            "version": 1,
            "embedding_dict": dict[str, list[int]],
            "columns": list[str],
            "normalize_individual_base": dict[str, bool],      # keyed by base feature name
            "constant_factors": dict[str, float],              # keyed by embedding column
            "impute_medians": dict[str, float] | None,         # None when missing_policy="drop"
        }

    Workflow
    --------
    1. Fit::

           batch, centroids, sf = fc.cluster_embedding(...)
           # centroids is a CentroidsDf

    2. Save::

           centroids.save("my_run/")

    3. Later::

           centroids = CentroidsDf.load("my_run/")

    4. Apply to new data::

           feat.assign_clusters_by_centroids(centroids)
           # embedding and all necessary transformation info are contained in centroids object
    """

    df: pd.DataFrame
    scaling_recipe: dict[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(self.df)}")
        if not isinstance(self.scaling_recipe, dict):
            raise TypeError(f"scaling_recipe must be a dict, got {type(self.scaling_recipe)}")

    # --- DataFrame delegation ------------------------------------------------

    def __getattr__(self, name: str):
        # Only called when attribute is not found on the wrapper itself.
        return getattr(self.df, name)

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self):
        return iter(self.df)

    def __getitem__(self, key):
        return self.df[key]

    def __repr__(self) -> str:
        return repr(self.df)

    def _repr_html_(self) -> str:  # pragma: no cover
        return self.df._repr_html_()

    def __array__(self, dtype=None) -> np.ndarray:  # pragma: no cover
        return np.asarray(self.df, dtype=dtype)

    def to_df(self) -> pd.DataFrame:
        """Return the underlying centroids DataFrame (no copy)."""
        return self.df

    # --- Persistence ---------------------------------------------------------

    @property
    def _meta_payload(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "scaling_recipe": self.scaling_recipe}

    def save(self, path: str | Path) -> Path:
        """
        Persist centroids and scaling recipe to disk.

        If *path* ends with ``/`` or is an existing directory, writes:

        - ``centroids.parquet``
        - ``meta.json``

        Otherwise uses *path* as a stem:

        - ``<stem>.parquet``
        - ``<stem>.json``

        Returns the path of the written parquet file.
        """
        p = Path(path)
        if str(path).endswith(("/", "\\")) or (p.exists() and p.is_dir()):
            out_dir = p
            out_dir.mkdir(parents=True, exist_ok=True)
            df_path = out_dir / "centroids.parquet"
            meta_path = out_dir / "meta.json"
        else:
            out_dir = p.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            df_path = p.with_suffix(".parquet")
            meta_path = p.with_suffix(".json")

        self.df.to_parquet(df_path)
        meta_path.write_text(json.dumps(self._meta_payload, indent=2))
        return df_path

    @classmethod
    def load(cls, path: str | Path) -> CentroidsDf:
        """
        Load a previously saved ``CentroidsDf``.

        Accepts either:

        - a directory containing ``centroids.parquet`` + ``meta.json``
        - a stem path (reads ``<stem>.parquet`` + ``<stem>.json``)
        """
        p = Path(path)
        if p.exists() and p.is_dir():
            df_path = p / "centroids.parquet"
            meta_path = p / "meta.json"
        else:
            df_path = p.with_suffix(".parquet")
            meta_path = p.with_suffix(".json")

        df = pd.read_parquet(df_path)
        meta = json.loads(meta_path.read_text())
        schema_version = int(meta.get("schema_version", 1))
        scaling_recipe = dict(meta.get("scaling_recipe", {}))
        return cls(df=df, scaling_recipe=scaling_recipe, schema_version=schema_version)
