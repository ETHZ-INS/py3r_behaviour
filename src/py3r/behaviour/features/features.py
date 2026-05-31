from __future__ import annotations

import copy
import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon
from sklearn.neighbors import KNeighborsRegressor

from py3r.behaviour.features.assets import _ASSET_KINDS, _LEGACY_ASSET_KINDS
from py3r.behaviour.features.axis import DynamicAxis, StaticAxis
from py3r.behaviour.features.boundary import DynamicBoundary, StaticBoundary
from py3r.behaviour.features.features_result import FeaturesResult
from py3r.behaviour.tracking.tracking import Tracking
from py3r.behaviour.util import series_utils
from py3r.behaviour.util.array_utils import rescale_array_by_dim
from py3r.behaviour.util.bmicro_utils import (
    predict_knn_on_embedding,
    train_knn_from_embeddings,
)
from py3r.behaviour.util.collection_utils import _Indexer
from py3r.behaviour.util.dataframe_utils import coarse_grain_dataframe, point_to_axis_distance
from py3r.behaviour.util.dev_utils import dev_mode
from py3r.behaviour.util.io_utils import (
    SchemaVersion,
    begin_save,
    read_dataframe,
    read_manifest,
    write_dataframe,
    write_manifest,
)
from py3r.behaviour.util.missing_tolerance import impute_frame
from py3r.behaviour.util.series_utils import (
    apply_normalization_to_df,
    compose_state_from_boolean_sources,
    normalize_df,
)
from py3r.behaviour.util.smoothing import smooth_series

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.neighbors import KNeighborsRegressor

    from py3r.behaviour.animation.animation_stream import AnimationStream
    from py3r.behaviour.classifier import BaseClassifier
    from py3r.behaviour.features.centroids_df import CentroidsDf
    from py3r.behaviour.summary.summary import Summary

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _axis_boundary_intersections(
    A: np.ndarray,
    B: np.ndarray,
    verts: np.ndarray,
    zones: set[str],
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    """Per-frame boolean: does the axis intersect the boundary in any of the given zones?

    Parameters
    ----------
    A, B : np.ndarray, shape (n, 2)
        Axis reference points per frame.
    verts : np.ndarray
        Either shape ``(N_verts, 2)`` for a static boundary or
        ``(n, N_verts, 2)`` for a dynamic boundary.
    zones : set of {"front", "within", "behind"}
        Which zones count as an intersection.  Zones are defined by the scalar
        projection parameter *t* of the intersection point onto A→B:

        - ``"behind"``: t ≤ 0  (at or before A)
        - ``"within"``: 0 < t < 1  (strictly between A and B)
        - ``"front"``:  t ≥ 1  (at or beyond B)
    eps : float
        Tolerance for near-parallel axis/edge pairs (treated as no intersection).

    Returns
    -------
    np.ndarray of bool, shape (n,)
    """
    static_boundary = verts.ndim == 2
    n_verts = verts.shape[0] if static_boundary else verts.shape[1]
    d = B - A  # (n, 2)
    result = np.zeros(len(A), dtype=bool)

    for i in range(n_verts):
        if static_boundary:
            V0 = verts[i]
            V1 = verts[(i + 1) % n_verts]
            e = V1 - V0  # (2,)
            f = V0 - A  # (n, 2)
            det = d[:, 1] * e[0] - d[:, 0] * e[1]
            t_num = e[0] * f[:, 1] - e[1] * f[:, 0]
            s_num = d[:, 0] * f[:, 1] - d[:, 1] * f[:, 0]
        else:
            V0 = verts[:, i, :]
            V1 = verts[:, (i + 1) % n_verts, :]
            e = V1 - V0  # (n, 2)
            f = V0 - A  # (n, 2)
            det = d[:, 1] * e[:, 0] - d[:, 0] * e[:, 1]
            t_num = e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]
            s_num = d[:, 0] * f[:, 1] - d[:, 1] * f[:, 0]

        nonzero = np.abs(det) > eps
        with np.errstate(invalid="ignore", divide="ignore"):
            t_val = np.where(nonzero, t_num / det, np.nan)
            s_val = np.where(nonzero, s_num / det, np.nan)

        on_edge = nonzero & (s_val >= 0.0) & (s_val <= 1.0)
        if "behind" in zones:
            result |= on_edge & (t_val <= 0.0)
        if "within" in zones:
            result |= on_edge & (t_val > 0.0) & (t_val < 1.0)
        if "front" in zones:
            result |= on_edge & (t_val >= 1.0)

    return result


class Features:
    """generates features from a pre-processed Tracking object."""

    def __init__(self, tracking: Tracking) -> None:
        self.tracking = tracking
        self.data = pd.DataFrame()
        self.meta = dict()
        self._assets: dict[str, Any] = {}
        self.handle = tracking.handle
        if "usermeta" in tracking.meta:
            self.meta["usermeta"] = tracking.meta["usermeta"]

        if "rescale_distance_method" not in self.tracking.meta.keys():
            warnings.warn(
                "distance has not been calibrated on these tracking data. "
                "some methods will be unavailable",
                stacklevel=2,
            )

    @property
    def tags(self) -> dict:
        """Tags delegate to the underlying Tracking — single source of truth."""
        return self.tracking.tags

    @tags.setter
    def tags(self, value: dict) -> None:
        self.tracking.tags = value

    def add_tag(self, tagname: str, tagvalue: str, overwrite: bool = False) -> None:
        """Add or update a tag. Delegates to the underlying Tracking."""
        self.tracking.add_tag(tagname, tagvalue, overwrite=overwrite)

    # Full round-trip persistence
    def save(
        self,
        dirpath: str,
        *,
        data_format: str = "parquet",
        overwrite: bool = False,
    ) -> None:
        """
        Save this Features object (and its nested Tracking) to a self-describing directory.

        Examples
        --------
        ```pycon
        >>> import tempfile, os
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> # add a trivial feature so data is not empty
        >>> s = pd.Series(range(len(t.data)), index=t.data.index)
        >>> f.store(s, 'counter', meta={})
        >>> with tempfile.TemporaryDirectory() as d:
        ...     f.save(d, data_format='csv', overwrite=True)
        ...     os.path.exists(os.path.join(d, 'manifest.json'))
        True

        ```
        """
        target = begin_save(dirpath, overwrite)
        # Save own data
        data_spec = write_dataframe(
            target,
            self.data,
            filename="data.parquet" if data_format == "parquet" else "data.csv",
            format=data_format,
        )
        # Save nested tracking in a subfolder
        tracking_sub = os.path.join(target, "tracking")
        self.tracking.save(tracking_sub, data_format=data_format, overwrite=True)
        manifest = {
            "schema_version": SchemaVersion,
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
            "handle": self.handle,
            "tags": self.tags,
            "meta": self.meta,
            "assets": self._serialize_assets(),
            "data": data_spec,
            "tracking_path": "tracking",
        }
        write_manifest(target, manifest)

    @classmethod
    def load(cls, dirpath: str) -> Features:
        """
        Load a Features object previously saved with save().

        Examples
        --------
        ```pycon
        >>> import tempfile, os
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> f.store(pd.Series(range(len(t.data)), index=t.data.index), 'counter', meta={})
        >>> with tempfile.TemporaryDirectory() as d:
        ...     f.save(d, data_format='csv', overwrite=True)
        ...     f2 = Features.load(d)
        >>> isinstance(f2, Features) and 'counter' in f2.data.columns
        True

        ```
        """
        manifest = read_manifest(dirpath)
        df = read_dataframe(dirpath, manifest["data"])
        tracking = Tracking.load(os.path.join(dirpath, manifest["tracking_path"]))
        obj = cls(tracking)
        obj.data = df
        obj.meta = manifest.get("meta", {})
        obj._assets = obj._deserialize_assets(manifest.get("assets"))
        obj.handle = manifest.get("handle", obj.handle)
        obj.tags = manifest.get("tags", obj.tags)
        return obj

    def copy(self) -> Features:
        """Creates an independent copy of this Features object.

        The returned object shares no mutable state with the original:
        Tracking is copied via Tracking.copy(), the features DataFrame
        via DataFrame.copy(), and meta/tags via deepcopy.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> f_copy = f.copy()
        >>> f_copy.handle == f.handle
        True
        >>> f_copy.tracking.data is not f.tracking.data
        True

        ```
        """
        result = type(self)(self.tracking.copy())
        result.data = self.data.copy()
        result.meta = copy.deepcopy(self.meta)
        result._assets = copy.deepcopy(self._assets)
        result.handle = self.handle
        result.tags = copy.deepcopy(self.tags)
        return result

    def coarse_grain(
        self: Self,
        window: int,
        method: Literal["mean", "median", "min", "max"] = "mean",
        non_numeric: Literal["drop", "nan", "first", "mode", "error"] = "drop",
        keep_assets: bool = True,
    ) -> Self:
        """
        Coarse-grain feature data over fixed, non-overlapping windows.

        Applies the same aggregation to both ``Features.data`` and the backing
        ``Tracking`` object so row counts and index alignment remain consistent.
        ``fps`` is divided by ``window`` to reflect the new effective frame rate.
        A ``"coarse_grain"`` entry is appended to ``meta["transforms"]``.

        Args:
            window: Number of consecutive rows to collapse into one.
            method: Aggregation applied to numeric feature columns within each window.
            non_numeric: How to handle non-numeric feature columns (e.g. string state
                labels). Pass ``"mode"`` to keep the most-frequent value per window,
                which is appropriate for categorical columns.
            keep_assets: If ``True``, assets (e.g. boundary objects) are deep-copied to
                the result. Set to ``False`` to avoid copying large assets when they
                are not needed at the coarser scale.

        Returns:
            New ``Features`` (or subclass) object with ``len(data) // window``
                rows and reduced fps.

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> vals = pd.Series(range(len(t.data)), index=t.data.index, dtype=float)
        >>> f.store(vals, 'counter', meta={})
        >>> len(f.data), f.tracking.meta['fps']
        (5, 30.0)

        ```

        Coarse-graining by 2 halves the row count and fps for both feature
        data and the backing Tracking:

        ```pycon
        >>> f2 = f.coarse_grain(2)
        >>> len(f2.data)
        3
        >>> f2.tracking.meta['fps']
        15.0
        >>> f2.handle
        'ex'

        ```

        The 5-row input produces 3 windows: two complete (rows 0–1, rows 2–3)
        and one partial (row 4 alone).  Incomplete trailing windows are
        retained — the single-row window aggregates to the row's own value:

        ```pycon
        >>> list(f2.data['counter'])
        [0.5, 2.5, 4.0]

        ```

        The backing Tracking is coarse-grained in sync — row counts match:

        ```pycon
        >>> len(f2.tracking.data) == len(f2.data)
        True

        ```

        Categorical columns are preserved with ``non_numeric='mode'``:

        ```pycon
        >>> labels = pd.Series(['A','A','B','B','A'], index=t.data.index)
        >>> f.store(labels, 'state', meta={})
        >>> f_mode = f.coarse_grain(2, non_numeric='mode')
        >>> list(f_mode.data['state'])
        ['A', 'B', 'A']

        ```

        The transform is recorded in meta:

        ```pycon
        >>> f2.meta['transforms'][-1]
        {'type': 'coarse_grain', 'window': 2, 'method': 'mean'}

        ```
        """
        coarse_tracking = self.tracking.coarse_grain(
            window=window,
            method=method,
            non_numeric=non_numeric,
        )
        coarse = type(self)(coarse_tracking)

        coarse.data = coarse_grain_dataframe(
            self.data,
            window=window,
            method=method,
            non_numeric=non_numeric,
        )

        coarse.meta = copy.deepcopy(self.meta)
        coarse.meta["transforms"] = [
            *coarse.meta.get("transforms", []),
            {
                "type": "coarse_grain",
                "window": int(window),
                "method": method,
            },
        ]

        coarse._assets = copy.deepcopy(self._assets) if keep_assets else {}
        coarse.handle = self.handle
        coarse.tags = copy.deepcopy(self.tags)
        return coarse

    def to_summary(self) -> Summary:
        """
        Create a `Summary` object from this `Features` object.

        This is a convenience wrapper around `Summary(self)`.

        Returns:
            A new summary object linked to this features object.

        Examples
        --------
        ```pycon
            >>> from py3r.behaviour.util.docdata import data_path
            >>> from py3r.behaviour.tracking.tracking import Tracking
            >>> from py3r.behaviour.features.features import Features
            >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
            ...     t = Tracking.from_dlc(str(p), handle='demo', fps=30)
            >>> f = Features(t)
            >>> s = f.to_summary()
            >>> from py3r.behaviour.summary.summary import Summary
            >>> isinstance(s, Summary)
            True
            >>> s.handle
            'demo'

            ```
        """
        from py3r.behaviour.summary.summary import Summary

        return Summary(self)

    @classmethod
    def concat(
        cls,
        features_list: list[Features],
        *,
        handle: str | None = None,
        reindex: Literal["rezero", "follow_previous", "keep_original"] = "follow_previous",
    ) -> Features:
        """
        Concatenate multiple Features objects along the time (frame) axis.

        This method concatenates both the underlying Tracking data and the
        computed features DataFrame. All Features objects must have:
        - Matching fps (in underlying Tracking)
        - Identical tracking column names
        - Identical feature column names

        Args:
            features_list: List of Features objects to concatenate, in temporal order.
            handle: Handle for the concatenated object. If None, uses first object's handle.
            reindex: How to handle frame indices. ``"rezero"`` reindexes all frames
                starting from 0. ``"follow_previous"`` continues from where the previous
                chunk ended. ``"keep_original"`` leaves indices untouched; duplicates
                are allowed.

        Returns:
            A new Features object containing all frames from input objects.

        Raises:
            ValueError: If features_list is empty, fps values don't match, or columns differ.

        Note:
            For context-dependent features (normalization, embeddings with temporal
            windows, etc.), consider whether you need to recompute features on
            concatenated Tracking data rather than concatenating pre-computed features.

        Examples
        --------
        Concatenate two features objects:

        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t1 = Tracking.from_dlc(str(p), handle='ex1', fps=30)
        ...     t2 = Tracking.from_dlc(str(p), handle='ex2', fps=30)
        >>> f1, f2 = Features(t1), Features(t2)
        >>> # Add a simple feature to both
        >>> f1.store(pd.Series([1,2,3,4,5], index=t1.data.index), 'val', meta={})
        >>> f2.store(pd.Series([6,7,8,9,10], index=t2.data.index), 'val', meta={})
        >>> combined = Features.concat([f1, f2], handle='combined')
        >>> len(combined.data) == len(f1.data) + len(f2.data)
        True
        >>> list(combined.data['val'])
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        ```

        Verify tracking is also concatenated:

        ```pycon
        >>> len(combined.tracking.data) == len(t1.data) + len(t2.data)
        True

        ```

        Concatenation metadata is recorded:

        ```pycon
        >>> 'concat' in combined.meta
        True
        >>> combined.meta['concat']['n_chunks']
        2

        ```
        """
        if not features_list:
            raise ValueError("Cannot concatenate empty list of Features objects")

        if len(features_list) == 1:
            result = features_list[0].copy()
            if handle is not None:
                result.handle = handle
            return result

        # Keys in meta that are expected to differ between chunks (not validated)
        _meta_ignore_keys = {"concat"}

        # Validate feature column consistency
        reference_cols = list(features_list[0].data.columns)
        for i, f in enumerate(features_list[1:], start=1):
            if list(f.data.columns) != reference_cols:
                raise ValueError(
                    f"Feature column mismatch: Features[0] has columns {reference_cols}, "
                    f"but Features[{i}] has columns {list(f.data.columns)}"
                )

        # Validate Features meta consistency (excluding ignored keys)
        # Note: Tracking meta is validated by Tracking.concat
        ref_meta = features_list[0].meta
        for i, f in enumerate(features_list[1:], start=1):
            ref_keys = set(ref_meta.keys()) - _meta_ignore_keys
            f_keys = set(f.meta.keys()) - _meta_ignore_keys
            if ref_keys != f_keys:
                raise ValueError(
                    f"Features meta key mismatch: Features[0] has keys {ref_keys}, "
                    f"but Features[{i}] has keys {f_keys}"
                )
            for key in ref_keys:
                if ref_meta[key] != f.meta[key]:
                    raise ValueError(
                        f"Features meta value mismatch for key '{key}': "
                        f"Features[0] has {ref_meta[key]!r}, "
                        f"but Features[{i}] has {f.meta[key]!r}"
                    )

        # Check handle consistency - warn if differs, use first
        handles = [f.handle for f in features_list]
        if len(set(handles)) > 1 and handle is None:
            warnings.warn(
                f"Handles differ across Features objects: {handles}. "
                f"Using first handle '{handles[0]}'. "
                f"Pass handle= parameter to specify explicitly.",
                stacklevel=2,
            )

        # Check tags consistency - warn if differs, use first
        first_tags = features_list[0].tags
        tags_differ = any(f.tags != first_tags for f in features_list[1:])
        if tags_differ:
            warnings.warn(
                f"Tags differ across Features objects. Using tags from first object: {first_tags}",
                stacklevel=2,
            )

        # Concatenate underlying Tracking objects (this validates Tracking meta)
        trackings = [f.tracking for f in features_list]
        result_handle = handle if handle is not None else features_list[0].handle
        combined_tracking = Tracking.concat(trackings, handle=result_handle, reindex=reindex)

        # Build concatenated features DataFrame with matching indices
        dfs = []
        chunk_info = combined_tracking.meta["concat"]["chunk_boundaries"]

        for i, f in enumerate(features_list):
            df = f.data.copy()
            info = chunk_info[i]
            n_frames = len(df)

            # Reindex to match the concatenated tracking
            new_start = info["concat_start_frame"]
            df.index = pd.RangeIndex(new_start, new_start + n_frames)
            dfs.append(df)

        combined_features_data = pd.concat(dfs, axis=0)
        combined_features_data.index.name = "frame"

        # Create result object
        result = cls(combined_tracking)
        result.data = combined_features_data

        # Build metadata (from first, add concat info)
        result.meta = copy.deepcopy(features_list[0].meta)
        result._assets = copy.deepcopy(features_list[0]._assets)
        result.meta["concat"] = {
            "n_chunks": len(features_list),
            "source_handles": handles,
            "reindexed": reindex,
        }

        # Use first object's tags
        result.tags = copy.deepcopy(first_tags)

        result.handle = result_handle
        return result

    def distance_between(self, point1: str, point2: str, dims=("x", "y")) -> FeaturesResult:
        """
        Returns distance from point1 to point2.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> res = f.distance_between('p1','p2')
        >>> isinstance(res, pd.Series) and len(res) == len(t.data)
        True

        ```
        """
        if "rescale_distance_method" not in self.tracking.meta.keys():
            warnings.warn("distance has not been calibrated", stacklevel=2)
        if "smoothing" not in self.tracking.meta.keys():
            warnings.warn("tracking data have not been smoothed", stacklevel=2)

        obs_distance = self.tracking.distance_between(point1, point2, dims=dims)
        name = f"distance_between_{point1}_and_{point2}_in_{''.join(dims)}"
        meta = {
            "function": "distance_between",
            "point1": point1,
            "point2": point2,
            "dims": dims,
        }
        return FeaturesResult(obs_distance, self, name, meta)

    def within_distance(
        self, point1: str, point2: str, distance: float, dims=("x", "y")
    ) -> FeaturesResult:
        """
        Returns True for frames where point1 is within specified distance of point2
        NA is propagated where inputs are missing (pd.NA).

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> res = f.within_distance('p1','p2', distance=15.0)
        >>> bool((isinstance(res, pd.Series) and res.notna().any()))
        True

        ```
        """
        obs_distance = self.distance_between(point1, point2, dims=dims)
        # Propagate NA: comparisons with missing distances should yield pd.NA
        mask = obs_distance.notna()
        result = pd.Series(pd.NA, index=obs_distance.index, dtype="boolean")
        result[mask] = (obs_distance[mask] <= distance).astype("boolean")
        name = f"within_distance_{point1}_to_{point2}_leq_{distance}_in_{''.join(dims)}"
        meta = {
            "function": "within_distance",
            "point1": point1,
            "point2": point2,
            "distance": distance,
            "dims": dims,
        }
        return FeaturesResult(result, self, name, meta)

    def get_point_median(self, point: str, dims=("x", "y")) -> tuple:
        """
        Return the per-dimension median coordinate for a tracked point.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> med = f.get_point_median('p1', dims=('x','y'))
        >>> isinstance(med, tuple) and len(med) == 2
        True

        ```
        """
        meds: list[float] = []
        for dim in dims:
            vals = self.tracking.data[point + "." + dim].to_numpy()
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                meds.append(np.nan)
            else:
                meds.append(float(np.median(finite)))
        return tuple(meds)

    def define_boundary(
        self,
        points: list[str],
        scaling: float,
        scaling_y: float = None,
        centre: str | list[str] = None,
    ) -> list[tuple[float, float]]:
        """Deprecated: use define_static_boundary or define_dynamic_boundary instead."""
        raise NotImplementedError(
            "Features.define_boundary() was removed; use Features.define_static_boundary() "
            "or Features.define_dynamic_boundary() instead."
        )

    # ------------------------------------------------------------------
    # Generic asset registry (flat dict keyed by name)
    # ------------------------------------------------------------------

    def _serialize_assets(self) -> dict:
        return {name: asset.to_dict() for name, asset in self._assets.items()}

    def _deserialize_assets(self, payload: dict | None) -> dict[str, Any]:
        payload = payload or {}
        result: dict[str, Any] = {}

        # Backward compatibility: files saved before the asset refactor stored
        # boundaries under a nested "boundaries" key with kinds "static"/"dynamic".
        for name, data in payload.get("boundaries", {}).items():
            kind = data.get("kind")
            cls = _ASSET_KINDS.get(kind) or _LEGACY_ASSET_KINDS.get(kind)
            if cls is None:
                raise ValueError(f"Unknown asset kind {kind!r} in saved assets.")
            result[name] = cls.from_dict(data)

        # Current flat format: each top-level key is an asset name.
        for name, data in payload.items():
            if name == "boundaries":
                continue
            kind = data.get("kind")
            cls = _ASSET_KINDS.get(kind)
            if cls is None:
                raise ValueError(f"Unknown asset kind {kind!r} in saved assets.")
            result[name] = cls.from_dict(data)

        return result

    def _register_asset(self, asset, *, name: str | None, overwrite: bool):
        if name is None:
            return asset
        if name in self._assets and not overwrite:
            raise ValueError(
                f"Asset with name {name!r} already exists. Set overwrite=True to replace it."
            )
        named = asset.with_name(name)
        self._assets[name] = named
        return named

    def get_asset(self, name: str):
        """Return a named geometric asset (boundary or line) by name.

        Raises:
            KeyError: If no asset with ``name`` is registered.
        """
        try:
            return self._assets[name]
        except KeyError:
            available = list(self._assets)
            raise KeyError(f"No asset {name!r} registered. Available: {available}") from None

    def list_assets(self) -> pd.DataFrame:
        """Return a table of all named geometric assets on this Features object.

        Returns:
            Indexed by asset name, columns: ``asset_type``, ``dims``, ``n_points``.
        """
        rows = []
        for name, asset in self._assets.items():
            n = (
                len(asset.vertices)
                if isinstance(asset, (StaticBoundary, StaticAxis))
                else len(asset.points)
            )
            rows.append(
                {
                    "name": name,
                    "asset_type": type(asset).__name__,
                    "dims": asset.dims,
                    "n_points": n,
                }
            )
        if not rows:
            return pd.DataFrame(columns=["asset_type", "dims", "n_points"])
        return pd.DataFrame(rows).set_index("name")

    def _resolve_anchor(self, points: list[str], dims: tuple[str, str], anchor):
        if anchor is None:
            anchor_points = points
        elif isinstance(anchor, str):
            anchor_points = [anchor]
        elif isinstance(anchor, list):
            if len(anchor) == 0:
                raise ValueError("anchor list cannot be empty.")
            anchor_points = anchor
        else:
            raise ValueError("anchor must be None, a point name, or list of point names.")
        meds = [self.get_point_median(point, dims=dims) for point in anchor_points]
        d1 = float(np.mean([m[0] for m in meds]))
        d2 = float(np.mean([m[1] for m in meds]))
        return (d1, d2), tuple(anchor_points)

    @staticmethod
    def _scale_points_2d(
        points: list[tuple[float, float]],
        anchor: tuple[float, float],
        scale_dim1: float,
        scale_dim2: float,
    ) -> list[tuple[float, float]]:
        ax1, ax2 = anchor
        return [
            (
                ax1 + (p1 - ax1) * scale_dim1,
                ax2 + (p2 - ax2) * scale_dim2,
            )
            for p1, p2 in points
        ]

    def define_static_boundary(
        self,
        points: list[str],
        *,
        dims: tuple[str, str] = ("x", "y"),
        anchor: str | list[str] | None = None,
        scale_dim1: float = 1.0,
        scale_dim2: float = 1.0,
        name: str | None = None,
        overwrite: bool = False,
    ) -> StaticBoundary:
        """
        Define a static boundary from point medians and optional scaling.

        Scaling is applied independently in each selected dimension about ``anchor``.
        """
        if len(points) < 3:
            raise ValueError("Static boundary requires at least 3 point names.")
        pointmedians = [self.get_point_median(point, dims=dims) for point in points]
        anchor_coords, anchor_points = self._resolve_anchor(points, dims, anchor)
        vertices = self._scale_points_2d(pointmedians, anchor_coords, scale_dim1, scale_dim2)
        boundary = StaticBoundary(
            vertices=tuple((float(x), float(y)) for x, y in vertices),
            dims=(dims[0], dims[1]),
            source_points=tuple(points),
            anchor_points=anchor_points,
            scale_dim1=float(scale_dim1),
            scale_dim2=float(scale_dim2),
            name=name,
        )
        return self._register_asset(boundary, name=name, overwrite=overwrite)

    def define_dynamic_boundary(
        self,
        points: list[str],
        *,
        dims: tuple[str, str] = ("x", "y"),
        anchor: str | list[str] | None = None,
        scale_dim1: float = 1.0,
        scale_dim2: float = 1.0,
        name: str | None = None,
        overwrite: bool = False,
    ) -> DynamicBoundary:
        """Define a dynamic boundary from ordered point names and optional scaling."""
        if len(points) < 3:
            raise ValueError("Dynamic boundary requires at least 3 point names.")
        # validate anchor eagerly for clearer user feedback
        _anchor_coords, anchor_points = self._resolve_anchor(points, dims, anchor)
        boundary = DynamicBoundary(
            points=tuple(points),
            dims=(dims[0], dims[1]),
            anchor_points=anchor_points,
            scale_dim1=float(scale_dim1),
            scale_dim2=float(scale_dim2),
            name=name,
        )
        return self._register_asset(boundary, name=name, overwrite=overwrite)

    def import_static_boundary(
        self,
        vertices: list[tuple[float, float]],
        *,
        dims: tuple[str, str] = ("x", "y"),
        name: str | None = None,
        overwrite: bool = False,
    ) -> StaticBoundary:
        """Escape hatch: import a precomputed static polygon in selected dims."""
        if len(vertices) < 3:
            raise ValueError("Imported static boundary requires at least 3 vertices.")
        verts = tuple((float(x), float(y)) for x, y in vertices)
        boundary = StaticBoundary(vertices=verts, dims=(dims[0], dims[1]), name=name)
        return self._register_asset(boundary, name=name, overwrite=overwrite)

    # ------------------------------------------------------------------
    # Axis asset factories
    # ------------------------------------------------------------------

    def define_static_axis(
        self,
        point1: str,
        point2: str,
        *,
        dims: tuple[str, ...] = ("x", "y"),
        offset: float = 0.0,
        name: str | None = None,
        overwrite: bool = False,
    ) -> StaticAxis:
        """Define a static axis from the medians of two keypoints.

        The axis is fixed in space: median coordinates are computed once from
        the full tracking session.  The offset is baked into the stored
        reference points at definition time.  The axis is always treated as
        infinite (no endpoints) in both distance computations and rendering.

        Args:
            point1: First keypoint defining the axis direction.
            point2: Second keypoint defining the axis direction.
            dims: Coordinate dimensions. Any number of dims is supported
                (e.g. ``("x", "y", "z")`` for 3-D axis distance).
            offset: Shift both reference points perpendicularly by this amount.
                Positive is to the right when facing from ``point1`` to ``point2``.
                Only supported for 2-D axes.
            name: If given, register the axis under this name.
            overwrite: Allow replacing an existing asset with the same name.
        """
        from py3r.behaviour.features.axis import _transform_axis_endpoints

        medians = [self.get_point_median(p, dims=dims) for p in (point1, point2)]
        A = np.array(medians[0], dtype=float)
        B = np.array(medians[1], dtype=float)
        A_t, B_t = _transform_axis_endpoints(A, B, offset=offset)
        axis = StaticAxis(
            vertices=(tuple(float(c) for c in A_t), tuple(float(c) for c in B_t)),
            dims=tuple(dims),
            source_points=(point1, point2),
            name=name,
        )
        return self._register_asset(axis, name=name, overwrite=overwrite)

    def define_dynamic_axis(
        self,
        point1: str,
        point2: str,
        *,
        dims: tuple[str, ...] = ("x", "y"),
        offset: float = 0.0,
        name: str | None = None,
        overwrite: bool = False,
    ) -> DynamicAxis:
        """Define a dynamic axis from two keypoint names.

        Reference-point coordinates are resolved per frame at compute time,
        with offset applied during each resolution.  The axis is always treated
        as infinite in both distance computations and rendering.

        Args:
            point1: First keypoint defining the axis direction.
            point2: Second keypoint defining the axis direction.
            dims: Coordinate dimensions.
            offset: Per-frame perpendicular displacement. Positive is to the right
                when facing from ``point1`` to ``point2``. Only supported for 2-D axes.
            name: If given, register the axis under this name.
            overwrite: Allow replacing an existing asset with the same name.
        """
        self.tracking._assert_valid_point(point1)
        self.tracking._assert_valid_point(point2)
        axis = DynamicAxis(
            points=(point1, point2),
            dims=tuple(dims),
            offset=offset,
            name=name,
        )
        return self._register_asset(axis, name=name, overwrite=overwrite)

    def import_static_axis(
        self,
        vertices: list[tuple[float, ...]],
        *,
        dims: tuple[str, ...] = ("x", "y"),
        name: str | None = None,
        overwrite: bool = False,
    ) -> StaticAxis:
        """Import a static axis from explicit reference-point coordinates.

        Args:
            vertices: Exactly two coordinate tuples in ``dims`` space.
            dims: Coordinate dimensions.
            name: If given, register the axis under this name.
            overwrite: Allow replacing an existing asset with the same name.
        """
        if len(vertices) != 2:
            raise ValueError(f"An axis requires exactly 2 reference points; got {len(vertices)}.")
        verts = tuple(tuple(float(c) for c in v) for v in vertices)
        axis = StaticAxis(vertices=verts, dims=tuple(dims), name=name)
        return self._register_asset(axis, name=name, overwrite=overwrite)

    def get_boundary(self, name: str) -> StaticBoundary | DynamicBoundary:
        """Return a named boundary asset.

        Raises ``KeyError`` if the name is not registered, ``TypeError`` if the
        registered asset is not a boundary.
        """
        asset = self.get_asset(name)
        if not isinstance(asset, (StaticBoundary, DynamicBoundary)):
            raise TypeError(
                f"Asset {name!r} is a {type(asset).__name__}, not a boundary. "
                "Use get_asset() to retrieve non-boundary assets."
            )
        return asset

    def list_boundaries(self) -> pd.DataFrame:
        """Return a compact table of named boundary assets on this Features object."""
        rows = []
        for name, asset in self._assets.items():
            if isinstance(asset, StaticBoundary):
                rows.append(
                    {
                        "name": name,
                        "kind": "static",
                        "n_points": len(asset.vertices),
                        "has_vertices": True,
                    }
                )
            elif isinstance(asset, DynamicBoundary):
                rows.append(
                    {
                        "name": name,
                        "kind": "dynamic",
                        "n_points": len(asset.points),
                        "has_vertices": False,
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["kind", "n_points", "has_vertices"])
        return pd.DataFrame(rows).set_index("name")

    def _resolve_boundary_ref(self, boundary) -> StaticBoundary | DynamicBoundary:
        resolved = self.get_asset(boundary) if isinstance(boundary, str) else boundary
        if not isinstance(resolved, (StaticBoundary, DynamicBoundary)):
            raise TypeError(
                "boundary must be a boundary name, StaticBoundary, or DynamicBoundary. "
                f"Got {type(resolved).__name__}."
            )
        return resolved

    def _resolve_axis_ref(self, axis) -> StaticAxis | DynamicAxis:
        resolved = self.get_asset(axis) if isinstance(axis, str) else axis
        if not isinstance(resolved, (StaticAxis, DynamicAxis)):
            raise TypeError(
                "axis must be an axis name, StaticAxis, or DynamicAxis. "
                f"Got {type(resolved).__name__}."
            )
        return resolved

    @staticmethod
    def _is_legacy_point_name_list(boundary) -> bool:
        return isinstance(boundary, list) and len(boundary) > 0 and isinstance(boundary[0], str)

    @staticmethod
    def _is_legacy_vertex_list(boundary) -> bool:
        return isinstance(boundary, list) and len(boundary) > 0 and not isinstance(boundary[0], str)

    @staticmethod
    def _short_boundary_id(boundary):
        b = [str(x) for x in boundary]
        if len(b) <= 4:
            return "_".join(b)
        return "_".join(b[:2] + ["..."] + b[-2:])

    def _within_boundary_static_impl(
        self,
        point: str,
        *,
        boundary_vertices: list[tuple[float, float]],
        dims: tuple[str, str],
        boundary_label: str,
        boundary_meta,
    ) -> FeaturesResult:
        if len(boundary_vertices) < 3:
            raise Exception("boundary encloses no area")
        boundary_has_nan = any(pd.isna(bx) or pd.isna(by) for bx, by in boundary_vertices)
        name = f"within_boundary_static_{point}_in_{boundary_label}"
        meta = {
            "function": "within_boundary",
            "point": point,
            "boundary": boundary_meta,
        }

        df = self.tracking.data
        px = df[point + "." + dims[0]].to_numpy(dtype=float)
        py = df[point + "." + dims[1]].to_numpy(dtype=float)
        valid = ~(np.isnan(px) | np.isnan(py))

        if boundary_has_nan:
            result = pd.Series(pd.array([pd.NA] * len(df), dtype="boolean"), index=df.index)
        else:
            poly = Polygon(boundary_vertices)
            result = pd.Series(pd.array([pd.NA] * len(df), dtype="boolean"), index=df.index)
            if valid.any():
                pts = shapely.points(px[valid], py[valid])
                result[valid] = shapely.contains(poly, pts)
        return FeaturesResult(result, self, name, meta)

    def _within_boundary_dynamic_impl(
        self,
        point: str,
        *,
        boundary_points: list[str],
        dims: tuple[str, str],
        scale_dim1: float,
        scale_dim2: float,
        anchor_points: list[str] | None,
        boundary_label: str,
        boundary_meta,
    ) -> FeaturesResult:
        if len(boundary_points) < 3:
            raise Exception("boundary encloses no area")

        name = f"within_boundary_dynamic_{point}_in_{boundary_label}"
        meta = {
            "function": "within_boundary",
            "point": point,
            "boundary": boundary_meta,
        }

        df = self.tracking.data
        px = df[point + "." + dims[0]].to_numpy(dtype=float)
        py = df[point + "." + dims[1]].to_numpy(dtype=float)
        boundary_obj = DynamicBoundary(
            points=tuple(boundary_points),
            dims=dims,
            anchor_points=tuple(anchor_points) if anchor_points is not None else None,
            scale_dim1=scale_dim1,
            scale_dim2=scale_dim2,
        )
        verts = boundary_obj.to_numpy_per_frame(df)  # (n, N_verts, 2), scaling applied

        valid = ~(np.isnan(px) | np.isnan(py) | np.any(np.isnan(verts), axis=(1, 2)))
        result = pd.Series(pd.array([pd.NA] * len(df), dtype="boolean"), index=df.index)
        if valid.any():
            coords = verts[valid]
            polys = shapely.polygons(shapely.linearrings(coords))
            pts = shapely.points(px[valid], py[valid])
            result[valid] = shapely.contains(polys, pts)
        return FeaturesResult(result, self, name, meta)

    def within_boundary_static(
        self, point: str, boundary, boundary_name: str = None
    ) -> FeaturesResult:
        raise NotImplementedError(
            "Features.within_boundary_static() was removed; use "
            "Features.within_boundary(point, boundary) with a StaticBoundary "
            "or stored boundary name instead."
        )

    def within_boundary_dynamic(
        self, point: str, boundary, boundary_name: str = None
    ) -> FeaturesResult:
        raise NotImplementedError(
            "Features.within_boundary_dynamic() was removed; use "
            "Features.within_boundary(point, boundary) with a DynamicBoundary "
            "or stored boundary name instead."
        )

    def within_boundary(self, point: str, boundary) -> FeaturesResult:
        """
        Main boundary inclusion API.

        Accepts a ``StaticBoundary`` or ``DynamicBoundary`` (or a stored boundary name).

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> b = f.define_dynamic_boundary(['p1','p2','p3'], name='tri')
        >>> mask = f.within_boundary('p1', b)
        >>> bool(isinstance(mask, pd.Series))
        True
        >>> mask2 = f.within_boundary('p1', 'tri')
        >>> bool(isinstance(mask2, pd.Series))
        True

        ```
        """
        if isinstance(boundary, (StaticAxis, DynamicAxis)):
            raise TypeError(
                "within_boundary does not accept axis assets. "
                "Use distance_to_axis() for axis-based features."
            )
        if isinstance(boundary, StaticBoundary):
            return self._within_boundary_static_impl(
                point,
                boundary_vertices=list(boundary.vertices),
                dims=boundary.dims,
                boundary_label=boundary.name or self._short_boundary_id(list(boundary.vertices)),
                boundary_meta=boundary.to_dict(),
            )
        if isinstance(boundary, DynamicBoundary):
            return self._within_boundary_dynamic_impl(
                point,
                boundary_points=list(boundary.points),
                dims=boundary.dims,
                scale_dim1=boundary.scale_dim1,
                scale_dim2=boundary.scale_dim2,
                anchor_points=list(boundary.anchor_points)
                if boundary.anchor_points is not None
                else None,
                boundary_label=boundary.name or self._short_boundary_id(list(boundary.points)),
                boundary_meta=boundary.to_dict(),
            )
        if isinstance(boundary, str):
            stored = self._resolve_boundary_ref(boundary)
            return self.within_boundary(point, stored)
        raise TypeError(
            "Unsupported boundary value. Expected StaticBoundary, DynamicBoundary, "
            "or stored boundary name."
        )

    def distance_to_boundary(
        self,
        point: str,
        boundary: str | DynamicBoundary | StaticBoundary,
        *,
        signed: bool = False,
    ) -> FeaturesResult:
        """
        Main boundary distance API.

        Accepts a ``StaticBoundary`` or ``DynamicBoundary`` (or a stored boundary name).

        Args:
            point: Keypoint name whose coordinates are measured.
            boundary: A ``StaticBoundary``, ``DynamicBoundary``, or the string name
                of a stored boundary.
            signed: If ``True``, distances are negated for points inside the boundary
                (negative = inside, positive = outside, zero = on boundary).

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> b = f.define_static_boundary(['p1','p2','p3'], name='tri')
        >>> d = f.distance_to_boundary('p1', b)
        >>> bool(isinstance(d, pd.Series))
        True
        >>> d2 = f.distance_to_boundary('p1', 'tri')
        >>> bool(isinstance(d2, pd.Series))
        True
        >>> ds = f.distance_to_boundary('p1', b, signed=True)
        >>> bool((ds <= 0).any() or (ds >= 0).any())
        True

        ```
        """
        if isinstance(boundary, (StaticAxis, DynamicAxis)):
            raise TypeError(
                "distance_to_boundary does not accept axis assets. "
                "Use distance_to_axis() for axis-based features."
            )
        if isinstance(boundary, StaticBoundary):
            return self._distance_to_boundary_static_impl(
                point,
                boundary_vertices=list(boundary.vertices),
                dims=boundary.dims,
                boundary_label=boundary.name or self._short_boundary_id(list(boundary.vertices)),
                boundary_meta=boundary.to_dict(),
                signed=signed,
            )
        if isinstance(boundary, DynamicBoundary):
            return self._distance_to_boundary_dynamic_impl(
                point,
                boundary_points=list(boundary.points),
                dims=boundary.dims,
                scale_dim1=boundary.scale_dim1,
                scale_dim2=boundary.scale_dim2,
                anchor_points=list(boundary.anchor_points)
                if boundary.anchor_points is not None
                else None,
                boundary_label=boundary.name or self._short_boundary_id(list(boundary.points)),
                boundary_meta=boundary.to_dict(),
                signed=signed,
            )
        if isinstance(boundary, str):
            stored = self._resolve_boundary_ref(boundary)
            return self.distance_to_boundary(point, stored, signed=signed)
        raise TypeError(
            "Unsupported boundary value. Expected StaticBoundary, DynamicBoundary, "
            "or stored boundary name."
        )

    def _distance_to_boundary_static_impl(
        self,
        point: str,
        *,
        boundary_vertices: list[tuple[float, float]],
        dims: tuple[str, str],
        boundary_label: str,
        boundary_meta,
        signed: bool = False,
    ) -> FeaturesResult:
        if len(boundary_vertices) < 3:
            raise Exception("boundary encloses no area")
        boundary_has_nan = any(pd.isna(bx) or pd.isna(by) for bx, by in boundary_vertices)
        name = f"distance_to_boundary_static_{point}_in_{boundary_label}"
        if signed:
            name += "_signed"
        meta = {
            "function": "distance_to_boundary",
            "point": point,
            "boundary": boundary_meta,
            "signed": signed,
        }

        df = self.tracking.data
        px = df[point + "." + dims[0]].to_numpy(dtype=float)
        py = df[point + "." + dims[1]].to_numpy(dtype=float)
        valid = ~(np.isnan(px) | np.isnan(py))

        if boundary_has_nan:
            result = pd.Series(np.nan, index=df.index)
        else:
            poly = Polygon(boundary_vertices)
            result = pd.Series(np.nan, index=df.index)
            if valid.any():
                pts = shapely.points(px[valid], py[valid])
                result[valid] = shapely.distance(poly.exterior, pts)
                if signed:
                    inside = shapely.within(pts, poly)
                    result[valid] *= np.where(inside, -1.0, 1.0)
        return FeaturesResult(result, self, name, meta)

    def _distance_to_boundary_dynamic_impl(
        self,
        point: str,
        *,
        boundary_points: list[str],
        dims: tuple[str, str],
        scale_dim1: float,
        scale_dim2: float,
        anchor_points: list[str] | None,
        boundary_label: str,
        boundary_meta,
        signed: bool = False,
    ) -> FeaturesResult:
        if len(boundary_points) < 3:
            raise Exception("boundary encloses no area")
        name = f"distance_to_boundary_dynamic_{point}_in_{boundary_label}"
        if signed:
            name += "_signed"
        meta = {
            "function": "distance_to_boundary",
            "point": point,
            "boundary": boundary_meta,
            "signed": signed,
        }

        df = self.tracking.data
        px = df[point + "." + dims[0]].to_numpy(dtype=float)
        py = df[point + "." + dims[1]].to_numpy(dtype=float)
        boundary_obj = DynamicBoundary(
            points=tuple(boundary_points),
            dims=dims,
            anchor_points=tuple(anchor_points) if anchor_points is not None else None,
            scale_dim1=scale_dim1,
            scale_dim2=scale_dim2,
        )
        verts = boundary_obj.to_numpy_per_frame(df)  # (n, N_verts, 2), scaling applied

        valid = ~(np.isnan(px) | np.isnan(py) | np.any(np.isnan(verts), axis=(1, 2)))
        result = pd.Series(np.nan, index=df.index)
        if valid.any():
            coords = verts[valid]
            polys = shapely.polygons(shapely.linearrings(coords))
            exteriors = shapely.get_exterior_ring(polys)
            pts = shapely.points(px[valid], py[valid])
            result[valid] = shapely.distance(exteriors, pts)
            if signed:
                inside = shapely.within(pts, polys)
                result[valid] *= np.where(inside, -1.0, 1.0)
        return FeaturesResult(result, self, name, meta)

    def distance_to_boundary_static(
        self, point: str, boundary, boundary_name: str = None
    ) -> FeaturesResult:
        raise NotImplementedError(
            "Features.distance_to_boundary_static() was removed; use "
            "Features.distance_to_boundary(point, boundary) with a StaticBoundary "
            "or stored boundary name instead."
        )

    def distance_to_boundary_dynamic(
        self, point: str, boundary, boundary_name: str | None = None
    ) -> FeaturesResult:
        raise NotImplementedError(
            "Features.distance_to_boundary_dynamic() was removed; use "
            "Features.distance_to_boundary(point, boundary) with a DynamicBoundary "
            "or stored boundary name instead."
        )

    def distance_to_axis(
        self,
        point: str,
        axis: str | StaticAxis | DynamicAxis,
        *,
        signed: bool = False,
    ) -> FeaturesResult:
        """Framewise perpendicular distance from a keypoint to an infinite axis.

        The axis is always treated as extending infinitely in both directions.
        Use :meth:`define_static_axis`, :meth:`define_dynamic_axis`, or
        :meth:`import_static_axis` to create axis assets.

        Args:
            point: Keypoint to measure from.
            axis: A two-point axis asset, or the name of a registered one.
            signed: If True, return a signed distance (2-D axes only). Positive means
                the point is to the right when facing from the first to the second axis
                reference point; negative means it is to the left. The sign convention
                matches :meth:`define_static_axis` ``offset``: an ``offset > 0`` shifts
                the axis rightward, so a point that was on the axis will have a negative
                signed distance from the offset-shifted one. Raises ``ValueError`` for
                non-2-D axes.

        Returns:
            Per-frame perpendicular distance series.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> ax = f.define_dynamic_axis('p1', 'p2')
        >>> res = f.distance_to_axis('p3', ax)
        >>> isinstance(res, pd.Series) and len(res) == len(t.data)
        True
        >>> res_signed = f.distance_to_axis('p3', ax, signed=True)
        >>> isinstance(res_signed, pd.Series) and len(res_signed) == len(t.data)
        True

        ```
        """
        resolved = self._resolve_axis_ref(axis)
        dims = resolved.dims

        if signed and len(dims) != 2:
            raise ValueError(
                f"signed=True requires a 2-D axis; axis {resolved.name!r} has dims {dims}."
            )

        if "rescale_distance_method" not in self.tracking.meta:
            warnings.warn("distance has not been calibrated", stacklevel=2)
        if "smoothing" not in self.tracking.meta:
            warnings.warn("tracking data have not been smoothed", stacklevel=2)

        df = self.tracking.data
        n = len(df)
        P = self.tracking.get_point_data(point, dims=list(dims)).to_numpy(dtype=float)

        if isinstance(resolved, StaticAxis):
            A = np.tile(np.array(resolved.vertices[0], dtype=float), (n, 1))
            B = np.tile(np.array(resolved.vertices[1], dtype=float), (n, 1))
        else:
            arr = resolved.to_numpy_per_frame(df)  # (n, 2, d)
            A = arr[:, 0, :]
            B = arr[:, 1, :]

        dist = point_to_axis_distance(P, A, B, signed=signed)
        series = pd.Series(dist, index=df.index)

        if resolved.name:
            axis_label = resolved.name
        elif isinstance(resolved, StaticAxis) and resolved.source_points:
            axis_label = "_".join(resolved.source_points)
        elif isinstance(resolved, DynamicAxis):
            axis_label = "_".join(resolved.points)
        else:
            axis_label = "axis"

        sign_suffix = "_signed" if signed else ""
        name_str = f"distance_to_axis_{point}_from_{axis_label}_in_{''.join(dims)}{sign_suffix}"
        meta = {
            "function": "distance_to_axis",
            "point": point,
            "axis": resolved.to_dict(),
            "signed": signed,
        }
        return FeaturesResult(series, self, name_str, meta)

    def axis_intersects_boundary(
        self,
        axis: str | StaticAxis | DynamicAxis,
        boundary: str | StaticBoundary | DynamicBoundary,
        *,
        dims: tuple[str, str] = ("x", "y"),
        zones: Literal["front", "within", "behind"]
        | set[Literal["front", "within", "behind"]]
        | None = None,
    ) -> FeaturesResult:
        """Per-frame boolean: does the axis line cross the boundary polygon?

        The axis is always treated as infinite.  Each intersection point is
        classified by its scalar projection *t* onto the A→B segment:

        - ``"behind"``: t ≤ 0  (at or before the first reference point)
        - ``"within"``: 0 < t < 1  (strictly between the two reference points)
        - ``"front"``:  t ≥ 1  (at or beyond the second reference point)

        A frame is ``True`` when at least one intersection with the boundary
        falls inside any of the requested ``zones``.  Frames where the axis is
        degenerate (A == B) or any coordinate is NaN are returned as ``pd.NA``.

        Args:
            axis: Two-point axis asset, or the name of a registered one.
            boundary: Polygon boundary asset, or the name of a registered one.
            dims: The 2-D coordinate space for the intersection test. Both the axis
                and the boundary must have exactly these dims.
            zones: Which zones count as an intersection. A single zone name (e.g.
                ``"front"``) or a set of names. Defaults to all three zones (any
                intersection anywhere along the infinite axis).

        Returns:
            Boolean series (pandas nullable boolean dtype).

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> ax = f.define_dynamic_axis('p1', 'p2')
        >>> b = f.define_dynamic_boundary(['p1', 'p2', 'p3'], name='tri')
        >>> res = f.axis_intersects_boundary(ax, b)
        >>> bool(isinstance(res, pd.Series))
        True

        ```
        """
        _VALID_ZONES: set[str] = {"front", "within", "behind"}
        if isinstance(zones, str):
            zones = {zones}
        if zones is None:
            zones = _VALID_ZONES
        else:
            unknown = zones - _VALID_ZONES
            if unknown:
                raise ValueError(
                    f"Unknown zone(s): {unknown!r}.  Valid zones are {_VALID_ZONES!r}."
                )
            if not zones:
                raise ValueError("zones must not be empty.")

        resolved_axis = self._resolve_axis_ref(axis)
        resolved_boundary = self._resolve_boundary_ref(boundary)

        if tuple(resolved_axis.dims) != tuple(dims):
            raise ValueError(f"axis dims {resolved_axis.dims!r} do not match dims={dims!r}.")
        if tuple(resolved_boundary.dims) != tuple(dims):
            raise ValueError(
                f"boundary dims {resolved_boundary.dims!r} do not match dims={dims!r}."
            )

        df = self.tracking.data
        for d in dims:
            if not any(col.endswith(f".{d}") for col in df.columns):
                raise ValueError(f"Dimension {d!r} not found in tracking data columns.")
        n = len(df)

        # Build A and B per frame.
        if isinstance(resolved_axis, StaticAxis):
            A = np.tile(np.array(resolved_axis.vertices[0], dtype=float), (n, 1))
            B = np.tile(np.array(resolved_axis.vertices[1], dtype=float), (n, 1))
        else:
            arr = resolved_axis.to_numpy_per_frame(df)  # (n, 2, 2)
            A = arr[:, 0, :]
            B = arr[:, 1, :]

        # Build boundary vertices per frame (scaling applied inside to_numpy_per_frame).
        if isinstance(resolved_boundary, StaticBoundary):
            verts = np.array(resolved_boundary.vertices, dtype=float)  # (N_verts, 2)
            boundary_valid = ~np.any(np.isnan(verts))  # scalar; all-or-nothing for static
            boundary_valid = np.full(n, boundary_valid)
        else:
            verts = resolved_boundary.to_numpy_per_frame(df)  # (n, N_verts, 2)
            boundary_valid = ~np.any(np.isnan(verts), axis=(1, 2))

        axis_nan = np.any(np.isnan(A) | np.isnan(B), axis=1)
        axis_degenerate = np.linalg.norm(B - A, axis=1) == 0.0
        valid = ~axis_nan & ~axis_degenerate & boundary_valid

        result = pd.Series(pd.array([pd.NA] * n, dtype="boolean"), index=df.index)
        if valid.any():
            v = verts if verts.ndim == 2 else verts[valid]
            result[valid] = _axis_boundary_intersections(A[valid], B[valid], v, zones)

        if resolved_axis.name:
            axis_label = resolved_axis.name
        elif isinstance(resolved_axis, StaticAxis) and resolved_axis.source_points:
            axis_label = "_".join(resolved_axis.source_points)
        elif isinstance(resolved_axis, DynamicAxis):
            axis_label = "_".join(resolved_axis.points)
        else:
            axis_label = "axis"

        boundary_label = resolved_boundary.name or self._short_boundary_id(
            list(
                resolved_boundary.vertices
                if isinstance(resolved_boundary, StaticBoundary)
                else resolved_boundary.points
            )
        )

        zones_str = "_".join(sorted(zones))
        name_str = f"axis_intersects_boundary_{axis_label}_{boundary_label}_{zones_str}"
        meta = {
            "function": "axis_intersects_boundary",
            "axis": resolved_axis.to_dict(),
            "boundary": resolved_boundary.to_dict(),
            "dims": list(dims),
            "zones": sorted(zones),
        }
        return FeaturesResult(result, self, name_str, meta)

    def area_of_boundary(
        self, boundary: str | StaticBoundary | DynamicBoundary, **kwargs
    ) -> FeaturesResult:
        """
        Return boundary area as a FeaturesResult.

        Accepts a ``StaticBoundary`` or ``DynamicBoundary`` (or a stored boundary name).

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> b = f.define_dynamic_boundary(['p1','p2','p3'], name='tri')
        >>> a = f.area_of_boundary(b)
        >>> bool(isinstance(a, pd.Series))
        True
        >>> a2 = f.area_of_boundary('tri')
        >>> bool(isinstance(a2, pd.Series))
        True

        ```
        """
        if "median" in kwargs or "boundary_name" in kwargs:
            raise TypeError(
                "area_of_boundary accepts only `boundary`. "
                "Use area_of_boundary(boundary) with a StaticBoundary, "
                "DynamicBoundary, or stored boundary name."
            )
        if len(kwargs) > 0:
            keys = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword argument(s): {keys}")

        if isinstance(boundary, str):
            stored = self._resolve_boundary_ref(boundary)
            return self.area_of_boundary(stored)

        if isinstance(boundary, StaticBoundary):
            boundary_label = boundary.name or self._short_boundary_id(list(boundary.vertices))
            name = f"area_of_boundary_{boundary_label}_static"
            meta = {"function": "area_of_boundary", "boundary": boundary.to_dict()}
            local_poly = Polygon(boundary.vertices)
            area = local_poly.area
            result = pd.Series(area, index=self.tracking.data.index)
            return FeaturesResult(result, self, name, meta)

        if isinstance(boundary, DynamicBoundary):
            boundary_label = boundary.name or self._short_boundary_id(list(boundary.points))
            name = f"area_of_boundary_{boundary_label}_dynamic"
            meta = {"function": "area_of_boundary", "boundary": boundary.to_dict()}
            df = self.tracking.data
            verts = boundary.to_numpy_per_frame(df)  # (n, N_verts, 2), scaling applied
            bx = verts[:, :, 0]
            by = verts[:, :, 1]
            bx_next = np.roll(bx, -1, axis=1)
            by_next = np.roll(by, -1, axis=1)
            result = pd.Series(
                0.5 * np.abs(np.sum(bx * by_next - bx_next * by, axis=1)),
                index=df.index,
            )
            return FeaturesResult(result, self, name, meta)

        if self._is_legacy_point_name_list(boundary):
            raise TypeError(
                "area_of_boundary no longer accepts point-name lists. "
                "Create a boundary with define_static_boundary()/define_dynamic_boundary() "
                "and pass that boundary (or its stored name)."
            )

        raise TypeError(
            "Unsupported boundary value. Expected boundary name, "
            "StaticBoundary, or DynamicBoundary."
        )

    def area_of_boundary_deprecated(
        self, boundary: list[str], median: bool = True
    ) -> FeaturesResult:
        """
        Deprecated legacy area API.

        Args:
            boundary: Ordered point names defining the polygon.
            median: ``True`` for static-median area, ``False`` for per-frame dynamic area.
        """
        raise NotImplementedError(
            "Features.area_of_boundary_deprecated() was removed; use "
            "Features.area_of_boundary(boundary) instead."
        )

    def acceleration(self, point: str, dims=("x", "y")) -> FeaturesResult:
        """
        Returns acceleration of point from previous frame to current frame, for each frame.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> acc = f.acceleration('p1')
        >>> isinstance(acc, pd.Series) and len(acc) == len(t.data)
        True

        ```
        """
        if "smoothing" not in self.tracking.meta.keys():
            warnings.warn("tracking data have not been smoothed", stacklevel=2)
        _speed = self.speed(point, dims=dims)
        _acceleration = _speed.diff() * self.tracking.meta["fps"]
        name = f"acceleration_of_{point}_in_{''.join(dims)}"
        meta = {"function": "acceleration", "point": point, "dims": dims}
        return FeaturesResult(_acceleration, self, name, meta)

    def azimuth(self, point1: str, point2: str) -> FeaturesResult:
        """
        Returns azimuth in radians from tracked point1 to tracked point2
        for each frame in the data, relative to the direction of the x-axis.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> ang = f.azimuth('p1','p2')
        >>> isinstance(ang, pd.Series) and len(ang) == len(t.data)
        True

        ```
        """
        if "smoothing" not in self.tracking.meta.keys():
            warnings.warn("tracking data have not been smoothed", stacklevel=2)

        _1x = self.tracking.data[point1 + ".x"]
        _1y = self.tracking.data[point1 + ".y"]
        _2x = self.tracking.data[point2 + ".x"]
        _2y = self.tracking.data[point2 + ".y"]

        result = np.arctan2((_2y - _1y), (_2x - _1x))
        name = f"azimuth_from_{point1}_to_{point2}"
        meta = {"function": "azimuth", "point1": point1, "point2": point2}
        return FeaturesResult(result, self, name, meta)

    def azimuth_deviation(
        self, basepoint: str, pointdirection1: str, pointdirection2: str
    ) -> FeaturesResult:
        """
        Compute the signed angular deviation (radians) between two directions
        from a common basepoint for each frame.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> dev = f.azimuth_deviation('p1','p2','p3')
        >>> bool((isinstance(dev, pd.Series) and len(dev) == len(t.data)))
        True

        ```
        """
        a1 = self.azimuth(basepoint, pointdirection1)
        a2 = self.azimuth(basepoint, pointdirection2)
        deviation = (a1 - a2 + np.pi) % (2 * np.pi) - np.pi
        name = f"azimuth_deviation_{basepoint}_to_{pointdirection1}_and_{pointdirection2}"
        meta = {
            "function": "azimuth_deviation",
            "basepoint": basepoint,
            "pointdirection1": pointdirection1,
            "pointdirection2": pointdirection2,
        }
        return FeaturesResult(deviation, self, name, meta)

    def within_azimuth_deviation(
        self,
        basepoint: str,
        pointdirection1: str,
        pointdirection2: str,
        deviation: float,
    ) -> FeaturesResult:
        """
        Return True for frames where the angular deviation between two rays
        from basepoint is <= deviation (radians).
        NA is propagated where inputs are missing (pd.NA).

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> mask = f.within_azimuth_deviation('p1','p2','p3', deviation=1.0)
        >>> bool((isinstance(mask, pd.Series) and mask.notna().any()))
        True

        ```
        """
        obs_deviation = self.azimuth_deviation(basepoint, pointdirection1, pointdirection2)
        # Propagate NA: comparisons with missing deviations should yield pd.NA
        mask = obs_deviation.notna()
        result = pd.Series(pd.NA, index=obs_deviation.index, dtype="boolean")
        result[mask] = (obs_deviation[mask] <= deviation).astype("boolean")
        name = (
            f"within_azimuth_deviation_{basepoint}_to_{pointdirection1}_"
            f"and_{pointdirection2}_leq_{deviation}"
        )
        meta = {
            "function": "within_angle_deviation",
            "basepoint": basepoint,
            "pointdirection1": pointdirection1,
            "pointdirection2": pointdirection2,
            "deviation": deviation,
        }
        return FeaturesResult(result, self, name, meta)

    def speed(self, point: str, dims=("x", "y")) -> FeaturesResult:
        """
        Returns average speed of point from previous frame to current frame, for each frame.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> sp = f.speed('p1')
        >>> isinstance(sp, pd.Series) and len(sp) == len(t.data)
        True

        ```
        """
        if "rescale_distance_method" not in self.tracking.meta.keys():
            warnings.warn("distance has not been calibrated", stacklevel=2)
        if "smoothing" not in self.tracking.meta.keys():
            warnings.warn("tracking data have not been smoothed", stacklevel=2)

        result = self.distance_change(point, dims=dims) * self.tracking.meta["fps"]
        name = f"speed_of_{point}_in_{''.join(dims)}"
        meta = {"function": "speed", "point": point, "dims": dims}
        return FeaturesResult(result, self, name, meta)

    def above_speed(self, point: str, speed: float, dims=("x", "y")) -> FeaturesResult:
        """
        Return True for frames where the point's speed is >= threshold.
        NA is propagated where inputs are missing (pd.NA).

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> m = f.above_speed('p1', speed=0.0)
        >>> isinstance(m, pd.Series) and len(m) == len(t.data)
        True

        ```
        """
        obs_speed = self.speed(point, dims=dims)
        mask = obs_speed.notna()
        result = pd.Series(pd.NA, index=obs_speed.index, dtype="boolean")
        result[mask] = (obs_speed[mask] >= speed).astype("boolean")
        name = f"above_speed_{point}_geq_{speed}_in_{''.join(dims)}"
        meta = {"function": "above_speed", "point": point, "speed": speed, "dims": dims}
        return FeaturesResult(result, self, name, meta)

    def all_above_speed(self, points: list, speed: float, dims=("x", "y")) -> FeaturesResult:
        """
        Return True for frames where all listed points are moving at least at the threshold speed.
        NA is propagated: if any input is NA at a frame, result is NA.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> m = f.all_above_speed(['p1','p2'], speed=0.0)
        >>> isinstance(m, pd.Series) and len(m) == len(t.data)
        True

        ```
        """
        df = pd.DataFrame([self.above_speed(point, speed, dims=dims) for point in points]).astype(
            "boolean"
        )
        # Manual NA-propagating "all" across points per frame to avoid ambiguous NA reductions
        has_false = (~df.fillna(True)).any(axis=0)
        has_na = df.isna().any(axis=0)
        result = pd.Series(pd.NA, index=df.columns, dtype="boolean")
        result[has_false] = False
        result[~has_false & ~has_na] = True
        points_str = "_".join(str(p) for p in points)
        name = f"all_above_speed_{points_str}_geq_{speed}_in_{''.join(dims)}"
        meta = {
            "function": "all_above_speed",
            "points": points,
            "speed": speed,
            "dims": dims,
        }
        return FeaturesResult(result, self, name, meta)

    def below_speed(self, point: str, speed: float, dims=("x", "y")) -> FeaturesResult:
        """
        Return True for frames where the point's speed is < threshold.
        NA is propagated where inputs are missing (pd.NA).

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> m = f.below_speed('p1', speed=9999.0)
        >>> isinstance(m, pd.Series) and len(m) == len(t.data)
        True

        ```
        """
        obs_speed = self.speed(point, dims=dims)
        mask = obs_speed.notna()
        result = pd.Series(pd.NA, index=obs_speed.index, dtype="boolean")
        result[mask] = (obs_speed[mask] < speed).astype("boolean")
        name = f"below_speed_{point}_lt_{speed}_in_{''.join(dims)}"
        meta = {"function": "below_speed", "point": point, "speed": speed, "dims": dims}
        return FeaturesResult(result, self, name, meta)

    def all_below_speed(self, points: list, speed: float, dims=("x", "y")) -> FeaturesResult:
        """
        Return True for frames where all listed points are moving slower than the threshold speed.
        NA is propagated: if any input is NA at a frame, result is NA.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> m = f.all_below_speed(['p1','p2'], speed=9999.0)
        >>> isinstance(m, pd.Series) and len(m) == len(t.data)
        True

        ```
        """
        df = pd.DataFrame([self.below_speed(point, speed, dims=dims) for point in points]).astype(
            "boolean"
        )
        # Manual NA-propagating "all" across points per frame to avoid ambiguous NA reductions
        has_false = (~df.fillna(True)).any(axis=0)
        has_na = df.isna().any(axis=0)
        result = pd.Series(pd.NA, index=df.columns, dtype="boolean")
        result[has_false] = False
        result[~has_false & ~has_na] = True
        points_str = "_".join(str(p) for p in points)
        name = f"all_below_speed_{points_str}_lt_{speed}_in_{''.join(dims)}"
        meta = {
            "function": "all_below_speed",
            "points": points,
            "speed": speed,
            "dims": dims,
        }
        return FeaturesResult(result, self, name, meta)

    def distance_change(self, point: str, dims=("x", "y")) -> FeaturesResult:
        """
        Return unsigned distance moved by point from previous to current frame, per frame.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> dc = f.distance_change('p1')
        >>> isinstance(dc, pd.Series) and len(dc) == len(t.data)
        True

        ```
        """
        if "rescale_distance_method" not in self.tracking.meta.keys():
            warnings.warn("distance has not been calibrated", stacklevel=2)
        if "smoothing" not in self.tracking.meta.keys():
            warnings.warn("tracking data have not been smoothed", stacklevel=2)

        result = np.sqrt(sum([(self.tracking.data[point + "." + dim].diff()) ** 2 for dim in dims]))
        name = f"distance_change_{point}_in_{''.join(dims)}"
        meta = {"function": "distance_change", "point": point, "dims": dims}
        return FeaturesResult(result, self, name, meta)

    def compose_state_from_booleans(
        self,
        sources: dict[str, str | pd.Series],
        *,
        priority: list[str] | None = None,
        none_label: str = "none",
    ) -> FeaturesResult:
        """
        Compose a categorical state series from labeled boolean sources.

        Args:
            sources: Mapping ``{state_label: source}``, where source is either a column
                name in ``self.data`` containing a boolean series, or a boolean pandas
                Series aligned/reindexable to ``self.data.index`` (e.g. a ``FeaturesResult``).
            priority: Optional label precedence when multiple sources are True in the same
                frame. Labels not listed are appended in insertion order.
            none_label: Label used when no source is True at a frame.

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> idx = t.data.index
        >>> f.store(pd.Series([True, False, True, False, True], index=idx).reindex(idx,
        ...         fill_value=False),
        ...         'in_corner', meta={})
        >>> f.store(pd.Series([False, True, True, False, True], index=idx).reindex(idx,
        ...         fill_value=False),
        ...         'in_food', meta={})
        >>> state = f.compose_state_from_booleans(
        ...     {"corner": "in_corner", "food": "in_food"},
        ...     priority=["food", "corner"],
        ... )
        >>> isinstance(state, pd.Series)
        True
        >>> set(state.dropna().unique()) >= {'corner', 'food', 'none'}
        True

        ```
        """
        if not isinstance(sources, dict) or len(sources) == 0:
            raise ValueError("sources must be a non-empty dict")
        if none_label in sources.keys():
            raise ValueError(f"none_label, '{none_label}', found in sources.keys()")

        resolved: dict[str, pd.Series] = {}
        source_spec = {}
        for label, source in sources.items():
            if isinstance(source, str):
                if source not in self.data.columns:
                    raise ValueError(f"Column '{source}' not found in features.data")
                resolved[label] = self.data[source]
                source_spec[label] = {"type": "column", "name": source}
            elif isinstance(source, pd.Series):
                resolved[label] = source
                source_spec[label] = {"type": "series"}
            else:
                raise TypeError(
                    f"Source '{label}' must be a column name (str) or pandas Series, "
                    f"got {type(source).__name__}."
                )

        state = compose_state_from_boolean_sources(
            resolved,
            index=self.tracking.data.index,
            priority=priority,
            none_label=none_label,
        )
        name = "state_from_booleans"
        meta = {
            "function": "compose_state_from_booleans",
            "labels": list(sources.keys()),
            "priority": priority,
            "none_label": none_label,
            "sources": source_spec,
        }
        return FeaturesResult(state, self, name, meta)

    def store(
        self,
        feature: pd.Series,
        name: str,
        overwrite: bool = False,
        meta: dict | None = None,
    ) -> None:
        """
        Store calculated feature with name and associated freeform metadata.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> s = pd.Series(range(len(t.data)), index=t.data.index)
        >>> f.store(s, 'counter', meta={'unit':'frames'})
        >>> 'counter' in f.data.columns and f.meta['counter']['unit'] == 'frames'
        True

        ```
        """
        if meta is None:
            meta = {}
        if name in self.data.columns:
            if overwrite:
                self.data[name] = feature
                warnings.warn("feature '" + name + "' overwritten", stacklevel=2)
            else:
                raise Exception(
                    "feature with name '"
                    + name
                    + "' already stored. set overwrite=True to overwrite"
                )
        else:
            self.data[name] = feature

        self.meta[name] = meta

    def classify(self, classifier: BaseClassifier, **kwargs):
        """
        Classify behaviour using a classifier with inputs from this Features object.
        Returns a FeaturesResult. Classifier output must be a pd.Series with same index.
        """
        result = classifier.predict(self, **kwargs)
        name = f"classified_{classifier.__class__.__name__}"
        meta = {"function": "classify", "classifier": classifier.__class__.__name__}
        return FeaturesResult(result, self, name, meta)

    def smooth(
        self,
        name: str,
        method: str,
        window: int,
        inplace: bool = False,
        **method_kwargs: Any,
    ) -> pd.Series:
        """
        Smooth a stored feature by name over a rolling window.

        Args:
            name: Name of the feature column in ``self.data`` to smooth.
            method: Smoothing method. One of:

                * ``'median'`` — rolling median (numerical).
                * ``'mean'`` — rolling mean (numerical).
                * ``'savgol'`` — Savitzky–Golay filter (SciPy). Extra kwargs e.g.
                  ``polyorder=3``, ``mode='interp'``.
                * ``'mode'`` — rolling mode (numerical or categorical).
                * ``'block'`` — applies ``block_filter`` then ``block_fill`` using
                  ``window`` for both ``min_block`` and ``max_gap``. Legacy
                  ``smooth_block`` behavior is available via
                  ``series_utils.smooth_block`` directly.

            window: Rolling window size.
            inplace: If True, overwrite the stored feature and update its metadata.
            **method_kwargs: Extra keyword arguments forwarded to the smoothing method.

        Returns:
            Smoothed series.
        """
        if "smoothing" in self.meta[name].keys():
            raise Exception("feature already smoothed")

        # Numeric, DRY path using common smoother
        if method in {"median", "mean", "savgol"}:
            smoothed = smooth_series(self.data[name], method=method, window=window, **method_kwargs)
            if inplace:
                self.data[name] = smoothed.copy()
        elif method == "mode":
            smoothed = series_utils.rolling_apply(self.data[name], window, series_utils.mode)
            if inplace:
                self.data[name] = smoothed.copy()
        elif method == "block":
            warnings.warn(
                "Legacy block behavior in Features.smooth(method='block') was removed. "
                "This now applies series_utils.block_filter followed by series_utils.block_fill "
                "using window for both min_block and max_gap. "
                "Deprecated legacy behavior remains available via series_utils.smooth_block.",
                stacklevel=2,
            )
            filtered = series_utils.block_filter(self.data[name], min_block=window)
            smoothed = series_utils.block_fill(
                filtered,
                max_gap=window,
                direction=method_kwargs.get("fill_direction", "both"),
                require_same_label=method_kwargs.get("fill_require_same_label", True),
            )
            if inplace:
                self.data[name] = smoothed.copy()
        else:
            raise Exception("method " + method + " not recognised")

        if not inplace:
            logger.info("inplace=False, feature " + name + " not overwritten")

        if inplace:
            logger.info("inplace=True, feature " + name + " overwritten")
            newmeta = dict()
            newmeta["base"] = self.meta[name]
            newmeta["smoothing"] = {
                "method": method,
                "window": window,
            }
            # Record any extra parameters used (e.g. polyorder/mode for savgol)
            for k, v in method_kwargs.items():
                newmeta["smoothing"][k] = v
            self.meta[name] = newmeta

        return smoothed

    def embedding_df(self, embedding: dict[str, list[int]]) -> pd.DataFrame:
        """
        Generate a time-series embedding DataFrame with per-column time shifts.

        Args:
            embedding: Mapping of feature column name to a list of integer time shifts.
                Positive shift pulls the value from the future (t+n); negative shift
                pulls from the past (t-n); zero is the current frame.

        Returns:
            One column per (feature, shift) pair, named ``<col>_t0``,
                ``<col>_t+n``, or ``<col>_t-n``.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> import pandas as pd
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> # prepare a simple feature to embed
        >>> s = pd.Series(range(len(t.data)), index=t.data.index)
        >>> f.store(s, 'counter', meta={})
        >>> emb = f.embedding_df({'counter':[0,1,-1]})
        >>> list(emb.columns)
        ['counter_t0', 'counter_t+1', 'counter_t-1']

        ```
        """
        missing = [col for col in embedding if col not in self.data.columns]
        if len(missing) > 0:
            raise ValueError(f"The following columns are not present in self.data: {missing}")
        data = {}
        for col, shifts in embedding.items():
            base_series = self.data[col]
            for shift in shifts:
                shifted = base_series.shift(
                    -shift
                )  # Reverse the sign: positive shift looks forward
                suffix = f"t{shift:+d}" if shift != 0 else "t0"
                data[f"{col}_{suffix}"] = shifted
        embed_df = pd.DataFrame(data, index=self.data.index)
        return embed_df

    def cluster_embedding(self, *args, **kwargs):
        """Removed in py3r.behaviour 3.3.0. Use :meth:`cluster_embedding_stream` instead."""
        raise NotImplementedError(
            "cluster_embedding() was removed in py3r.behaviour 3.3.0.  "
            "Use cluster_embedding_stream() instead.\n"
            "Note: cluster_embedding_stream uses MiniBatchKMeans (stochastic updates) "
            "rather than the full-batch KMeans of the old method — results will not be "
            "bit-for-bit identical.  For well-separated data the partition will match; "
            "increase n_epochs and batch_size to improve convergence for harder cases.  "
            "To reproduce results from py3r ≤ 3.2.1 exactly, pin to that version."
        )

    def cluster_embedding_stream(
        self,
        embedding_dict: dict[str, list[int]],
        n_clusters: int,
        random_state: int = 0,
        *,
        normalize: bool = False,
        normalize_details: dict[str, Literal["individual", "global", "none"]] | None = None,
        feature_weights: dict[str, float] | None = None,
        missing_policy: Literal["drop", "impute_weight"] = "drop",
        chunk_size: int = 10_000,
        n_epochs: int = 3,
        batch_size: int = 1024,
    ) -> tuple[FeaturesResult, CentroidsDf]:
        """
        Memory-friendly clustering on a single Features object.

        Delegates to ``FeaturesCollection.cluster_embedding_stream``.
        See that method for full parameter documentation.

        Returns:
            Tuple of ``(FeaturesResult, CentroidsDf)``.

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> f.store(pd.Series(range(len(t.data)), index=t.data.index), 'counter')
        >>> result, centroids = f.cluster_embedding_stream(
        ...     {'counter': [0]}, n_clusters=2)
        >>> hasattr(centroids, 'columns')
        True
        >>> len(result) == len(f.data)
        True

        ```
        """
        from py3r.behaviour.features.features_collection import FeaturesCollection

        fc = FeaturesCollection.from_list([self])
        batch, centroids = fc.cluster_embedding_stream(
            embedding_dict,
            n_clusters,
            random_state,
            normalize=normalize,
            normalize_details=normalize_details,
            feature_weights=feature_weights,
            missing_policy=missing_policy,
            chunk_size=chunk_size,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )
        return batch[self.handle], centroids

    def assign_clusters_by_centroids(
        self,
        centroids_df: CentroidsDf | pd.DataFrame,
        embedding: dict[str, list[int]] | None = None,
        *,
        allow_missing_features: Literal["self", "centroids", "both"] | None = None,
        scaling_factors: dict[str, float] | None = None,
        impute_means: pd.Series | None = None,
        # Removed legacy params; retained for explicit migration errors.
        rescale_factors: dict | None = None,
        custom_scaling: dict[str, dict] | None = None,
    ) -> FeaturesResult:
        """
        Assign cluster labels to this Features object using pre-fitted centroids.

        Args:
            centroids_df: Cluster centres. Passing a
                :class:`~py3r.behaviour.features.centroids_df.CentroidsDf` (the
                object returned by ``cluster_embedding*``) is preferred: the method
                will automatically apply the stored ``scaling_recipe``, including any
                per-recording individual normalisation, and infer the *embedding*
                from the recipe so it need not be passed separately.

                If a plain ``pd.DataFrame`` is passed, *embedding* and optionally
                *scaling_factors* must be provided (legacy path).
            embedding: The embedding dict used during fitting. Required when *centroids_df*
                is a plain ``pd.DataFrame``; inferred from the recipe when
                *centroids_df* is a :class:`CentroidsDf`.
            allow_missing_features: Controls whether cluster assignment is permitted when
                the full embedding space is not available, by projecting into a shared
                subspace of the columns that *both* sides can provide.

                * ``"self"`` – tolerate base features missing from *this* object
                  (e.g. a missing animal in a multi-animal recording). *centroids_df*
                  is expected to cover the full training embedding; only the columns
                  ``self`` can actually produce are used.
                * ``"centroids"`` – tolerate the centroids having fewer columns
                  than the full embedding ``self`` would generate (e.g. centroids
                  fitted on a reduced feature set). *self* must still carry all
                  requested base features; only the centroid columns are used.
                * ``"both"`` – tolerate gaps on either side; the strict intersection
                  of what ``self`` can produce and what the centroids contain is used.

                In all three cases a :class:`UserWarning` is issued that identifies
                which columns were dropped and from which side, so the caller can
                verify the subspace is sensible. A :exc:`ValueError` is raised when
                no columns remain after intersection regardless of the chosen mode.

                ``None`` (default) raises if the column sets do not match exactly.
            scaling_factors: Per-embedding-column constant multipliers. Applied only
                when *centroids_df* is a plain DataFrame (legacy path).
            impute_means: Per-column fill values (training-set column means) for NaN
                imputation. When *centroids_df* is a :class:`CentroidsDf` this is
                read automatically from the ``scaling_recipe``; pass explicitly only
                to override.
            rescale_factors: Removed. Raises ``NotImplementedError``; pass
                ``scaling_factors`` instead.
            custom_scaling: Removed. Raises ``NotImplementedError``; use
                ``build_column_weights()`` and pass ``scaling_factors`` instead.

        Returns:
            Series of cluster IDs (0 .. n_clusters-1).

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> # add a simple feature to embed
        >>> f.store(pd.Series(range(len(t.data)), index=t.data.index), 'counter', meta={})
        >>> emb = {'counter':[0, 1]}
        >>> df = f.embedding_df(emb)
        >>> # make 2 simple centroids matching columns
        >>> cents = pd.DataFrame([[0, 0], [1, 1]], columns=df.columns)
        >>> labels = f.assign_clusters_by_centroids(cents, emb)
        >>> isinstance(labels, pd.Series) and len(labels) == len(t.data)
        True

        ```
        """
        import warnings

        from sklearn.metrics.pairwise import pairwise_distances_argmin

        from py3r.behaviour.features.centroids_df import CentroidsDf

        if rescale_factors is not None:
            raise NotImplementedError("rescale_factors was removed; pass scaling_factors instead.")
        if custom_scaling is not None:
            raise NotImplementedError(
                "custom_scaling was removed; use build_column_weights() "
                "and pass scaling_factors instead."
            )

        # Detect legacy argument order: assign_clusters_by_centroids(embedding, centroids_df).
        # A dict can only be the embedding; a DataFrame/CentroidsDf can only be centroids.
        if isinstance(centroids_df, dict):
            warnings.warn(
                "The argument order for assign_clusters_by_centroids has changed: "
                "pass centroids first, then (optionally) embedding. "
                "Old: feat.assign_clusters_by_centroids(embedding, centroids_df) — "
                "New: feat.assign_clusters_by_centroids(centroids_df, embedding)",
                DeprecationWarning,
                stacklevel=2,
            )
            centroids_df, embedding = embedding, centroids_df

        # Unwrap CentroidsDf and extract scaling recipe.
        scaling_recipe: dict | None = None
        underlying_df: pd.DataFrame
        if isinstance(centroids_df, CentroidsDf):
            scaling_recipe = centroids_df.scaling_recipe
            underlying_df = centroids_df.df
            recipe_embedding = scaling_recipe.get("embedding_dict")
            if embedding is not None and recipe_embedding is not None:
                if embedding != recipe_embedding:
                    raise ValueError(
                        "The provided embedding dict does not match the one stored in the "
                        "CentroidsDf scaling recipe. Pass centroids only (without embedding) "
                        "to use the recipe's embedding, or ensure the dicts match."
                    )
                warnings.warn(
                    "The embedding dict is already stored in the CentroidsDf scaling recipe "
                    "and will be used automatically; passing it explicitly is redundant.",
                    UserWarning,
                    stacklevel=2,
                )
            if embedding is None:
                embedding = recipe_embedding
        else:
            underlying_df = centroids_df

        if embedding is None:
            raise ValueError("embedding is required when centroids_df is a plain DataFrame")

        # When self is allowed to have missing base features, pre-filter the
        # embedding dict so that embedding_df() does not raise.
        if allow_missing_features in ("self", "both"):
            missing_bases = [k for k in embedding if k not in self.data.columns]
            if missing_bases:
                warnings.warn(
                    f"allow_missing_features={allow_missing_features!r}: the following base "
                    f"feature(s) are absent from self and their embedding columns will be "
                    f"excluded from the subspace assignment: {missing_bases}",
                    UserWarning,
                    stacklevel=2,
                )
                embedding = {k: v for k, v in embedding.items() if k not in missing_bases}

        embed_df = self.embedding_df(embedding)

        if scaling_recipe is not None:
            # Recipe path (authoritative): apply individual norm then constant factors.
            cols_expected = scaling_recipe.get("columns")
            if cols_expected is not None and list(embed_df.columns) != list(cols_expected):
                if allow_missing_features is None:
                    raise ValueError(
                        "Embedding columns do not match centroids scaling recipe columns"
                    )
                # With allow_missing_features the column sets will be reconciled below.
            embed_df = embed_df.copy()
            for base, do_individual in (
                scaling_recipe.get("normalize_individual_base") or {}
            ).items():
                if not do_individual:
                    continue
                if base not in self.data.columns:
                    if allow_missing_features in ("self", "both"):
                        # Already warned above when filtering the embedding; just skip.
                        continue
                    raise ValueError(f"Base feature '{base}' missing for individual normalization")
                vals = self.data[base].to_numpy(dtype=np.float64)
                finite = vals[np.isfinite(vals)]
                std = float(np.std(finite)) if finite.size > 0 else 1.0
                std = std if std > 0 else 1.0
                base_cols = [c for c in embed_df.columns if c.startswith(base + "_t")]
                if not base_cols:
                    raise ValueError(f"No embedding columns found for base feature '{base}'")
                embed_df.loc[:, base_cols] = embed_df[base_cols] / std
            constant = scaling_recipe.get("constant_factors") or {}
            if constant:
                # Only scale columns present in embed_df; extra recipe keys are ignored
                # (they would otherwise inject NaN columns via DataFrame * Series alignment).
                constant_aligned = {k: v for k, v in constant.items() if k in embed_df.columns}
                if constant_aligned:
                    embed_df = embed_df * pd.Series(constant_aligned)
            # Read impute_means from recipe unless the caller already provided one.
            # Backward-compat: old recipes used the key "impute_medians".
            recipe_impute = scaling_recipe.get("impute_means") or scaling_recipe.get(
                "impute_medians"
            )
            if impute_means is not None and recipe_impute is not None:
                warnings.warn(
                    "impute_means is already stored in the CentroidsDf scaling recipe "
                    "and would be used automatically; passing it explicitly overrides the "
                    "recipe values.",
                    UserWarning,
                    stacklevel=2,
                )
            elif impute_means is None and recipe_impute is not None:
                impute_means = pd.Series(recipe_impute)
            applied_meta: dict = {"scaling_recipe": scaling_recipe}
        elif scaling_factors is not None:
            # Legacy path: plain constant multipliers.
            embed_df = embed_df * pd.Series(scaling_factors)
            applied_meta = {"scaling_factors": scaling_factors}
        else:
            applied_meta = {}

        if allow_missing_features is not None:
            # Reconcile columns: work in the intersection of what self produced
            # and what the centroids contain, with side-specific diagnostics.
            self_cols = set(embed_df.columns)
            centroid_cols = set(underlying_df.columns)

            only_in_self = sorted(self_cols - centroid_cols)
            only_in_centroids = sorted(centroid_cols - self_cols)

            # only_in_self: self produced columns the centroids don't have
            #   → centroids are "missing" those → tolerated by "centroids" / "both"
            if only_in_self and allow_missing_features in ("centroids", "both"):
                warnings.warn(
                    f"allow_missing_features={allow_missing_features!r}: {len(only_in_self)} "
                    f"embedding column(s) produced by self have no counterpart in the centroids "
                    f"and will be dropped: {only_in_self}",
                    UserWarning,
                    stacklevel=2,
                )
            elif only_in_self:
                raise ValueError(
                    f"Columns present in the self embedding but absent from the centroids "
                    f"(pass allow_missing_features='centroids' or 'both' to allow this): "
                    f"{only_in_self}"
                )

            # only_in_centroids: centroids have columns self couldn't produce
            #   → self is "missing" those base features → tolerated by "self" / "both"
            if only_in_centroids and allow_missing_features in ("self", "both"):
                warnings.warn(
                    f"allow_missing_features={allow_missing_features!r}: {len(only_in_centroids)} "
                    f"centroid column(s) have no counterpart in the self embedding "
                    f"and will be dropped: {only_in_centroids}",
                    UserWarning,
                    stacklevel=2,
                )
            elif only_in_centroids:
                raise ValueError(
                    f"Columns present in the centroids but absent from the self embedding "
                    f"(pass allow_missing_features='self' or 'both' to allow this): "
                    f"{only_in_centroids}"
                )

            # Preserve embed_df column order for the shared subspace.
            shared_cols = [c for c in embed_df.columns if c in centroid_cols]
            if not shared_cols:
                raise ValueError(
                    "No columns remain in common between the self embedding and the centroids "
                    "after filtering. self produced: "
                    f"{sorted(self_cols)}, centroids have: {sorted(centroid_cols)}"
                )

            embed_df = embed_df[shared_cols]
            underlying_df = underlying_df[shared_cols]
            if impute_means is not None:
                impute_means = impute_means[impute_means.index.isin(shared_cols)]
        else:
            if not embed_df.columns.equals(underlying_df.columns):
                raise ValueError("Columns in embedding and centroids do not match")

        if impute_means is not None:
            embed_df, _ = impute_frame(embed_df, impute_means)
            mask = pd.Series(True, index=embed_df.index)
            embed_values = embed_df.values
        else:
            mask = embed_df.notna().all(axis=1)
            embed_values = embed_df[mask].values
        centroids_values = underlying_df.values

        labels = pd.Series(pd.NA, index=embed_df.index, dtype="Int64")
        if len(embed_values) > 0:
            labels[mask] = pairwise_distances_argmin(embed_values, centroids_values)

        name = f"kmeans_{len(underlying_df.index)}"
        meta = {
            "function": "assign_clusters_by_centroids",
            "embedding": embedding,
            "allow_missing_features": allow_missing_features,
            **applied_meta,
        }
        return FeaturesResult(labels, self, name, meta)

    @dev_mode
    def train_knn_regressor(
        self,
        *,
        source_embedding: dict[str, list[int]],
        target_embedding: dict[str, list[int]],
        n_neighbors: int = 5,
        normalize_source: bool = False,
        **kwargs,
    ):
        """
        Developer mode: not available in public release yet.

        Train a KNN regressor to predict target from source embedding on this object.
        If normalize_source is True, normalize source and return rescale factors.
        Returns (model, input_cols, target_cols[, rescale_factors]).
        """
        train_embed = self.embedding_df(source_embedding)
        target_embed = self.embedding_df(target_embedding)
        rescale_factors = None
        if normalize_source:
            train_embed, rescale_factors = normalize_df(train_embed)
        model, train_cols, target_cols = train_knn_from_embeddings(
            [train_embed], [target_embed], n_neighbors, **kwargs
        )
        if normalize_source:
            return model, train_cols, target_cols, rescale_factors
        else:
            return model, train_cols, target_cols

    @dev_mode
    def predict_knn(
        self,
        model: KNeighborsRegressor,
        source_embedding: dict[str, list[int]],
        target_embedding: dict[str, list[int]],
        rescale_factors: dict = None,
    ) -> pd.DataFrame:
        """
        Developer mode: not available in public release yet.

        Predict using a trained KNN regressor on this Features object.
        If rescale_factors is provided, normalize the source embedding before prediction.
        The prediction will match the shape and columns of self.embedding_df(target_embedding).
        """
        test_embed = self.embedding_df(source_embedding)
        if rescale_factors is not None:
            test_embed = apply_normalization_to_df(test_embed, rescale_factors)
        target_embed = self.embedding_df(target_embedding)
        preds = predict_knn_on_embedding(model, test_embed, target_embed.columns)
        # Ensure the output DataFrame has the same index and columns as target_embed
        preds = preds.reindex(index=target_embed.index, columns=target_embed.columns)
        return preds

    @dev_mode
    @staticmethod
    def rms_error_between_embeddings(
        ground_truth: pd.DataFrame, prediction: pd.DataFrame, rescale: dict | str = None
    ) -> pd.Series:
        """
        Developer mode: not available in public release yet.

        Compute RMS for each row between two embedding DataFrames.
        If rescale is a dict, normalize both with it before computing error.
        If rescale == 'auto', compute factors from ground_truth and apply to both.
        Returns Series indexed like inputs; NaN where either input has NaNs.
        """
        if not ground_truth.columns.equals(prediction.columns) or not ground_truth.index.equals(
            prediction.index
        ):
            raise ValueError("Input DataFrames must have the same columns and index")
        if rescale is not None:
            if rescale == "auto":
                ground_truth, rescale_factors = normalize_df(ground_truth)
                prediction = apply_normalization_to_df(prediction, rescale_factors)
            elif isinstance(rescale, dict):
                ground_truth = apply_normalization_to_df(ground_truth, rescale)
                prediction = apply_normalization_to_df(prediction, rescale)
            else:
                raise ValueError("rescale must be None, a dict, or 'auto'")
        diff = ground_truth - prediction
        # Compute RMS error for each row, ignoring rows with any NaNs
        rms = np.sqrt((diff**2).mean(axis=1))
        # Set to NaN if either input row has any NaNs
        mask = ground_truth.notna().all(axis=1) & prediction.notna().all(axis=1)
        rms[~mask] = np.nan
        return rms

    @property
    def loc(self):
        return _Indexer(self, self._loc)

    @property
    def iloc(self):
        return _Indexer(self, self._iloc)

    def _loc(self, idx):
        if isinstance(idx, tuple):
            row_idx, col_idx = idx
            new_tracking = self.tracking.loc[row_idx]
        else:
            new_tracking = self.tracking.loc[idx]
        new = self.__class__(new_tracking)
        new.data = self.data.loc[idx].copy()
        new.meta = copy.deepcopy(self.meta)
        new._assets = copy.deepcopy(self._assets)
        new.handle = self.handle
        new.tags = copy.deepcopy(self.tags)
        return new

    def _iloc(self, idx):
        if isinstance(idx, tuple):
            row_idx, col_idx = idx
            new_tracking = self.tracking.iloc[row_idx]
        else:
            new_tracking = self.tracking.iloc[idx]
        new = self.__class__(new_tracking)
        new.data = self.data.iloc[idx].copy()
        new.meta = copy.deepcopy(self.meta)
        new._assets = copy.deepcopy(self._assets)
        new.handle = self.handle
        new.tags = copy.deepcopy(self.tags)
        return new

    def __getitem__(self, idx):
        return self.loc[idx]

    def define_elliptical_boundary_from_params(
        self,
        centre: str | list[str],
        major_axis_length: float,
        minor_axis_length: float,
        angle_in_radians: float = 0.0,
        n_points: int = 100,
    ) -> list[tuple[float, float]]:
        """
        Generate a polygonal approximation of an ellipse as a list of (x, y) tuples,
        around `centre` using explicit parameters.
        `centre` can be a point name or list of point names (then centre = mean of medians).

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> poly = f.define_elliptical_boundary_from_params(
        ...     'p1', major_axis_length=10, minor_axis_length=6,
        ...     angle_in_radians=0.0, n_points=32)
        >>> isinstance(poly, list) and len(poly) == 32
        True

        ```
        """
        from py3r.behaviour.util.ellipse_utils import ellipse_points

        if isinstance(centre, str):
            cx, cy = self.get_point_median(centre)
        elif isinstance(centre, list):
            centrepointmedians = [self.get_point_median(point) for point in centre]
            xcoords = np.array([point[0] for point in centrepointmedians])
            ycoords = np.array([point[1] for point in centrepointmedians])
            cx, cy = (xcoords.mean(), ycoords.mean())
        return ellipse_points(
            cx,
            cy,
            major_axis_length / 2,
            minor_axis_length / 2,
            angle_in_radians,
            n_points,
        )

    def define_elliptical_boundary_from_points(
        self,
        points: list[str],
        n_points: int = 100,
        scaling: float = 1.0,
        smallness_weight: float = 0.1,
    ) -> list[tuple[float, float]]:
        """
        Fit an ellipse to the median coordinates of the given tracked points (at least 4)
        and return a polygonal approximation. After fitting, the ellipse is scaled by `scaling`.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> # Use exactly 4 points to avoid requiring skimage in tests
        >>> poly = f.define_elliptical_boundary_from_points(
        ...     ['p1','p3','p2','p3'], n_points=20, scaling=1.0)
        >>> isinstance(poly, list) and len(poly) == 20
        True

        ```
        """
        import numpy as np

        from py3r.behaviour.util.ellipse_utils import (
            ellipse_points,
            fit_ellipse_least_squares,
        )

        if not isinstance(points, list) or len(points) < 4:
            raise ValueError("'points' must be a list of at least 4 tracked point names.")
        coords = np.array([self.get_point_median(p) for p in points])
        if len(points) == 4:
            warnings.warn(
                "fitting ellipse to only 4 points, using size constraint to fit ellipse",
                stacklevel=2,
            )
            cx, cy, a_len, b_len, theta = fit_ellipse_least_squares(
                coords, smallness_weight=smallness_weight
            )
        else:
            from skimage.measure import EllipseModel

            model = EllipseModel()
            model.estimate(coords)
            cx, cy, a_len, b_len, theta = model.params

        return ellipse_points(cx, cy, a_len * scaling, b_len * scaling, theta, n_points)

    def animation_stream(
        self,
        *,
        points: list[str],
        lines: list[tuple[str, str]] | None = None,
        boundaries: list[str] | None = None,
        axes: list[str] | None = None,
        features: list[str | None] | dict[str | None, str | None] | None = None,
        dims: tuple[str, ...] = ("x", "y"),
        view: dict | None = None,
        canvas_size: tuple[int, int] = (800, 800),
        bg_color: tuple[int, int, int] = (0, 0, 0),
        style: dict | None = None,
        pixel_coords: bool = False,
        undo_meta_scaling: bool = False,
    ) -> AnimationStream:
        """
        Build an OpenCV-backed animation stream from Features + boundary assets.

        **For style dict documentation and worked examples, see the
        [Animation guide](../animation.md).**

        This wraps the same renderer used by :meth:`Tracking.animation_stream`,
        while additionally resolving named boundaries stored in ``self._assets``.
        Static and dynamic boundaries are resolved to per-boundary arrays and
        rendered in boundary order.

        Args:
            points: Point names to render as circles.
            lines: Line segments connecting point pairs.
            boundaries: Boundary names (or refs resolvable by ``_resolve_boundary_ref``)
                to draw. Order controls draw stacking.
            axes: Axis asset names (registered via :meth:`define_static_axis`,
                :meth:`define_dynamic_axis`, or :meth:`import_static_axis`) to
                draw as infinite lines clipped to the canvas boundary.
                Styled via ``style["axes"]``.
            features: Per-frame scalar feature columns from ``self.data`` to render as
                text overlays. If a list is provided, each column is shown as
                ``name: value``. If a dict is provided, keys are display labels and
                values are source column names. ``None`` or ``""`` entries insert a
                blank spacer line.
            dims: Coordinate dimensions. For 3D, use ``("x","y","z")``. Boundary
                definitions are interpreted in their native 2D ``dims`` and can be
                projected in 3D via ``view``.
            view: 3D view options for projection (``azim``, ``elev``, ``proj``,
                ``camera_distance``, ``focal_length``, ``boundary_z``, ``pad``).
            canvas_size: Canvas size as ``(width, height)``.
            bg_color: Background color in BGR.
            style: Style overrides for points/lines/boundaries.
            pixel_coords: If True, coordinates are treated as absolute pixel values.
            undo_meta_scaling: If True, invert tracking meta scaling before rendering.

        Returns:
            Stream object with ``get_frame()``, ``read()``, ``play()``, and ``save()``.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> with data_path("py3r.behaviour.tracking._data", "dlc_single.csv") as p:
        ...     t = Tracking.from_dlc(str(p), handle="ex", fps=30)
        >>> f = Features(t)
        >>> f.data["speed"] = [0.0, 1.0, 0.0, 1.0, 0.0]
        >>> style = {
        ...     "points": {
        ...         "default": {"color": (0, 255, 255), "radius": 3},  # default
        ...         "p1": {"color": (0, 255, 0), "radius": 5},  # static override
        ...         "p2": {  # dynamic override (source must be in Features.data)
        ...             "radius": {"from": "speed", "map": {0.0: 2, 1.0: 6}}
        ...         },
        ...     }
        ... }
        >>> stream = f.animation_stream(
        ...     points=["p1", "p2"],
        ...     lines=[("p1", "p2")],
        ...     features={"spd": "speed"},
        ...     pixel_coords=True,
        ...     canvas_size=(96, 72),
        ...     style=style,
        ... )
        >>> stream.frame_count
        5
        >>> stream.get_frame(1).shape
        (72, 96, 3)

        ```
        """
        from py3r.behaviour.animation import (
            build_animation_stream,
            collect_dynamic_source_names_from_style,
        )

        line_points = {p for line in (lines or []) for p in line}
        all_points = sorted(set(points) | line_points)
        point_names, points_arr = self.tracking.points_to_numpy(
            all_points, dims=dims, undo_meta_scaling=undo_meta_scaling
        )
        boundary_arrays = (
            self.boundaries_to_arrays(
                boundaries,
                dims=dims,
                undo_meta_scaling=undo_meta_scaling,
            )
            if boundaries is not None
            else []
        )
        axis_arrays = (
            self.axes_to_arrays(
                axes,
                dims=(dims[0], dims[1]),
                undo_meta_scaling=undo_meta_scaling,
            )
            if axes is not None
            else []
        )
        text_overlays = None
        if features is not None:
            text_overlays = []
            if isinstance(features, dict):
                pairs = list(features.items())
            else:
                pairs = [(name, name) for name in features]
            for label, col in pairs:
                if label in (None, "") or col in (None, ""):
                    text_overlays.append(("", None))
                    continue
                if col not in self.data.columns:
                    raise ValueError(f"Feature column {col} not found for text overlay")
                text_overlays.append((str(label), self.data[col].to_numpy(copy=True)))
        style_sources = None
        if style is not None:
            needed = collect_dynamic_source_names_from_style(style)
            if needed:
                style_sources = {}
                for name in needed:
                    if name in self.data.columns:
                        style_sources[name] = self.data[name].to_numpy(copy=True)
                    else:
                        raise ValueError(
                            f"Dynamic style source '{name}' not found in Features.data"
                        )
        return build_animation_stream(
            points=points_arr,
            point_names=point_names,
            draw_points=points,
            lines=lines,
            view=view,
            boundary_z=(view or {}).get("boundary_z", 0.0),
            frame_ids=self.tracking.data.index.to_numpy(copy=True),
            fps=float(self.tracking.meta.get("fps", 30.0)),
            boundary_arrays=boundary_arrays,
            axis_arrays=axis_arrays,
            canvas_size=canvas_size,
            bg_color=bg_color,
            style=style,
            style_sources=style_sources,
            text_overlays=text_overlays,
            pixel_coords=pixel_coords,
            bounds_pad=float((view or {}).get("pad", 0.05)),
        )

    def boundaries_to_arrays(
        self,
        boundaries: list[str],
        *,
        dims: tuple[str, ...] = ("x", "y"),
        undo_meta_scaling: bool = False,
    ) -> list[tuple[str, np.ndarray]]:
        """
        Resolve named boundary assets into per-boundary arrays.

        Args:
            boundaries: Stored boundary names (or refs accepted by ``_resolve_boundary_ref``).
            dims: Requested coordinate dimensions. Boundary dims must match
                ``(dims[0], dims[1])``.
            undo_meta_scaling: If True, invert tracking scaling metadata before
                resolving dynamic boundary coordinates.

        Returns:
            Boundary arrays as ``[(boundary_name, arr), ...]`` where each arr
                has shape ``(n_frames, n_vertices, 2)``.

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> df = pd.DataFrame(
        ...     {
        ...         "a.x": [0.0, 0.0],
        ...         "a.y": [0.0, 0.0],
        ...         "b.x": [1.0, 1.0],
        ...         "b.y": [0.0, 0.0],
        ...         "c.x": [1.0, 1.0],
        ...         "c.y": [1.0, 1.0],
        ...     }
        ... )
        >>> f = Features(Tracking(df, meta={"fps": 30.0}, handle="demo"))
        >>> _ = f.define_static_boundary(["a", "b", "c"], name="tri")
        >>> arrays = f.boundaries_to_arrays(["tri"])
        >>> arrays[0][1].shape
        (2, 3, 2)

        ```
        """
        source_df = self.tracking.data
        factors = (
            self.tracking._undo_rescale_factors((dims[0], dims[1])) if undo_meta_scaling else {}
        )
        boundary_arrays: list[tuple[str, np.ndarray]] = []
        expected_boundary_dims = (dims[0], dims[1])
        for boundary_ref in boundaries:
            boundary = self._resolve_boundary_ref(boundary_ref)
            if boundary.dims != expected_boundary_dims:
                raise ValueError(
                    f"Boundary {boundary.name or boundary_ref} dims {boundary.dims} "
                    f"do not match requested xy dims {expected_boundary_dims}"
                )
            boundary_name = str(boundary_ref)
            if isinstance(boundary, StaticBoundary):
                poly = np.asarray(boundary.to_numpy(), dtype=float)
                poly_stack = np.repeat(poly[None, :, :], len(source_df), axis=0)
                poly_stack = rescale_array_by_dim(
                    poly_stack,
                    dims=(dims[0], dims[1]),
                    factors=factors,
                    dim_axis=2,
                    copy=False,
                )
                boundary_arrays.append((boundary_name, poly_stack))
            elif isinstance(boundary, DynamicBoundary):
                poly_stack = boundary.to_numpy_per_frame(source_df)
                poly_stack = rescale_array_by_dim(
                    poly_stack,
                    dims=(dims[0], dims[1]),
                    factors=factors,
                    dim_axis=2,
                    copy=False,
                )
                boundary_arrays.append((boundary_name, poly_stack))
        return boundary_arrays

    def axes_to_arrays(
        self,
        axes: list[str],
        *,
        dims: tuple[str, str] = ("x", "y"),
        undo_meta_scaling: bool = False,
    ) -> list[tuple[str, np.ndarray]]:
        """Resolve named axis assets into per-axis reference-point arrays for animation.

        Args:
            axes: Axis asset names (registered via :meth:`define_static_axis`,
                :meth:`define_dynamic_axis`, or :meth:`import_static_axis`).
            dims: Coordinate dimensions. Must be a 2-tuple; axis asset dims must match.
            undo_meta_scaling: If True, invert tracking meta scaling before
                resolving coordinates.

        Returns:
            Axis arrays as ``[(axis_name, arr), ...]`` where each arr has
                shape ``(n_frames, 2, 2)``.
        """
        source_df = self.tracking.data
        factors = (
            self.tracking._undo_rescale_factors((dims[0], dims[1])) if undo_meta_scaling else {}
        )
        result: list[tuple[str, np.ndarray]] = []
        for axis_ref in axes:
            axis = self._resolve_axis_ref(axis_ref)
            if axis.dims != dims:
                raise ValueError(
                    f"Axis {axis.name or axis_ref!r} dims {axis.dims} "
                    f"do not match requested dims {dims}"
                )
            axis_name = str(axis_ref)
            if isinstance(axis, StaticAxis):
                seg = axis.to_numpy()  # (2, 2)
                seg_stack = np.repeat(seg[np.newaxis, :, :], len(source_df), axis=0)
            else:
                seg_stack = axis.to_numpy_per_frame(source_df)  # (n, 2, 2)
            seg_stack = rescale_array_by_dim(
                seg_stack,
                dims=(dims[0], dims[1]),
                factors=factors,
                dim_axis=2,
                copy=False,
            )
            result.append((axis_name, seg_stack))
        return result
