from __future__ import annotations

import copy
import logging
import os
import sys
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon
from sklearn.neighbors import KNeighborsRegressor

from py3r.behaviour.features.features_result import FeaturesResult
from py3r.behaviour.tracking.tracking import Tracking
from py3r.behaviour.util import series_utils
from py3r.behaviour.util.bmicro_utils import (
    predict_knn_on_embedding,
    train_knn_from_embeddings,
)
from py3r.behaviour.util.collection_utils import _Indexer
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
    apply_custom_scaling,
    apply_normalization_to_df,
    normalize_df,
)
from py3r.behaviour.util.smoothing import smooth_series

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.neighbors import KNeighborsRegressor

    from py3r.behaviour.classifier import BaseClassifier

logger = logging.getLogger(__name__)
logformat = "%(funcName)s(): %(message)s"
logging.basicConfig(stream=sys.stdout, format=logformat)
logger.setLevel(logging.INFO)


class Features:
    """
    generates features from a pre-processed Tracking object
    """

    def __init__(self, tracking: Tracking) -> None:
        self.tracking = tracking
        self.data = pd.DataFrame()
        self.meta = dict()
        self.handle = tracking.handle
        self.tags = tracking.tags
        if "usermeta" in tracking.meta:
            self.meta["usermeta"] = tracking.meta["usermeta"]

        if "rescale_distance_method" not in self.tracking.meta.keys():
            warnings.warn(
                "distance has not been calibrated on these tracking data. "
                "some methods will be unavailable",
                stacklevel=2,
            )

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
        result.handle = self.handle
        result.tags = copy.deepcopy(self.tags)
        return result

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

        Parameters
        ----------
        features_list : list[Features]
            List of Features objects to concatenate, in temporal order.
        handle : str, optional
            Handle for the concatenated object. If None, uses first object's handle.
        reindex : {"rezero", "follow_previous", "keep_original"}, default "follow_previous"
            How to handle frame indices:
            - "rezero": Reindex all frames starting from 0 (0, 1, 2, ...).
            - "follow_previous": Each chunk continues from where the previous
              ended. If chunk 1 ends at frame n, chunk 2 starts at n+1.
            - "keep_original": Leave indices untouched; duplicates are allowed.

        Returns
        -------
        Features
            A new Features object containing all frames from input objects.

        Raises
        ------
        ValueError
            If features_list is empty, fps values don't match, or columns differ.

        Notes
        -----
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
        returns distance from point1 to point2

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
        returns True for frames where point1 is within specified distance of point2
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
        """
        Static rescaled boundary from point medians.
        'centre' can be str or list[str]; median of those points or of boundary if None.
        'scaling' scales boundary; 'scaling_y' scales y-axis (default: same as scaling).
        """

        # get point medians
        pointmedians = [self.get_point_median(point) for point in points]
        # get centre
        if centre is not None:
            if isinstance(centre, str):
                boundarycentre = self.get_point_median(centre)
            elif isinstance(centre, list):
                centrepointmedians = [self.get_point_median(point) for point in centre]
                xcoords = np.array([point[0] for point in centrepointmedians])
                ycoords = np.array([point[1] for point in centrepointmedians])
                boundarycentre = (xcoords.mean(), ycoords.mean())
            else:
                raise ValueError(f"centre must be a string or list of strings, not {type(centre)}")
        else:
            xcoords = np.array([point[0] for point in pointmedians])
            ycoords = np.array([point[1] for point in pointmedians])
            boundarycentre = (xcoords.mean(), ycoords.mean())

        def rescale(val1: float, val2: float, factor: float) -> float:
            output = val1 + (val2 - val1) * (1 - factor)
            return output

        if scaling_y is not None:
            rescaledpoints = [
                (
                    rescale(point[0], boundarycentre[0], scaling),
                    rescale(point[1], boundarycentre[1], scaling_y),
                )
                for point in pointmedians
            ]
        else:
            rescaledpoints = [
                (
                    rescale(point[0], boundarycentre[0], scaling),
                    rescale(point[1], boundarycentre[1], scaling),
                )
                for point in pointmedians
            ]

        return rescaledpoints

    @staticmethod
    def _short_boundary_id(boundary):
        b = [str(x) for x in boundary]
        if len(b) <= 4:
            return "_".join(b)
        return "_".join(b[:2] + ["..."] + b[-2:])

    def within_boundary_static(
        self, point: str, boundary: list[tuple[float, float]], boundary_name: str = None
    ) -> FeaturesResult:
        """
        checks whether point is inside polygon defined by ordered list of boundary points
        boundary points must be specified as a list of numerical tuples

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
        >>> boundary = f.define_boundary(['p1','p2','p3'], scaling=1.0)
        >>> res = f.within_boundary_static('p1', boundary)
        >>> bool((isinstance(res, pd.Series) and res.notna().any()))
        True

        ```
        """
        if len(boundary) < 3:
            raise Exception("boundary encloses no area")
        boundary_has_nan = any(pd.isna(bx) or pd.isna(by) for bx, by in boundary)
        boundary_id = self._short_boundary_id(boundary)
        name = f"within_boundary_static_{point}_in_{boundary_name or boundary_id}"
        meta = {
            "function": "within_boundary_static",
            "point": point,
            "boundary": boundary,
        }
        if boundary_name is not None:
            meta["boundary_name"] = boundary_name

        df = self.tracking.data
        px = df[point + ".x"].to_numpy(dtype=float)
        py = df[point + ".y"].to_numpy(dtype=float)
        point_nan = np.isnan(px) | np.isnan(py)

        if boundary_has_nan:
            result = pd.Series(pd.array([pd.NA] * len(df), dtype="boolean"), index=df.index)
        else:
            poly = Polygon(boundary)
            contained = shapely.contains(poly, shapely.points(px, py))
            result = pd.Series(pd.array(contained, dtype="boolean"), index=df.index)
            result[point_nan] = pd.NA

        return FeaturesResult(result, self, name, meta)

    def within_boundary_dynamic(
        self, point: str, boundary: list[str], boundary_name: str = None
    ) -> FeaturesResult:
        """
        checks whether point is inside polygon defined by ordered list of boundary points
        boundary points must be specified as a list of names of tracked points

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
        >>> res = f.within_boundary_dynamic('p1', ['p1','p2','p3'])
        >>> bool((isinstance(res, pd.Series) and res.notna().any()))
        True

        ```
        """
        if len(boundary) < 3:
            raise Exception("boundary encloses no area")

        boundary_id = self._short_boundary_id(boundary)
        name = f"within_boundary_dynamic_{point}_in_{boundary_name or boundary_id}"
        meta = {
            "function": "within_boundary_dynamic",
            "point": point,
            "boundary": boundary,
        }
        if boundary_name is not None:
            meta["boundary_name"] = boundary_name

        df = self.tracking.data
        px = df[point + ".x"].to_numpy(dtype=float)
        py = df[point + ".y"].to_numpy(dtype=float)
        bx = np.column_stack([df[b + ".x"].to_numpy(dtype=float) for b in boundary])
        by = np.column_stack([df[b + ".y"].to_numpy(dtype=float) for b in boundary])

        valid = ~(np.isnan(px) | np.isnan(py) | np.any(np.isnan(bx) | np.isnan(by), axis=1))

        result = pd.Series(pd.array([pd.NA] * len(df), dtype="boolean"), index=df.index)
        if valid.any():
            coords = np.stack([bx[valid], by[valid]], axis=-1)
            polys = shapely.polygons(shapely.linearrings(coords))
            pts = shapely.points(px[valid], py[valid])
            result[valid] = shapely.contains(polys, pts)

        return FeaturesResult(result, self, name, meta)

    def within_boundary(
        self, point: str, boundary: list, median: bool = True, boundary_name: str = None
    ) -> FeaturesResult:
        """
        deprecated: use within_boundary_static or within_boundary_dynamic instead
        checks whether point is inside polygon defined by ordered list of boundary points
        boundary points may either be specified as a list of numerical tuples,
        or as a list of names of tracked points.
        Optionally, pass boundary_name for a custom short name in the feature name/meta.
        """
        warnings.warn(
            "within_boundary is deprecated, use within_boundary_static or within_boundary_dynamic",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(boundary[0], str):
            if not median:
                return self.within_boundary_dynamic(point, boundary, boundary_name)
            if median:
                static_boundary = self.define_boundary(boundary, 1.0)
                return self.within_boundary_static(point, static_boundary, boundary_name)
        else:
            return self.within_boundary_static(point, boundary, boundary_name)

    def distance_to_boundary(
        self,
        point: str,
        boundary: list[str],
        median: bool = True,
        boundary_name: str = None,
    ) -> FeaturesResult:
        """
        Deprecated: use distance_to_boundary_static or distance_to_boundary_dynamic instead
        returns distance from point to boundary
        Optionally, pass boundary_name for a custom short name in the feature name/meta.
        """
        warnings.warn(
            "distance_to_boundary is deprecated; use "
            "distance_to_boundary_static or distance_to_boundary_dynamic",
            DeprecationWarning,
            stacklevel=2,
        )
        if median:
            static_boundary = self.define_boundary(boundary, 1.0)
            return self.distance_to_boundary_static(point, static_boundary, boundary_name)
        else:
            return self.distance_to_boundary_dynamic(point, boundary, boundary_name)

    def distance_to_boundary_static(
        self, point: str, boundary: list[tuple[float, float]], boundary_name: str = None
    ) -> FeaturesResult:
        """
        Returns distance from point to a static boundary defined by a list of (x, y) tuples.
        If boundary_name is provided, it overrides the automatic id.
        NaN is returned if the point or any boundary vertex is NaN.
        """
        if len(boundary) < 3:
            raise Exception("boundary encloses no area")
        boundary_has_nan = any(pd.isna(bx) or pd.isna(by) for bx, by in boundary)
        boundary_id = self._short_boundary_id(boundary)
        name = f"distance_to_boundary_static_{point}_in_{boundary_name or boundary_id}"
        meta = {
            "function": "distance_to_boundary_static",
            "point": point,
            "boundary": boundary,
        }
        if boundary_name is not None:
            meta["boundary_name"] = boundary_name

        df = self.tracking.data
        px = df[point + ".x"].to_numpy(dtype=float)
        py = df[point + ".y"].to_numpy(dtype=float)
        point_nan = np.isnan(px) | np.isnan(py)

        if boundary_has_nan:
            result = pd.Series(np.nan, index=df.index)
        else:
            exterior = Polygon(boundary).exterior
            distances = shapely.distance(exterior, shapely.points(px, py))
            distances[point_nan] = np.nan
            result = pd.Series(distances, index=df.index)

        return FeaturesResult(result, self, name, meta)

    def distance_to_boundary_dynamic(
        self, point: str, boundary: list[str], boundary_name: str | None = None
    ) -> FeaturesResult:
        """
        Returns distance from point to a dynamic boundary defined by a list of point names.
        If boundary_name is provided, it overrides the automatic id.
        NaN is returned if the point or any boundary vertex is NaN.
        """
        if len(boundary) < 3:
            raise Exception("boundary encloses no area")
        boundary_id = self._short_boundary_id(boundary)
        name = f"distance_to_boundary_dynamic_{point}_in_{boundary_name or boundary_id}"
        meta = {
            "function": "distance_to_boundary_dynamic",
            "point": point,
            "boundary": boundary,
        }
        if boundary_name is not None:
            meta["boundary_name"] = boundary_name

        df = self.tracking.data
        px = df[point + ".x"].to_numpy(dtype=float)
        py = df[point + ".y"].to_numpy(dtype=float)
        bx = np.column_stack([df[b + ".x"].to_numpy(dtype=float) for b in boundary])
        by = np.column_stack([df[b + ".y"].to_numpy(dtype=float) for b in boundary])

        valid = ~(np.isnan(px) | np.isnan(py) | np.any(np.isnan(bx) | np.isnan(by), axis=1))

        result = pd.Series(np.nan, index=df.index)
        if valid.any():
            coords = np.stack([bx[valid], by[valid]], axis=-1)
            polys = shapely.polygons(shapely.linearrings(coords))
            exteriors = shapely.get_exterior_ring(polys)
            pts = shapely.points(px[valid], py[valid])
            result[valid] = shapely.distance(exteriors, pts)

        return FeaturesResult(result, self, name, meta)

    def area_of_boundary(self, boundary: list[str], median: bool = True) -> FeaturesResult:
        """
        returns area of boundary as a FeaturesResult (constant for static, per-frame for dynamic)

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
        >>> res = f.area_of_boundary(['p1','p2','p3'], median=True)
        >>> isinstance(res, pd.Series) and res.nunique() == 1
        True

        ```
        """
        kind = "static" if median else "dynamic"
        name = f"area_of_boundary_{self._short_boundary_id(boundary)}_{kind}"
        meta = {"function": "area_of_boundary", "boundary": boundary, "median": median}
        if median:
            warnings.warn("using median (static) boundary", stacklevel=2)
            static_boundary = [self.get_point_median(i) for i in boundary]
            local_poly = Polygon(static_boundary)
            area = local_poly.area
            # Create a constant Series with the same index as self.tracking.data
            result = pd.Series(area, index=self.tracking.data.index)
        else:
            warnings.warn("using fully dynamic boundary", stacklevel=2)
            data = self.tracking.data
            bx = np.column_stack([data[b + ".x"].to_numpy(dtype=float) for b in boundary])
            by = np.column_stack([data[b + ".y"].to_numpy(dtype=float) for b in boundary])
            bx_next = np.roll(bx, -1, axis=1)
            by_next = np.roll(by, -1, axis=1)
            result = pd.Series(
                0.5 * np.abs(np.sum(bx * by_next - bx_next * by, axis=1)),
                index=data.index,
            )
        return FeaturesResult(result, self, name, meta)

    def acceleration(self, point: str, dims=("x", "y")) -> FeaturesResult:
        """
        returns acceleration of point from previous frame to current frame, for each frame

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
        returns azimuth in radians from tracked point1 to tracked point2
        for each frame in the data, relative to the direction of the x-axis

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
        returns average speed of point from previous frame to current frame, for each frame

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
        **method_kwargs,
    ) -> pd.Series:
        """
        Smooth feature with method over rolling window. If inplace=True, feature and
        metadata are updated in place.
        method:
            'median' : median in window (numerical)
            'mean' : mean in window (numerical)
            'savgol' : Savitzky–Golay (SciPy). Kwargs e.g. polyorder=3, mode='interp'.
            'mode' : mode in window (numerical or non-numerical)
            'block' : removes labels that occur in blocks of less than length window
                      and replaces them with value from previous block unless there is
                      no previous block, in which case replaced from next block after smoothing
                      note: all nan values will be filled using this method (dangerous!)
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
            smoothed = series_utils.smooth_block(self.data[name], window)
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

    def embedding_df(self, embedding: dict[str, list[int]]):
        """
        generate a time series embedding dataframe with specified time shifts for each column,
        where embedding is a dict mapping column names to lists of shifts
        positive shift: value from the future (t+n)
        negative shift: value from the past (t-n)

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

    def cluster_embedding(
        self,
        embedding_dict: dict[str, list[int]],
        n_clusters: int,
        random_state: int = 0,
        *,
        auto_normalize: bool = False,
        rescale_factors: dict | None = None,
        lowmem: bool = False,
        decimation_factor: int = 10,
        custom_scaling: dict[str, dict] | None = None,
        missing_policy: Literal["drop", "impute_weight"] = "drop",
    ):
        """
        Perform k-means clustering on a single Features object.

        Delegates to ``FeaturesCollection.cluster_embedding`` so all
        clustering logic remains centralised.

        Returns
        -------
        (FeaturesResult, centroids DataFrame, normalization_factors or None)

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
        >>> result, centroids, norm = f.cluster_embedding({'counter': [0]}, n_clusters=2)
        >>> isinstance(centroids, pd.DataFrame)
        True
        >>> len(result) == len(f.data)
        True

        ```
        """
        from py3r.behaviour.features.features_collection import FeaturesCollection

        fc = FeaturesCollection.from_list([self])
        batch, centroids, norm = fc.cluster_embedding(
            embedding_dict,
            n_clusters,
            random_state,
            auto_normalize=auto_normalize,
            rescale_factors=rescale_factors,
            lowmem=lowmem,
            decimation_factor=decimation_factor,
            custom_scaling=custom_scaling,
            missing_policy=missing_policy,
        )
        return batch[self.handle], centroids, norm

    def cluster_embedding_stream(
        self,
        embedding_dict: dict[str, list[int]],
        n_clusters: int,
        random_state: int = 0,
        *,
        auto_normalize: bool = False,
        rescale_factors: dict | None = None,
        custom_scaling: dict[str, dict] | None = None,
        missing_policy: Literal["drop", "impute_weight"] = "drop",
        chunk_size: int = 10_000,
        n_epochs: int = 3,
        batch_size: int = 1024,
    ):
        """
        Memory-friendly clustering on a single Features object.

        Delegates to ``FeaturesCollection.cluster_embedding_stream``.
        See that method for full parameter documentation.

        Returns
        -------
        (FeaturesResult, centroids DataFrame, scaling_factors or None)

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
        >>> result, centroids, norm = f.cluster_embedding_stream(
        ...     {'counter': [0]}, n_clusters=2)
        >>> isinstance(centroids, pd.DataFrame)
        True
        >>> len(result) == len(f.data)
        True

        ```
        """
        from py3r.behaviour.features.features_collection import FeaturesCollection

        fc = FeaturesCollection.from_list([self])
        batch, centroids, scaling = fc.cluster_embedding_stream(
            embedding_dict,
            n_clusters,
            random_state,
            auto_normalize=auto_normalize,
            rescale_factors=rescale_factors,
            custom_scaling=custom_scaling,
            missing_policy=missing_policy,
            chunk_size=chunk_size,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )
        return batch[self.handle], centroids, scaling

    def assign_clusters_by_centroids(
        self,
        embedding: dict[str, list[int]],
        centroids_df: pd.DataFrame,
        *,
        rescale_factors: dict | None = None,
        custom_scaling: dict[str, dict] | None = None,
        impute_medians: pd.Series | None = None,
    ) -> FeaturesResult:
        """
        new_embed_df: (n_samples, n_features)  DataFrame of your new time-shifted embedding
        centroids_df: (n_clusters, n_features) DataFrame of cluster centers
        Returns a Series of cluster IDs (0..n_clusters-1) indexed like new_embed_df.

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
        >>> labels = f.assign_clusters_by_centroids(emb, cents)
        >>> isinstance(labels, pd.Series) and len(labels) == len(t.data)
        True

        ```
        """
        from sklearn.metrics.pairwise import pairwise_distances_argmin

        embed_df = self.embedding_df(embedding)
        # Apply the same scaling/normalization used during centroid fitting, if provided
        if rescale_factors is not None and custom_scaling is not None:
            raise ValueError("rescale_factors and custom_scaling are mutually exclusive")
        if rescale_factors is not None:
            embed_df = apply_normalization_to_df(embed_df, rescale_factors)
        elif custom_scaling is not None:
            embed_df = apply_custom_scaling(embed_df, custom_scaling)
        # check that columns are the same
        if not embed_df.columns.equals(centroids_df.columns):
            raise ValueError("Columns in embedding and centroids do not match")

        # Optionally impute using provided medians (from training)
        if impute_medians is not None:
            embed_df, _ = impute_frame(embed_df, impute_medians)
            mask = pd.Series(True, index=embed_df.index)
            embed_values = embed_df.values
        else:
            mask = embed_df.notna().all(axis=1)
            embed_values = embed_df[mask].values
        centroids_values = centroids_df.values

        labels = pd.Series(pd.NA, index=embed_df.index, dtype="Int64")
        if len(embed_values) > 0:
            labels[mask] = pairwise_distances_argmin(embed_values, centroids_values)

        name = f"kmeans_{len(centroids_df.index)}"
        meta = {
            "function": "assign_clusters_by_centroids",
            "embedding": embedding,
            "rescale_factors": rescale_factors,
            "custom_scaling": custom_scaling,
            "impute_medians": (impute_medians.to_dict() if impute_medians is not None else None),
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
        new.handle = self.handle
        return new

    def _iloc(self, idx):
        if isinstance(idx, tuple):
            row_idx, col_idx = idx
            new_tracking = self.tracking.loc[row_idx]
        else:
            new_tracking = self.tracking.loc[idx]
        new = self.__class__(new_tracking)
        new.data = self.data.iloc[idx].copy()
        new.meta = copy.deepcopy(self.meta)
        new.handle = self.handle
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
