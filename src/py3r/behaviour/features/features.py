from __future__ import annotations
import copy
import logging
import sys
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.errors import GEOSException
from sklearn.neighbors import KNeighborsRegressor
import os

from py3r.behaviour.tracking.tracking import Tracking
from py3r.behaviour.util import series_utils
from py3r.behaviour.util.bmicro_utils import (
    train_knn_from_embeddings,
    predict_knn_on_embedding,
)
from py3r.behaviour.util.collection_utils import _Indexer
from py3r.behaviour.util.dev_utils import dev_mode
from py3r.behaviour.util.series_utils import normalize_df, apply_normalization_to_df
from py3r.behaviour.features.features_result import FeaturesResult
from py3r.behaviour.util.io_utils import (
    SchemaVersion,
    begin_save,
    write_manifest,
    read_manifest,
    write_dataframe,
    read_dataframe,
)

if TYPE_CHECKING:
    from py3r.behaviour.classifier import BaseClassifier
    import pandas as pd
    from sklearn.neighbors import KNeighborsRegressor

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
                "distance has not been calibrated on these tracking data. some methods will be unavailable"
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
    def load(cls, dirpath: str) -> "Features":
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

    def distance_between(
        self, point1: str, point2: str, dims=("x", "y")
    ) -> FeaturesResult:
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
            warnings.warn("distance has not been calibrated")
        if "smoothing" not in self.tracking.meta.keys():
            warnings.warn("tracking data have not been smoothed")

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
        result = obs_distance <= distance
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
        return tuple(self.tracking.data[point + "." + dim].median() for dim in dims)

    def define_boundary(
        self,
        points: list[str],
        scaling: float,
        scaling_y: float = None,
        centre: str | list[str] = None,
    ) -> list[tuple[float, float]]:
        """
        takes a list of defined points, and creates a static rescaled list of point coordinates based on median location of those points
        'centre' (point about which to scale) can be a string or list of strings, in which case the median of the points will be used as the centre
        if 'centre' is None, the median of all the boundary points will be used as the centre
        'scaling' is the factor by which to scale the boundary points, and 'scaling_y' is the factor by which to scale the y-axis
        if 'scaling_y' is not provided, 'scaling' will be applied to both axes
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
                raise ValueError(
                    f"centre must be a string or list of strings, not {type(centre)}"
                )
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

        def local_contains_static(x):
            px, py = x[point + ".x"], x[point + ".y"]
            if pd.isna(px) or pd.isna(py) or boundary_has_nan:
                return np.nan
            local_point = Point(px, py)
            local_poly = Polygon(boundary)
            return local_poly.contains(local_point)

        result = self.tracking.data.apply(local_contains_static, axis=1)
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

        def local_contains_dynamic(x):
            px, py = x[point + ".x"], x[point + ".y"]
            bdry_pts = [(x[i + ".x"], x[i + ".y"]) for i in boundary]
            boundary_has_nan = any(pd.isna(bx) or pd.isna(by) for bx, by in bdry_pts)
            if pd.isna(px) or pd.isna(py) or boundary_has_nan:
                return np.nan
            local_point = Point(px, py)
            local_poly = Polygon(bdry_pts)
            return local_poly.contains(local_point)

        result = self.tracking.data.apply(local_contains_dynamic, axis=1)
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
                return self.within_boundary_static(
                    point, static_boundary, boundary_name
                )
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
            "distance_to_boundary is deprecated, use distance_to_boundary_static or distance_to_boundary_dynamic",
            DeprecationWarning,
            stacklevel=2,
        )
        if median:
            static_boundary = self.define_boundary(boundary, 1.0)
            return self.distance_to_boundary_static(
                point, static_boundary, boundary_name
            )
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

        def row_distance(x):
            px, py = x[point + ".x"], x[point + ".y"]
            if pd.isna(px) or pd.isna(py) or boundary_has_nan:
                return np.nan
            local_point = Point(px, py)
            local_poly = Polygon(boundary)
            return local_poly.exterior.distance(local_point)

        result = self.tracking.data.apply(row_distance, axis=1)
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

        def row_distance(x):
            px, py = x[point + ".x"], x[point + ".y"]
            bdry_pts = [(x[i + ".x"], x[i + ".y"]) for i in boundary]
            boundary_has_nan = any(pd.isna(bx) or pd.isna(by) for bx, by in bdry_pts)
            if pd.isna(px) or pd.isna(py) or boundary_has_nan:
                return np.nan
            local_point = Point(px, py)
            local_poly = Polygon(bdry_pts)
            return local_poly.exterior.distance(local_point)

        result = self.tracking.data.apply(row_distance, axis=1)
        return FeaturesResult(result, self, name, meta)

    def area_of_boundary(
        self, boundary: list[str], median: bool = True
    ) -> FeaturesResult:
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
        name = f"area_of_boundary_{self._short_boundary_id(boundary)}_{'static' if median else 'dynamic'}"
        meta = {"function": "area_of_boundary", "boundary": boundary, "median": median}
        if median:
            warnings.warn("using median (static) boundary")
            static_boundary = [self.get_point_median(i) for i in boundary]
            local_poly = Polygon(static_boundary)
            area = local_poly.area
            # Create a constant Series with the same index as self.tracking.data
            result = pd.Series(area, index=self.tracking.data.index)
        else:
            warnings.warn("using fully dynamic boundary")

            def row_area(x):
                try:
                    local_poly = Polygon([(x[i + ".x"], x[i + ".y"]) for i in boundary])
                    return local_poly.area
                except GEOSException:
                    return np.nan

            result = self.tracking.data.apply(row_area, axis=1)
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
            warnings.warn("tracking data have not been smoothed")
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
            warnings.warn("tracking data have not been smoothed")

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
        name = (
            f"azimuth_deviation_{basepoint}_to_{pointdirection1}_and_{pointdirection2}"
        )
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
        obs_deviation = self.azimuth_deviation(
            basepoint, pointdirection1, pointdirection2
        )
        result = obs_deviation <= deviation
        name = f"within_azimuth_deviation_{basepoint}_to_{pointdirection1}_and_{pointdirection2}_leq_{deviation}"
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
            warnings.warn("distance has not been calibrated")
        if "smoothing" not in self.tracking.meta.keys():
            warnings.warn("tracking data have not been smoothed")

        result = self.distance_change(point, dims=dims) * self.tracking.meta["fps"]
        name = f"speed_of_{point}_in_{''.join(dims)}"
        meta = {"function": "speed", "point": point, "dims": dims}
        return FeaturesResult(result, self, name, meta)

    def above_speed(self, point: str, speed: float, dims=("x", "y")) -> FeaturesResult:
        """
        Return True for frames where the point's speed is >= threshold.

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
        result = obs_speed >= speed
        name = f"above_speed_{point}_geq_{speed}_in_{''.join(dims)}"
        meta = {"function": "above_speed", "point": point, "speed": speed, "dims": dims}
        return FeaturesResult(result, self, name, meta)

    def all_above_speed(
        self, points: list, speed: float, dims=("x", "y")
    ) -> FeaturesResult:
        """
        Return True for frames where all listed points are moving at least at the threshold speed.

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
        df = pd.DataFrame(
            [self.above_speed(point, speed, dims=dims) for point in points]
        )
        result = df.all(axis=0)
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
        result = obs_speed < speed
        name = f"below_speed_{point}_lt_{speed}_in_{''.join(dims)}"
        meta = {"function": "below_speed", "point": point, "speed": speed, "dims": dims}
        return FeaturesResult(result, self, name, meta)

    def all_below_speed(
        self, points: list, speed: float, dims=("x", "y")
    ) -> FeaturesResult:
        """
        Return True for frames where all listed points are moving slower than the threshold speed.

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
        df = pd.DataFrame(
            [self.below_speed(point, speed, dims=dims) for point in points]
        )
        result = df.all(axis=0)
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
        returns unsigned distance moved by point from previous frame to current frame, for each frame

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
            warnings.warn("distance has not been calibrated")
        if "smoothing" not in self.tracking.meta.keys():
            warnings.warn("tracking data have not been smoothed")

        result = np.sqrt(
            sum([(self.tracking.data[point + "." + dim].diff()) ** 2 for dim in dims])
        )
        name = f"distance_change_{point}_in_{''.join(dims)}"
        meta = {"function": "distance_change", "point": point, "dims": dims}
        return FeaturesResult(result, self, name, meta)

    def store(
        self,
        feature: pd.Series,
        name: str,
        overwrite: bool = False,
        meta: dict = dict(),
    ) -> None:
        """
        stores calculated feature with name and associated freeform metadata object

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
        if name in self.data.columns:
            if overwrite:
                self.data[name] = feature
                warnings.warn("feature '" + name + "' overwritten")
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
        classify behaviour using a classifier with inputs from this Features object
        returns a FeaturesResult object with the classification result
        this means that the output of the classifier should be a pd.Series with the same index as this Features object
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
        center: bool = True,
        inplace: bool = False,
    ) -> pd.Series:
        """
        smooths specified feature with specified method over rolling window. if inplace=True then feature
        will be directly edited and metadata updated
        method:
            'median' : median of value in window, requires numerical series values
            'mean' : mean of value in window, requires numerical series values
            'mode' : mode value in window, works with numerical or non-numerical types
            'block' : removes labels that occur in blocks of less than length window
                      and replaces them with value from previous block unless there is
                      no previous block, in which case replaced from next block after smoothing
                      note: all nan values will be filled using this method (dangerous!)
        """
        if "smoothing" in self.meta[name].keys():
            raise Exception("feature already smoothed")

        if method == "median":
            smoothed = self.data[name].rolling(window=window, center=center).median()
            if inplace:
                self.data[name] = smoothed.copy()
        elif method == "mean":
            smoothed = self.data[name].rolling(window=window, center=center).mean()
            if inplace:
                self.data[name] = smoothed.copy()
        elif method == "mode":
            smoothed = series_utils.rolling_apply(
                self.data[name], window, series_utils.mode, center=center
            )
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
                "center": center,
            }
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
            raise ValueError(
                f"The following columns are not present in self.data: {missing}"
            )
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

    def assign_clusters_by_centroids(
        self, embedding: dict[str, list[int]], centroids_df: pd.DataFrame
    ) -> pd.Series:
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
        # check that columns are the same
        if not embed_df.columns.equals(centroids_df.columns):
            raise ValueError("Columns in embedding and centroids do not match")

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
            "centroids_df": centroids_df,
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

        Train a KNN regressor to predict a target embedding from a feature embedding on this Features object.
        If normalize_source is True, normalize the source embedding before training and return the rescale factors.
        Returns the trained model, input columns, target columns, and (optionally) the rescale factors.
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

        Compute the root mean squared error (RMS) for each row between two embedding DataFrames.
        If rescale is a dict, normalize both DataFrames using this dict before computing the error.
        If rescale == 'auto', compute normalization factors from ground_truth and apply to both DataFrames.
        Returns a Series indexed like the input DataFrames, with NaN for rows where either input has NaNs.
        """
        if not ground_truth.columns.equals(
            prediction.columns
        ) or not ground_truth.index.equals(prediction.index):
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
        `centre` can be a single point name or a list of point names.
        if `centre` is a list, the boundary will be centred on the mean of the median coordinates of the points.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> poly = f.define_elliptical_boundary_from_params('p1', major_axis_length=10, minor_axis_length=6, angle_in_radians=0.0, n_points=32)
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
        >>> poly = f.define_elliptical_boundary_from_points(['p1','p2','p3','p2'], n_points=20, scaling=1.0)
        >>> isinstance(poly, list) and len(poly) == 20
        True

        ```
        """
        from py3r.behaviour.util.ellipse_utils import (
            ellipse_points,
            fit_ellipse_least_squares,
        )
        import numpy as np

        if not isinstance(points, list) or len(points) < 4:
            raise ValueError(
                "'points' must be a list of at least 4 tracked point names."
            )
        coords = np.array([self.get_point_median(p) for p in points])
        if len(points) == 4:
            warnings.warn(
                "fitting ellipse to only 4 points, using size constraint to fit ellipse"
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
