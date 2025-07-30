from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import shapely as sp
from shapely.errors import GEOSException

import warnings
import logging
import sys
import copy

from py3r.behaviour.tracking import (
    Tracking,
    TrackingCollection,
    MultipleTrackingCollection,
)
from py3r.behaviour.exceptions import BatchProcessError
from py3r.behaviour.util import series_utils
from py3r.behaviour.util.bmicro_utils import (
    train_knn_from_embeddings,
    predict_knn_on_embedding,
)
from py3r.behaviour.util.collection_utils import _Indexer, BatchResult
from py3r.behaviour.util.series_utils import normalize_df, apply_normalization_to_df

logger = logging.getLogger(__name__)
logformat = "%(funcName)s(): %(message)s"
logging.basicConfig(stream=sys.stdout, format=logformat)
logger.setLevel(logging.INFO)


class FeaturesResult(pd.Series):
    def __init__(self, series, features_obj, column_name, params):
        super().__init__(series)
        self._features_obj = features_obj
        self._column_name = column_name
        self._params = params
        self.name = column_name  # Set the Series name for plotting/legend

    def store(self, name=None, meta=None, overwrite=False):
        if name is None:
            name = self._column_name
        if meta is None:
            meta = self._params
        self._features_obj.store(self, name, overwrite=overwrite, meta=meta)
        return name

    @property
    def _constructor(self):
        # Ensures pandas operations return pd.Series, not FeaturesResult
        return pd.Series


class Features:
    """
    generates features from a pre-processed Tracking object
    """

    def __init__(self, tracking: Tracking) -> None:
        self.tracking = tracking
        self.data = pd.DataFrame()
        self.meta = dict()
        self.handle = tracking.handle
        if "usermeta" in tracking.meta:
            self.meta["usermeta"] = tracking.meta["usermeta"]

        if "rescale_distance_method" not in self.tracking.meta.keys():
            warnings.warn(
                "distance has not been calibrated on these tracking data. some methods will be unavailable"
            )

    def distance_between(
        self, point1: str, point2: str, dims=("x", "y")
    ) -> FeaturesResult:
        """
        returns distance from point1 to point2
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
        """returns average speed of point from previous frame to current frame, for each frame"""
        if "rescale_distance_method" not in self.tracking.meta.keys():
            warnings.warn("distance has not been calibrated")
        if "smoothing" not in self.tracking.meta.keys():
            warnings.warn("tracking data have not been smoothed")

        result = self.distance_change(point, dims=dims) * self.tracking.meta["fps"]
        name = f"speed_of_{point}_in_{''.join(dims)}"
        meta = {"function": "speed", "point": point, "dims": dims}
        return FeaturesResult(result, self, name, meta)

    def above_speed(self, point: str, speed: float, dims=("x", "y")) -> FeaturesResult:
        obs_speed = self.speed(point, dims=dims)
        result = obs_speed >= speed
        name = f"above_speed_{point}_geq_{speed}_in_{''.join(dims)}"
        meta = {"function": "above_speed", "point": point, "speed": speed, "dims": dims}
        return FeaturesResult(result, self, name, meta)

    def all_above_speed(
        self, points: list, speed: float, dims=("x", "y")
    ) -> FeaturesResult:
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
        obs_speed = self.speed(point, dims=dims)
        result = obs_speed < speed
        name = f"below_speed_{point}_lt_{speed}_in_{''.join(dims)}"
        meta = {"function": "below_speed", "point": point, "speed": speed, "dims": dims}
        return FeaturesResult(result, self, name, meta)

    def all_below_speed(
        self, points: list, speed: float, dims=("x", "y")
    ) -> FeaturesResult:
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
        """returns unsigned distance moved by point from previous frame to current frame, for each frame"""
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
        """stores calculated feature with name and associated freeform metadata object"""
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

    def cluster_embedding(
        self, embedding: dict[str, list[int]], n_clusters: int
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        cluster the embedding using k-means,
        ensuring that the cluster label is nan where a row in the embedding has nan values
        returns the labels and the centroids
        """
        embed_df = self.embedding_df(embedding)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embed_df)
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=embed_df.columns)
        labels = pd.Series(kmeans.labels_, index=self.data.index)
        labels.loc[embed_df.isna().any(axis=1)] = np.nan
        return labels, centroids

    def assign_clusters_by_centroids(
        self, embedding: dict[str, list[int]], centroids_df: pd.DataFrame
    ) -> pd.Series:
        """
        new_embed_df: (n_samples, n_features)  DataFrame of your new time-shifted embedding
        centroids_df: (n_clusters, n_features) DataFrame of cluster centers
        Returns a Series of cluster IDs (0..n_clusters-1) indexed like new_embed_df.
        """
        embed_df = self.embedding_df(embedding)
        # check that columns are the same
        if not embed_df.columns.equals(centroids_df.columns):
            raise ValueError("Columns in embedding and centroids do not match")

        mask = embed_df.notna().all(axis=1)
        embed_values = embed_df[mask].values
        centroids_values = centroids_df.values

        # compute squared Euclidean distances: result shape (n_samples, n_clusters)
        d2 = np.sum(
            (embed_values[:, None, :] - centroids_values[None, :, :]) ** 2, axis=2
        )
        labels = np.full(len(embed_df), fill_value=np.nan)
        labels[mask] = np.argmin(d2, axis=1)

        return pd.Series(labels, index=embed_df.index, name="cluster")

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

    def predict_knn(
        self,
        model: KNeighborsRegressor,
        source_embedding: dict[str, list[int]],
        target_embedding: dict[str, list[int]],
        rescale_factors: dict = None,
    ) -> pd.DataFrame:
        """
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

    @staticmethod
    def rms_error_between_embeddings(
        ground_truth: pd.DataFrame, prediction: pd.DataFrame, rescale: dict | str = None
    ) -> pd.Series:
        """
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


class FeaturesCollection:
    """
    Collection of Features objects, keyed by name.
    note: type-hints refer to Features, but factory methods allow for other classes
    these are intended ONLY for subclasses of Features, and this is enforced
    """

    features_dict: dict[str, Features]

    def __init__(self, features_dict: dict[str, Features]):
        self.features_dict = features_dict

    def __getattr__(self, name):
        def batch_method(*args, **kwargs):
            results = {}
            for key, obj in self.features_dict.items():
                try:
                    method = getattr(obj, name)
                    results[key] = method(*args, **kwargs)
                except Exception as e:
                    raise BatchProcessError(
                        collection_name=None,
                        object_name=getattr(e, "object_name", key),
                        method=getattr(e, "method", name),
                        original_exception=getattr(e, "original_exception", e),
                    ) from e
            return BatchResult(results, self)

        return batch_method

    @classmethod
    def from_tracking_collection(
        cls, tracking_collection: TrackingCollection, feature_cls=Features
    ):
        """
        Create a FeaturesCollection from a TrackingCollection.
        """
        if not issubclass(feature_cls, Features):
            raise TypeError(
                f"feature_cls must be Features or a subclass, got {feature_cls}"
            )
        # check that dict handles match tracking handles
        for handle, t in tracking_collection.tracking_dict.items():
            if handle != t.handle:
                raise ValueError(
                    f"Key '{handle}' does not match object's handle '{t.handle}'"
                )
        return cls(
            {
                handle: feature_cls(t)
                for handle, t in tracking_collection.tracking_dict.items()
            }
        )

    @classmethod
    def from_list(cls, features_list: list[Features]):
        """
        Create a FeaturesCollection from a list of Features objects, keyed by handle
        """
        handles = [obj.handle for obj in features_list]
        if len(handles) != len(set(handles)):
            raise Exception("handles must be unique")
        features_dict = {obj.handle: obj for obj in features_list}
        return cls(features_dict)

    def cluster_embedding(
        self,
        embedding_dict: dict[str, list[int]],
        n_clusters: int,
        random_state: int = 0,
    ):
        """
        Perform k-means clustering across all Features objects using the specified embedding.
        Returns a dictionary of label Series (one per Features, keyed by name) and the centroids DataFrame.
        """

        # Build embeddings and keep names in sync
        embedding_dfs = {
            name: f.embedding_df(embedding_dict)
            for name, f in self.features_dict.items()
        }
        # Check all embeddings have the same columns
        columns = next(iter(embedding_dfs.values())).columns
        if not all(df.columns.equals(columns) for df in embedding_dfs.values()):
            raise ValueError("All embeddings must have the same columns")

        # Concatenate with keys to create a MultiIndex
        combined = pd.concat(embedding_dfs.values(), axis=0, keys=embedding_dfs.keys())
        valid_mask = combined.notna().all(axis=1)
        valid_combined = combined[valid_mask]

        # Fit kmeans only on valid rows
        model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
            valid_combined
        )
        centroids = pd.DataFrame(model.cluster_centers_, columns=combined.columns)

        # Assign cluster labels: nan for rows with any nan, cluster for valid rows
        combined_labels = pd.Series(np.nan, index=combined.index, name="cluster")
        combined_labels.loc[valid_mask] = model.labels_

        # Split back to per-object labels using the first level of the MultiIndex
        labels_dict = {
            name: combined_labels.xs(name, level=0).astype("Int64")
            for name in embedding_dfs.keys()
        }

        return labels_dict, centroids

    def train_knn_regressor(
        self,
        embedding: dict[str, list[int]],
        target_embedding: dict[str, list[int]],
        n_neighbors: int = 5,
        **kwargs,
    ):
        """
        Train a kNN regressor to predict a target embedding from a feature embedding.
        Both embedding and target_embedding are dicts mapping column names to time shifts.
        Returns the trained model, the indices used for training, and the feature/target DataFrames.
        """
        train_embeds = self.embedding_df(embedding)
        target_embeds = self.embedding_df(target_embedding)

        # do leave one out training
        for i in range(len(train_embeds)):
            train_embeds_i = train_embeds.drop(i)
            target_embeds_i = target_embeds.drop(i)
            model, train_columns, target_columns = train_knn_from_embeddings(
                train_embeds_i, target_embeds_i, n_neighbors, **kwargs
            )
            predictions = predict_knn_on_embedding(
                model, train_embeds_i, target_columns
            )
            self.store(predictions, "knn_predictions", overwrite=True)
        model, train_columns, target_columns = train_knn_from_embeddings(
            train_embeds_i, target_embeds_i, n_neighbors, **kwargs
        )
        return model, train_columns, target_columns

    def plot(self, arg=None, figsize=(8, 2), show: bool = True, title: str = None):
        """
        Plot features for all collections in the MultipleFeaturesCollection.
        - If arg is a BatchResult or dict: treat as batch result and plot for each collection.
        - Otherwise: treat as column name(s) or None and plot for each collection.
        - If title is provided, it will be used as the overall title for the figure.
        """
        import matplotlib.pyplot as plt

        if arg is None:
            # Plot all columns for each Features object
            features_dict = {
                handle: obj.data for handle, obj in self.features_dict.items()
            }
            plot_type = "all"
        elif isinstance(arg, (str, list)):
            # Plot specified column(s) for each Features object
            if isinstance(arg, str):
                columns = [arg]
            else:
                columns = arg
            features_dict = {}
            for handle, obj in self.features_dict.items():
                # Only include columns that exist in this Features object
                cols = [col for col in columns if col in obj.data]
                if cols:
                    features_dict[handle] = obj.data[cols]
            plot_type = "columns"
        elif isinstance(arg, dict):
            # Batch result: plot each FeaturesResult
            features_dict = arg
            plot_type = "batch"
        else:
            raise TypeError(
                "Argument must be None, a string, a list of strings, or a batch result dict."
            )

        n = len(features_dict)
        if n == 0:
            raise ValueError("No features to plot.")
        fig, axes = plt.subplots(
            n, 1, figsize=(figsize[0], figsize[1] * n), sharex=True
        )
        if n == 1:
            axes = [axes]
        for ax, (handle, data) in zip(axes, features_dict.items()):
            if plot_type == "batch":
                # FeaturesResult: plot as a single series
                ax.plot(data.index, data.values, label=getattr(data, "name", "value"))
            else:
                # DataFrame: plot all columns or selected columns
                if isinstance(data, pd.Series):
                    ax.plot(data.index, data.values, label=data.name)
                else:
                    data.plot(ax=ax)
            ax.set_title(str(handle))
            ax.set_xlabel("frame")
            ax.legend()
        if title is not None:
            fig.suptitle(title, fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
        else:
            plt.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def store(
        self,
        results_dict: dict[str, FeaturesResult],
        name: str = None,
        meta: dict = None,
        overwrite: bool = False,
    ):
        """
        Store all FeaturesResult objects in a one-layer dict (as returned by batch methods).
        Example:
            results = features_collection.speed('nose')
            features_collection.store(results)
        """
        for v in results_dict.values():
            if hasattr(v, "store"):
                v.store(name=name, meta=meta, overwrite=overwrite)
            else:
                raise ValueError(f"{v} is not a FeaturesResult object")

    @property
    def loc(self):
        return _Indexer(self, self._loc)

    @property
    def iloc(self):
        return _Indexer(self, self._iloc)

    def _loc(self, idx):
        return self.__class__({k: v.loc[idx] for k, v in self.features_dict.items()})

    def _iloc(self, idx):
        return self.__class__({k: v.iloc[idx] for k, v in self.features_dict.items()})

    def __getitem__(self, key):
        """
        Get Features by handle (str), by integer index, or by slice.
        """
        if isinstance(key, int):
            handle = list(self.features_dict)[key]
            return self.features_dict[handle]
        elif isinstance(key, slice):
            handles = list(self.features_dict)[key]
            return self.__class__({h: self.features_dict[h] for h in handles})
        else:
            return self.features_dict[key]

    def __setitem__(self, key, value):
        """
        Set Features by handle (str).
        """
        if not isinstance(value, Features):
            raise TypeError(f"Value must be a Features, got {type(value).__name__}")
        warnings.warn(
            "Direct assignment to FeaturesCollection is deprecated and may be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.features_dict[key] = value

    def keys(self):
        """Return the keys of the features_dict."""
        return self.features_dict.keys()

    def values(self):
        """Return the values of the features_dict."""
        return self.features_dict.values()

    def items(self):
        """Return the items of the features_dict."""
        return self.features_dict.items()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.features_dict)} Features objects>"


class MultipleFeaturesCollection:
    """
    Collection of FeaturesCollection objects, keyed by name.
    """

    def __init__(self, features_collections: dict[str, FeaturesCollection]):
        self.features_collections = features_collections

    def __setitem__(self, key, value):
        """
        Set FeaturesCollection by handle (str).
        """
        if not isinstance(value, FeaturesCollection):
            raise TypeError(
                f"Value must be a FeaturesCollection, got {type(value).__name__}"
            )
        warnings.warn(
            "Direct assignment to MultipleFeaturesCollection is deprecated and may be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.features_collections[key] = value

    @classmethod
    def from_multiple_tracking_collection(
        cls,
        multiple_tracking_collection: MultipleTrackingCollection,
        feature_cls=Features,
    ):
        """
        Factory method to create MultipleFeaturesCollection from a MultipleTrackingCollection object.
        """
        collections = {}
        for (
            coll_name,
            tracking_collection,
        ) in multiple_tracking_collection.tracking_collections.items():
            collections[coll_name] = FeaturesCollection.from_tracking_collection(
                tracking_collection, feature_cls=feature_cls
            )
        return cls(collections)

    def __getattr__(self, name):
        def batch_method(*args, **kwargs):
            results = {}
            for coll_name, collection in self.features_collections.items():
                try:
                    results[coll_name] = getattr(collection, name)(*args, **kwargs)
                except Exception as e:
                    raise BatchProcessError(
                        collection_name=coll_name,
                        object_name=getattr(e, "object_name", None),
                        method=getattr(e, "method", None),
                        original_exception=getattr(e, "original_exception", e),
                    ) from e
            return BatchResult(results, self)

        return batch_method

    def store(
        self,
        results_dict: dict[str, dict[str, FeaturesResult]],
        name: str = None,
        meta: dict = None,
        overwrite: bool = False,
    ):
        """
        Store all FeaturesResult objects in a two-layer dict (as returned by batch methods).
        Example:
            results = multiple_features_collection.speed('nose')
            multiple_features_collection.store(results)
        """
        for group_dict in results_dict.values():
            for v in group_dict.values():
                if hasattr(v, "store"):
                    v.store(name=name, meta=meta, overwrite=overwrite)
                else:
                    raise ValueError(f"{v} is not a FeaturesResult object")

    def plot(self, arg=None, figsize=(8, 2), show=True):
        """
        Plot features for all collections in the MultipleFeaturesCollection.
        - If arg is a BatchResult or dict: treat as batch result and plot for each collection.
        - Otherwise: treat as column name(s) or None and plot for each collection.
        """
        figs_axes = {}
        # If arg is a BatchResult or dict, treat as batch result
        if isinstance(arg, dict):
            for coll_name, group_dict in arg.items():
                if coll_name in self.features_collections:
                    figs_axes[coll_name] = self.features_collections[coll_name].plot(
                        group_dict, figsize=figsize, show=show, title=coll_name
                    )
        else:
            for coll_name, collection in self.features_collections.items():
                figs_axes[coll_name] = collection.plot(
                    arg, figsize=figsize, show=show, title=coll_name
                )
        return figs_axes

    def cluster_embedding(
        self,
        embedding_dict: dict[str, list[int]],
        n_clusters: int,
        random_state: int = 0,
    ):
        # Step 1: Build all embeddings
        all_embeddings = {}
        for coll_name, collection in self.features_collections.items():
            for feat_name, features in collection.features_dict.items():
                embed_df = features.embedding_df(embedding_dict)
                all_embeddings[(coll_name, feat_name)] = embed_df

        # Step 2: Concatenate
        combined = pd.concat(
            all_embeddings.values(),
            keys=all_embeddings.keys(),
            names=["collection", "feature", "frame"],
        )

        # Step 3: Mask
        valid_mask = combined.notna().all(axis=1)
        valid_combined = combined[valid_mask]

        # Step 4: Cluster
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
            valid_combined
        )
        centroids = pd.DataFrame(model.cluster_centers_, columns=combined.columns)

        # Step 5: Assign labels
        combined_labels = pd.Series(np.nan, index=combined.index, name="cluster")
        combined_labels.loc[valid_mask] = model.labels_

        # Step 6: Split
        nested_labels = {}
        for (coll_name, feat_name), _ in all_embeddings.items():
            idx = (coll_name, feat_name)
            # Get all rows for this (collection, feature)
            labels = combined_labels.xs(idx, level=["collection", "feature"])
            if coll_name not in nested_labels:
                nested_labels[coll_name] = {}
            nested_labels[coll_name][feat_name] = labels.astype("Int64")

        # Step 7: Return
        return nested_labels, centroids

    def knn_cross_predict_rms_matrix(
        self,
        source_embedding: dict[str, list[int]],
        target_embedding: dict[str, list[int]],
        n_neighbors: int = 5,
        normalize_source: bool = False,
        normalize_pred: dict | str = None,
        collections: list[str] = None,
    ):
        """
        For each Features object in each selected FeaturesCollection in the MultipleFeaturesCollection,
        train a kNN regressor and use it to predict the target embedding in every Features object
        in each selected FeaturesCollection, storing the full RMS error Series in a DataFrame for each collection pair.
        The same set of collections is used for both source and target.
        Returns a dict of DataFrames: { "from<source>_to_<target>": DataFrame }.
        """
        results = {}
        # Determine which collections to use
        all_keys = list(self.features_collections.keys())
        if collections is None:
            collections = all_keys

        for source_coll_name in collections:
            source_coll = self.features_collections[source_coll_name]
            for target_coll_name in collections:
                target_coll = self.features_collections[target_coll_name]
                df = pd.DataFrame(
                    index=source_coll.features_dict.keys(),
                    columns=target_coll.features_dict.keys(),
                    dtype=object,
                )
                for source_feat_name, source_feat in source_coll.features_dict.items():
                    # Train regressor
                    if normalize_source:
                        model, in_cols, out_cols, rescale_factors = (
                            source_feat.train_knn_regressor(
                                source_embedding=source_embedding,
                                target_embedding=target_embedding,
                                n_neighbors=n_neighbors,
                                normalize_source=True,
                            )
                        )
                    else:
                        model, in_cols, out_cols = source_feat.train_knn_regressor(
                            source_embedding=source_embedding,
                            target_embedding=target_embedding,
                            n_neighbors=n_neighbors,
                        )
                        rescale_factors = None

                    for (
                        target_feat_name,
                        target_feat,
                    ) in target_coll.features_dict.items():
                        preds = target_feat.predict_knn(
                            model,
                            source_embedding=source_embedding,
                            target_embedding=target_embedding,
                            rescale_factors=rescale_factors,
                        )
                        ground_truth = target_feat.embedding_df(target_embedding)
                        rms = Features.rms_error_between_embeddings(
                            ground_truth, preds, rescale=normalize_pred
                        )
                        df.at[source_feat_name, target_feat_name] = rms
                key = f"from{source_coll_name}_to_{target_coll_name}"
                results[key] = df
        return results

    def cross_predict_rms(
        self,
        source_embedding: dict[str, list[int]],
        target_embedding: dict[str, list[int]],
        n_neighbors: int = 5,
        normalize_source: bool = False,
        normalize_pred: dict | str = None,
        set1: list[str] = None,
        set2: list[str] = None,
    ):
        """
        Performs two types of cross-prediction:
        1. Within-collection leave-one-out: For each Features object in each collection in set1 or set2 (union), trains a kNN regressor on all other Features objects in the same collection, predicts on the left-out object, and stores the RMS error Series.
        2. Between-collection: For each ordered pair of collections (A, B) with A in set1, B in set2, and A != B, trains a kNN regressor on all Features objects in A, predicts on all Features objects in B, and stores the RMS error Series for each Features object in B.

        Args:
            source_embedding: dict mapping feature names to time shifts for input embedding.
            target_embedding: dict mapping feature names to time shifts for target embedding.
            n_neighbors: Number of neighbors for kNN.
            normalize_source: Whether to normalize the source embedding during training.
            normalize_pred: Normalization for RMS calculation ('auto', dict, or None).
            set1: List of collection keys for the first set (default: all).
            set2: List of collection keys for the second set (default: all).

        Returns:
            dict with keys:
                'within': {collection: {feature_name: rms_series}}
                'between': {fromA_to_B: {target_feature_name: rms_series}}
        """
        results = {"within": {}, "between": {}}
        all_keys = list(self.features_collections.keys())
        if set1 is None:
            set1 = all_keys
        if set2 is None:
            set2 = all_keys
        # Union for within
        within_collections = sorted(set(set1) | set(set2))

        # Within-collection leave-one-out
        for coll_name in within_collections:
            coll = self.features_collections[coll_name]
            rms_dict = {}
            for left_out_name, left_out_feat in coll.features_dict.items():
                # Train on all others
                train_feats = [
                    f for n, f in coll.features_dict.items() if n != left_out_name
                ]
                train_embeds = [f.embedding_df(source_embedding) for f in train_feats]
                target_embeds = [f.embedding_df(target_embedding) for f in train_feats]
                if normalize_source:
                    # Use normalization from the training set
                    train_embeds_norm, rescale_factors = normalize_df(
                        pd.concat(train_embeds)
                    )
                    lengths = [len(e) for e in train_embeds]
                    starts = np.cumsum([0] + lengths[:-1])
                    train_embeds_norm_list = [
                        train_embeds_norm.iloc[start : start + length]
                        for start, length in zip(starts, lengths)
                    ]
                    model, in_cols, out_cols = train_knn_from_embeddings(
                        train_embeds_norm_list, target_embeds, n_neighbors
                    )
                else:
                    model, in_cols, out_cols = train_knn_from_embeddings(
                        train_embeds, target_embeds, n_neighbors
                    )
                    rescale_factors = None
                # Predict on left-out
                if normalize_source and rescale_factors is not None:
                    test_embed = left_out_feat.embedding_df(source_embedding)
                    test_embed = apply_normalization_to_df(test_embed, rescale_factors)
                else:
                    test_embed = left_out_feat.embedding_df(source_embedding)
                target_embed = left_out_feat.embedding_df(target_embedding)
                preds = predict_knn_on_embedding(
                    model, test_embed, target_embed.columns
                )
                preds = preds.reindex(
                    index=target_embed.index, columns=target_embed.columns
                )
                rms = Features.rms_error_between_embeddings(
                    target_embed, preds, rescale=normalize_pred
                )
                rms_dict[left_out_name] = rms
            results["within"][coll_name] = rms_dict

        # Between-collection: all ordered pairs (A, B) with A in set1, B in set2, and A != B
        for coll1 in set1:
            for coll2 in set2:
                if coll1 == coll2:
                    continue
                source_coll = self.features_collections[coll1]
                target_coll = self.features_collections[coll2]
                # Train on all in coll1
                train_embeds = [
                    f.embedding_df(source_embedding)
                    for f in source_coll.features_dict.values()
                ]
                target_embeds = [
                    f.embedding_df(target_embedding)
                    for f in source_coll.features_dict.values()
                ]
                if normalize_source:
                    train_embeds_norm, rescale_factors = normalize_df(
                        pd.concat(train_embeds)
                    )
                    lengths = [len(e) for e in train_embeds]
                    starts = np.cumsum([0] + lengths[:-1])
                    train_embeds_norm_list = [
                        train_embeds_norm.iloc[start : start + length]
                        for start, length in zip(starts, lengths)
                    ]
                    model, in_cols, out_cols = train_knn_from_embeddings(
                        train_embeds_norm_list, target_embeds, n_neighbors
                    )
                else:
                    model, in_cols, out_cols = train_knn_from_embeddings(
                        train_embeds, target_embeds, n_neighbors
                    )
                    rescale_factors = None
                # Predict on all in coll2
                rms_dict = {}
                for target_feat_name, target_feat in target_coll.features_dict.items():
                    if normalize_source and rescale_factors is not None:
                        test_embed = target_feat.embedding_df(source_embedding)
                        test_embed = apply_normalization_to_df(
                            test_embed, rescale_factors
                        )
                    else:
                        test_embed = target_feat.embedding_df(source_embedding)
                    target_embed = target_feat.embedding_df(target_embedding)
                    preds = predict_knn_on_embedding(
                        model, test_embed, target_embed.columns
                    )
                    preds = preds.reindex(
                        index=target_embed.index, columns=target_embed.columns
                    )
                    rms = Features.rms_error_between_embeddings(
                        target_embed, preds, rescale=normalize_pred
                    )
                    rms_dict[target_feat_name] = rms
                key = f"from{coll1}_to_{coll2}"
                results["between"][key] = rms_dict
        # Also do all ordered pairs (A, B) with A in set2, B in set1, and A != B
        for coll1 in set2:
            for coll2 in set1:
                if coll1 == coll2:
                    continue
                source_coll = self.features_collections[coll1]
                target_coll = self.features_collections[coll2]
                # Train on all in coll1
                train_embeds = [
                    f.embedding_df(source_embedding)
                    for f in source_coll.features_dict.values()
                ]
                target_embeds = [
                    f.embedding_df(target_embedding)
                    for f in source_coll.features_dict.values()
                ]
                if normalize_source:
                    train_embeds_norm, rescale_factors = normalize_df(
                        pd.concat(train_embeds)
                    )
                    lengths = [len(e) for e in train_embeds]
                    starts = np.cumsum([0] + lengths[:-1])
                    train_embeds_norm_list = [
                        train_embeds_norm.iloc[start : start + length]
                        for start, length in zip(starts, lengths)
                    ]
                    model, in_cols, out_cols = train_knn_from_embeddings(
                        train_embeds_norm_list, target_embeds, n_neighbors
                    )
                else:
                    model, in_cols, out_cols = train_knn_from_embeddings(
                        train_embeds, target_embeds, n_neighbors
                    )
                    rescale_factors = None
                # Predict on all in coll2
                rms_dict = {}
                for target_feat_name, target_feat in target_coll.features_dict.items():
                    if normalize_source and rescale_factors is not None:
                        test_embed = target_feat.embedding_df(source_embedding)
                        test_embed = apply_normalization_to_df(
                            test_embed, rescale_factors
                        )
                    else:
                        test_embed = target_feat.embedding_df(source_embedding)
                    target_embed = target_feat.embedding_df(target_embedding)
                    preds = predict_knn_on_embedding(
                        model, test_embed, target_embed.columns
                    )
                    preds = preds.reindex(
                        index=target_embed.index, columns=target_embed.columns
                    )
                    rms = Features.rms_error_between_embeddings(
                        target_embed, preds, rescale=normalize_pred
                    )
                    rms_dict[target_feat_name] = rms
                key = f"from{coll1}_to_{coll2}"
                results["between"][key] = rms_dict
        return results

    @staticmethod
    def plot_cross_predict_results(
        results,
        within_keys=None,
        between_keys=None,
        plot_type="bar",  # 'bar', 'point', or 'violin'
        figsize=(10, 6),
        show=True,
    ):
        """
        Plot summary statistics from cross_predict_rms_leaveoneout_and_between results.

        Args:
            results: dict as returned by cross_predict_rms_leaveoneout_and_between
            within_keys: list of collection names to include from 'within'
            between_keys: list of between keys (e.g. 'fromA_to_B') to include from 'between'
            plot_type: 'bar', 'point', or 'violin'
            figsize: tuple for figure size
            show: whether to call plt.show()
        """
        # Gather data
        records = []
        # Within
        if within_keys is not None:
            for coll in within_keys:
                for feat, series in results["within"].get(coll, {}).items():
                    arr = series.dropna().values
                    for v in arr:
                        records.append(
                            {"Category": f"within_{coll}", "Feature": feat, "RMS": v}
                        )
        # Between
        if between_keys is not None:
            for comp in between_keys:
                for feat, series in results["between"].get(comp, {}).items():
                    arr = series.dropna().values
                    for v in arr:
                        records.append({"Category": comp, "Feature": feat, "RMS": v})

        df = pd.DataFrame(records)

        plt.figure(figsize=figsize)
        if plot_type == "bar":
            # Bar plot: mean of means per category
            means = df.groupby("Category").RMS.mean()
            means.plot(kind="bar", yerr=df.groupby("Category").RMS.std(), capsize=4)
            plt.ylabel("Mean RMS (mean of means)")
            plt.title("RMS prediction error by category")
        elif plot_type == "point":
            # Point plot: mean RMS per feature, grouped by category
            means = df.groupby(["Category", "Feature"]).RMS.mean().reset_index()
            sns.pointplot(data=means, x="Feature", y="RMS", hue="Category", dodge=True)
            plt.ylabel("mean RMS error")
            plt.title(f"{within_keys[0]} vs {within_keys[1]}")
            plt.xticks(rotation=90)
        elif plot_type == "violin":
            # Violin plot: all raw RMS values
            sns.violinplot(data=df, x="Category", y="RMS", inner="point")
            plt.ylabel("RMS")
            plt.title("RMS prediction error by category")
        else:
            raise ValueError("plot_type must be 'bar', 'point', or 'violin'")

        plt.tight_layout()
        if show:
            plt.show()
        return df  # Return the DataFrame for further inspection if needed

    @staticmethod
    def dumbbell_plot_cross_predict(
        results, within_key, between_key, figsize=(3, 3), show=True
    ):
        """
        Plot a vertical dumbbell plot: x-axis is category (Within/Between), y-axis is RMS,
        each feature is a line connecting its within and between mean RMS.

        Args:
            results: dict as returned by cross_predict_rms_leaveoneout_and_between
            within_key: collection name for 'within' (e.g., 'POD14')
            between_key: key for 'between' (e.g., 'fromX_to_POD14')
            figsize: tuple for figure size
            show: whether to call plt.show()
        """
        features = sorted(
            set(
                list(results["within"].get(within_key, {}).keys())
                + list(results["between"].get(between_key, {}).keys())
            )
        )
        data = []
        for feat in features:
            mean_within = (
                results["within"]
                .get(within_key, {})
                .get(feat, pd.Series(dtype=float))
                .mean()
            )
            mean_between = (
                results["between"]
                .get(between_key, {})
                .get(feat, pd.Series(dtype=float))
                .mean()
            )
            data.append(
                {"Feature": feat, "Within": mean_within, "Between": mean_between}
            )
        df = pd.DataFrame(data)

        # Prepare for plotting
        x = [0, 1]  # 0 = Within, 1 = Between
        plt.figure(figsize=figsize)
        for i, row in df.iterrows():
            plt.plot(x, [row["Within"], row["Between"]], color="gray", lw=2, zorder=1)
            plt.scatter(
                x, [row["Within"], row["Between"]], s=60, color="black", zorder=2
            )
            # plt.text(-0.05, row['Within'], row['Feature'], va='center', ha='right', fontsize=9)
        plt.xticks(x, ["Within", "Between"])
        plt.ylabel("Mean RMS")
        plt.title(f"Dumbbell Plot: {within_key} vs {between_key}")
        plt.tight_layout()
        if show:
            plt.show()
        return df  # Return the DataFrame for further inspection if needed

    @property
    def loc(self):
        return _Indexer(self, self._loc)

    @property
    def iloc(self):
        return _Indexer(self, self._iloc)

    def _loc(self, idx):
        return self.__class__(
            {k: v.loc[idx] for k, v in self.features_collections.items()}
        )

    def _iloc(self, idx):
        return self.__class__(
            {k: v.iloc[idx] for k, v in self.features_collections.items()}
        )

    def __getitem__(self, key):
        """
        Get FeaturesCollection by handle (str), by integer index, or by slice.
        """
        if isinstance(key, int):
            handle = list(self.features_collections)[key]
            return self.features_collections[handle]
        elif isinstance(key, slice):
            handles = list(self.features_collections)[key]
            return self.__class__({h: self.features_collections[h] for h in handles})
        else:
            return self.features_collections[key]

    def keys(self):
        """Return the keys of the features_collections."""
        return self.features_collections.keys()

    def values(self):
        """Return the values of the features_collections."""
        return self.features_collections.values()

    def items(self):
        """Return the items of the features_collections."""
        return self.features_collections.items()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.features_collections)} FeaturesCollection objects>"
