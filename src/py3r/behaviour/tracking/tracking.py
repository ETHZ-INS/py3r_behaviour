from __future__ import annotations
import copy
import re
import warnings
from typing import Dict, Any, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from py3r.behaviour.util.collection_utils import _Indexer
from py3r.behaviour.util.smoothing import apply_smoothing
from py3r.behaviour.util.io_utils import (
    SchemaVersion,
    begin_save,
    write_manifest,
    read_manifest,
    write_dataframe,
    read_dataframe,
)

Self = TypeVar("Self", bound="Tracking")


class Tracking:
    """
    Represent frame-by-frame tracked keypoints with convenience loaders and tools.

    A `Tracking` holds a pandas DataFrame of columns like `p1.x`, `p1.y`,
    `p1.z`, `p1.likelihood` with index named `frame`. Most users will create
    objects via factory methods and then call instance methods to process or
    analyze trajectories.

    Quick start with realistic CSVs stored in the package data:

    - Load from DLC CSV
    - Load from DLC multi-animal CSV
    - Load from YOLO3R CSV
    - Inspect points, distances
    - Filter, interpolate, smooth
    - Rescale by known distance, trim, check time
    - Save and slice (`loc` / `iloc`)
    - Minimal plotting

    Examples:
        Create test DLC CSV and load a Tracking:

        >>> from importlib import resources as ir
        >>> from py3r.behaviour.tracking import _data
        >>> p = ir.files(_data) / 'dlc_single.csv'
        >>> t = Tracking.from_dlc(str(p), handle='ex', options=LoadOptions(fps=30))
        >>> len(t.data), t.meta['fps'], t.handle
        (5, 30.0, 'ex')
        >>> t.data[['p1.x','p1.y','p1.z','p1.likelihood']].head(2).reset_index().values.tolist()
        [[0, 0.0, 0.0, 0.0, 1.0], [1, 1.0, 2.0, 3.0, 0.75]]

        Load from DLC multi-animal (DLCMA):

        >>> p_ma = ir.files(_data) / 'dlcma_multi.csv'
        >>> tma = Tracking.from_dlcma(str(p_ma), handle='ma', options=LoadOptions(fps=30))
        >>> tma.meta['fps'], tma.handle
        (30.0, 'ma')

        Load from YOLO3R (3D columns present):

        >>> p_y = ir.files(_data) / 'yolo3r.csv'
        >>> ty = Tracking.from_yolo3r(str(p_y), handle='y3r', options=LoadOptions(fps=30))
        >>> 'p1.z' in ty.data.columns and 'p1.likelihood' in ty.data.columns
        True
        >>> ty.data[['p1.x','p1.y','p1.z','p1.likelihood']].head(2).reset_index().values.tolist()
        [[0, 0.0, 0.0, 0.0, 1.0], [1, 1.0, 2.0, 3.0, 0.9]]

        Inspect points and distances:

        >>> names = t.get_point_names()
        >>> sorted(names)[:3]
        ['p1', 'p2', 'p3']
        >>> d = t.distance_between('p1', 'p2')
        >>> len(d) == len(t.data)
        True

        Filter low-likelihood positions and interpolate:

        >>> t2 = Tracking.from_dlc(str(p), handle='ex2', options=LoadOptions(fps=30))
        >>> _ = t2.filter_likelihood(0.2)
        >>> import numpy as np
        >>> np.isnan(t2.data['p1.x']).any()
        True
        >>> _ = t2.interpolate(method='nearest', limit=1)
        >>> t2.data.columns.str.endswith('.likelihood').any() and t2.meta['interpolation']['method'] == 'nearest'
        True

        Smooth all points with window=1 (no-op) and check metadata:

        >>> sm = t.generate_smoothdict([t.get_point_names()], [1], ['mean'])
        >>> _ = t.smooth(sm)
        >>> 'smoothing' in t.meta
        True

        Rescale by known distance between two points (uniform across dims):

        >>> _ = t.rescale_by_known_distance('p1', 'p2', distance_in_metres=2.0)
        >>> t.meta['distance_units']
        'm'

        Trim frames and verify time window:

        >>> t3 = Tracking.from_dlc(str(p), handle='ex3', options=LoadOptions(fps=30))
        >>> _ = t3.trim(startframe=2, endframe=7)
        >>> t3.data.index[0] == 2 and t3.data.index[-1] == 7
        True
        >>> t3.time_as_expected(mintime=0.0, maxtime=10.0)
        True

        Save to disk (CSV and _meta.json):

        >>> with tempfile.TemporaryDirectory() as d:
        ...     out = os.path.join(d, 'out.csv')
        ...     _ = t.save(out)
        ...     os.path.exists(out) and os.path.exists(out.replace('.csv', '_meta.json'))
        True

        Slice with loc/iloc and keep handle:

        >>> t4 = Tracking.from_dlc(str(p), handle='ex4', options=LoadOptions(fps=30))
        >>> t4s = t4.loc[0:3]
        >>> isinstance(t4s, Tracking) and t4s.handle == 'ex4'
        True
        >>> t4s2 = t4.iloc[0:2]
        >>> isinstance(t4s2, Tracking) and len(t4s2.data) == 2
        True

        Minimal plotting (no display):

        >>> _ = t.plot(show=False)

        Tagging and user metadata:

        >>> t.add_tag('session', 'S1')
        >>> t.tags['session']
        'S1'
        >>> t.add_usermeta({'group': 'G1'}, overwrite=True)
        >>> t.meta['usermeta']['group']
        'G1'
    """

    data: pd.DataFrame
    meta: dict
    handle: str
    tags: dict[str, str]

    @classmethod
    def from_dlc(
        cls: Type[Self],
        filepath: str,
        *,
        handle: str,
        fps: float,
        aspectratio_correction: float = 1.0,
        tags: dict[str, str] | None = None,
    ) -> Self:
        """
        loads a Tracking object from a (single animal) deeplabcut tracking csv
        """
        # read header
        header = pd.read_csv(filepath, header=None, nrows=3)
        cols = [
            ".".join(i) for i in zip(list(header.iloc[1, 1:]), list(header.iloc[2, 1:]))
        ]
        scorer = header.iloc[0, 1]

        # setup data
        data = pd.read_csv(filepath, skiprows=3, header=None)
        data.set_index(0, inplace=True)
        data.index.rename("frame", inplace=True)
        data.columns = cols

        meta = {
            "filepath": filepath,
            "fps": float(fps),
            "aspectratio_correction": float(aspectratio_correction),
            "network": scorer,
        }

        data = cls._apply_aspectratio_correction(data, float(aspectratio_correction))

        return cls(data, meta, handle, tags)

    @classmethod
    def from_dlcma(
        cls: Type[Self],
        filepath: str,
        *,
        handle: str,
        fps: float,
        aspectratio_correction: float = 1.0,
        tags: dict[str, str] | None = None,
    ) -> Self:
        """
        loads a Tracking object from a multi-animal deeplabcut tracking csv
        """
        # read header
        header = pd.read_csv(filepath, header=None, nrows=4)
        cols = [
            ".".join(i)
            for i in zip(
                list(header.iloc[1, 1:]),
                list(header.iloc[2, 1:]),
                list(header.iloc[3, 1:]),
            )
        ]
        scorer = header.iloc[0, 1]

        # setup data
        data = pd.read_csv(filepath, skiprows=4, header=None)
        data.set_index(0, inplace=True)
        data.index.rename("frame", inplace=True)
        data.columns = cols

        # add meta specific to DLC
        meta = {
            "filepath": filepath,
            "fps": float(fps),
            "aspectratio_correction": float(aspectratio_correction),
            "network": scorer,
        }

        data = cls._apply_aspectratio_correction(data, float(aspectratio_correction))

        return cls(data, meta, handle, tags)

    @classmethod
    def from_yolo3r(
        cls: Type[Self],
        filepath: str,
        *,
        handle: str,
        fps: float,
        aspectratio_correction: float = 1.0,
        tags: dict[str, str] | None = None,
    ) -> Self:
        """
        loads a Tracking object from a single- or multi-animal yolo csv in 3R hub format
        """
        # setup data
        data = pd.read_csv(filepath, index_col="frame_index")
        data.index.rename("frame", inplace=True)
        newcols = [re.sub(".conf$", ".likelihood", col) for col in data.columns]
        data.columns = newcols

        # drop bounding-box-related columns
        # assumes bbox column names have 3 dot delimited sections
        for col in data.columns:
            if len(col.split(".")) == 3:
                data.drop(columns=col, inplace=True)
            if col.split(".")[-2] == "max_dim":
                data.drop(columns=col, inplace=True)

        meta = {
            "filepath": filepath,
            "fps": float(fps),
            "aspectratio_correction": float(aspectratio_correction),
        }

        data = cls._apply_aspectratio_correction(data, float(aspectratio_correction))

        return cls(data, meta, handle, tags)

    @staticmethod
    def _apply_aspectratio_correction(
        df: pd.DataFrame, correction: float
    ) -> pd.DataFrame:
        """
        rescales all x values within tracking object by aspectratio correction factor
        """
        if correction == 1.0:
            return df

        # adjust dataframe
        tracked_points = list(set([".".join(i.split(".")[0:-1]) for i in df.columns]))
        df_corrected = df.copy()
        for point in tracked_points:
            df_corrected[point + ".x"] = df_corrected[point + ".x"] * correction
        return df_corrected

    def __init__(
        self,
        data: pd.DataFrame,
        meta: Dict[str, Any],
        handle: str,
        tags: dict[str, str] = None,
    ) -> None:
        if not isinstance(meta, dict):
            raise TypeError(f"meta must be a dictionary, got {type(meta).__name__}")
        if "fps" not in meta:
            raise ValueError("meta dictionary must contain 'fps' key")
        self.data = data
        self.meta = meta
        self.handle = handle
        self.tags = tags if tags is not None else {}

    # ----------- Instance methods -----------

    def add_usermeta(self, usermeta: dict, overwrite: bool = False) -> None:
        """
        adds or updates user-defined metadata
        """
        if not isinstance(usermeta, dict):
            raise TypeError(
                f"usermeta must be a dictionary, got {type(usermeta).__name__}"
            )

        if "usermeta" in self.meta and not overwrite:
            raise Exception(
                "user defined metadata already stored, set overwrite=True to overwrite"
            )

        self.meta["usermeta"] = usermeta
        if overwrite:
            warnings.warn("usermeta may be overwritten")

    def add_tag(self, tagname: str, tagvalue: str, overwrite: bool = False) -> None:
        """
        adds or updates a tag
        """
        if not isinstance(tagname, str):
            raise TypeError(f"tagname must be a string, got {type(tagname).__name__}")
        if tagname in self.tags and not overwrite:
            raise Exception(
                f"tag {tagname} already exists, set overwrite=True to overwrite"
            )
        self.tags[tagname] = tagvalue

    # New round-trip save/load that preserves full state in a directory
    def save(
        self,
        dirpath: str,
        *,
        data_format: str = "parquet",
        overwrite: bool = False,
    ) -> None:
        """
        Save this Tracking into a self-describing directory for exact round-trip.
        """
        target = begin_save(dirpath, overwrite)
        # write data
        data_spec = write_dataframe(
            target,
            self.data,
            filename="data.parquet" if data_format == "parquet" else "data.csv",
            format=data_format,
        )
        # write manifest
        manifest = {
            "schema_version": SchemaVersion,
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
            "handle": self.handle,
            "tags": self.tags,
            "meta": self.meta,
            "data": data_spec,
        }
        write_manifest(target, manifest)

    @classmethod
    def load(cls: Type[Self], dirpath: str) -> Self:
        """
        Load a Tracking (or subclass) previously saved with save().
        """
        manifest = read_manifest(dirpath)
        df = read_dataframe(dirpath, manifest["data"])
        handle = manifest["handle"]
        meta = manifest["meta"]
        tags = manifest.get("tags", {})
        return cls(df, meta, handle, tags)

    def strip_column_names(self) -> None:
        """strips out all column name string apart from last two sections delimited by dots"""
        stripped_colnames = [".".join(col.split(".")[-2:]) for col in self.data.columns]
        self.data.columns = stripped_colnames

    def time_as_expected(self, mintime: float, maxtime: float) -> bool:
        """
        checks that the total length of the tracking data is between mintime seconds and maxtime seconds
        """
        if "trim" in self.meta.keys():
            warnings.warn("tracking data have been trimmed")
        totalframes = self.data.index[-1] - self.data.index[0]
        totaltime = totalframes / self.meta["fps"]

        return (mintime <= totaltime) & (maxtime >= totaltime)

    def trim(self, startframe: int | None = None, endframe: int | None = None) -> None:
        """
        trims the tracking data object between startframe and endframe
        """
        if startframe is not None:
            if (self.data.index[0] > startframe) or (self.data.index[-1] < startframe):
                raise Exception("startframe not in data")
        if endframe is not None:
            if (self.data.index[0] > endframe) or (self.data.index[-1] < endframe):
                raise Exception("endframe not in data")

        datatrim = self.data.loc[startframe:endframe, :].copy()
        self.data = datatrim

        self.meta["trim"] = {"startframe": startframe, "endframe": endframe}

    def filter_likelihood(self, threshold: float) -> None:
        """sets all tracking position values with likelihood less than threshold to np.nan"""
        if "filter_likelihood_threshold" in self.meta.keys():
            raise Exception(
                "likelihood already filtered. re-load the raw data to change filter."
            )
        if "smoothing" in self.meta.keys():
            warnings.warn(
                "these data have been smoothed. you should filter likelihood before smoothing"
            )

        for point in self.get_point_names():
            self.data.loc[
                (self.data[point + ".likelihood"] <= threshold), point + ".x"
            ] = np.nan
            self.data.loc[
                (self.data[point + ".likelihood"] <= threshold), point + ".y"
            ] = np.nan
            if point + ".z" in self.data.columns:
                self.data.loc[
                    (self.data[point + ".likelihood"] <= threshold), point + ".z"
                ] = np.nan

        self.meta["filter_likelihood_threshold"] = threshold

    def distance_between(self, point1: str, point2: str, dims=("x", "y")) -> pd.Series:
        """framewise distance between two points"""
        distance = np.sqrt(
            sum(
                [
                    (self.data[point1 + "." + dim] - self.data[point2 + "." + dim]) ** 2
                    for dim in dims
                ]
            )
        )
        return distance

    def get_point_names(self) -> list:
        """list of tracked point names"""
        tracked_points = list(
            set([".".join(i.split(".")[:-1]) for i in self.data.columns])
        )
        return tracked_points

    def rescale_by_known_distance(
        self, point1: str, point2: str, distance_in_metres: float, dims=("x", "y")
    ) -> None:
        """rescale all dims by known distance between two points"""
        if "rescale_distance_method" in self.meta.keys():
            if self.meta["rescale_distance_method"] == "two_point_scalar_uniform":
                if any(d in self.meta["rescale_factor"].keys() for d in dims):
                    raise Exception(
                        "distance already rescaled in this dim. re-load the raw data to change scaling"
                    )
            else:
                raise Exception(
                    "distance already rescaled. re-load the raw data to change scaling"
                )

        tracking_distance = self.distance_between(point1, point2, dims=dims).median()
        rescale_factor = distance_in_metres / tracking_distance

        tracked_points = self.get_point_names()

        for point in tracked_points:
            for dim in dims:
                self.data[point + "." + dim] = (
                    self.data[point + "." + dim] * rescale_factor
                )

        self.meta["rescale_distance_method"] = "two_point_scalar_uniform"
        self.meta["rescale_factor"] = {dim: rescale_factor for dim in dims}
        self.meta["distance_units"] = "m"

    def _generate_partial_smoothdict(
        self, points: list, window: int, smoothtype: str
    ) -> dict:
        """make partial smoothdict for points"""
        smoothdict = dict()
        for key in points:
            smoothdict[key] = {"window": window, "type": smoothtype}
        return smoothdict

    def generate_smoothdict(
        self, pointslists: list, windows: list, smoothtypes: list
    ) -> dict:
        """make smoothdict for multiple point lists"""
        # deprecation warning
        warnings.warn(
            "generate_smoothdict is deprecated. use smooth_all instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        assert len(pointslists) == len(windows)
        assert len(pointslists) == len(smoothtypes)

        smoothdict = dict()
        for i in range(len(pointslists)):
            partial = self._generate_partial_smoothdict(
                pointslists[i], windows[i], smoothtypes[i]
            )
            if len(set(smoothdict.keys()).intersection(set(partial.keys()))) > 0:
                raise Exception("duplicate points detected")
            smoothdict = {**smoothdict, **partial}
        return smoothdict

    def smooth(self, smoothing_params: dict) -> None:
        """
        runs rolling mean or median filter of specified window length over specified
        points. All points within the tracking data must be specified, even if the rolling
        window has length 1
        smooth_dict has format
        {pointname:{window:windowlength,type:smoothtype}}
        where windowlength:int and smoothtype:str in {'mean','median'}
        """
        # deprecation warning
        warnings.warn(
            "smooth is deprecated. use smooth_all instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if "smoothing" in self.meta.keys():
            raise Exception(
                "data already smoothed. load again to use different smoothing"
            )

        all_points = self.get_point_names()

        if len(set(all_points).difference(smoothing_params.keys())) > 0:
            raise Exception("all tracked points must be specified for smoothing")

        for point in smoothing_params.keys():
            if smoothing_params[point]["type"] == "mean":
                self.data[point + ".x"] = (
                    self.data[point + ".x"]
                    .rolling(window=smoothing_params[point]["window"])
                    .mean()
                )
                self.data[point + ".y"] = (
                    self.data[point + ".y"]
                    .rolling(window=smoothing_params[point]["window"])
                    .mean()
                )
                if point + ".z" in self.data.columns:
                    self.data[point + ".z"] = (
                        self.data[point + ".z"]
                        .rolling(window=smoothing_params[point]["window"])
                        .mean()
                    )
            if smoothing_params[point]["type"] == "median":
                self.data[point + ".x"] = (
                    self.data[point + ".x"]
                    .rolling(window=smoothing_params[point]["window"])
                    .median()
                )
                self.data[point + ".y"] = (
                    self.data[point + ".y"]
                    .rolling(window=smoothing_params[point]["window"])
                    .median()
                )
                if point + ".z" in self.data.columns:
                    self.data[point + ".z"] = (
                        self.data[point + ".z"]
                        .rolling(window=smoothing_params[point]["window"])
                        .median()
                    )

        self.meta["smoothing"] = smoothing_params

    def smooth_all(
        self,
        window: int | None = 3,
        method: str = "mean",
        *overrides: tuple[list[str] | tuple[str, ...] | str, str, int | None],
        dims: tuple[str, ...] = ("x", "y"),
        strict: bool = False,
        inplace: bool = True,
        smoother=None,
        smoother_kwargs: dict | None = None,
    ) -> "Tracking | None":
        """
        Smooth all tracked points using a default method/window, with optional override groups.

        - window/method: default applied to any point without override
        - overrides: zero or more tuples of (points, method, window), where
            - points: list/tuple of point names (or a single str)
            - method: 'median' or 'mean'
            - window: int (or None to skip smoothing for those points)
        - dims: coordinate dimensions to smooth
        - strict: require an effective window for every point
        - inplace: mutate or return a new object
        """
        # Normalize override groups into a point->spec dict
        overrides_dict: dict[str, dict] = {}
        for grp in overrides:
            if not (isinstance(grp, tuple) and len(grp) == 3):
                raise ValueError(
                    "each override must be a tuple: (points, method, window)"
                )
            pts, m, w = grp
            if isinstance(pts, str):
                pts_list = [pts]
            elif isinstance(pts, (list, tuple)):
                pts_list = list(pts)
            else:
                raise ValueError("points must be a list/tuple of names or a single str")
            for p in pts_list:
                overrides_dict[p] = {"method": m, "window": w}

        self._validate_smoothing_inputs(method, dims, overrides_dict)
        points = self.get_point_names()
        specs = self._resolve_smoothing_specs(
            default_method=method,
            default_window=window,
            overrides=overrides_dict,
            points=points,
            strict=strict,
        )
        df_target = self.data if inplace else self.data.copy()
        df_smoothed = apply_smoothing(
            df_target, specs, dims, smoother=smoother, smoother_kwargs=smoother_kwargs
        )
        meta_entry = self._build_smoothing_meta(specs, dims)
        if inplace:
            self.data = df_smoothed
            self.meta["smoothing"] = meta_entry
            return None
        new_meta = copy.deepcopy(self.meta)
        new_meta["smoothing"] = meta_entry
        return self.__class__(df_smoothed, new_meta, self.handle, self.tags)

    def _validate_smoothing_inputs(
        self,
        method: str,
        dims: tuple[str, ...],
        overrides: dict | None,
    ) -> None:
        if "smoothing" in self.meta.keys():
            raise Exception(
                "data already smoothed. load again to use different smoothing"
            )
        if method not in {"median", "mean"}:
            raise ValueError("method must be one of {'median','mean'}")
        if not set(dims).issubset({"x", "y", "z"}):
            raise ValueError("dims must be a subset of {'x','y','z'}")
        if overrides:
            unknown = set(overrides.keys()) - set(self.get_point_names())
            if unknown:
                raise ValueError(f"overrides contain unknown points: {sorted(unknown)}")

    def _resolve_smoothing_specs(
        self,
        *,
        default_method: str,
        default_window: int | None,
        overrides: dict[str, dict],
        points: list[str],
        strict: bool,
    ) -> dict[str, dict]:
        allowed_methods = {"median", "mean"}
        specs: dict[str, dict] = {}
        for p in points:
            m = default_method
            w = default_window
            spec = overrides.get(p)
            if spec is None:
                pass
            elif isinstance(spec, dict):
                if "method" in spec:
                    if spec["method"] not in allowed_methods:
                        raise ValueError(
                            f"override for {p}: method must be one of {allowed_methods}"
                        )
                    m = spec["method"]
                if "window" in spec:
                    w = int(spec["window"]) if spec["window"] is not None else None
            else:
                raise ValueError(
                    f"Invalid override for {p}: expected dict with keys 'method'/'window', got {type(spec)}"
                )
            if strict and (w is None or w <= 0):
                raise ValueError(
                    f"No valid window resolved for point '{p}' with strict=True"
                )
            specs[p] = {"method": m, "window": None if not w or w <= 0 else int(w)}
        return specs

    def _build_smoothing_meta(
        self, specs: dict[str, dict], dims: tuple[str, ...]
    ) -> dict:
        return {"spec": specs, "dims": list(dims)}

    def interpolate(self, method: str = "linear", limit: int = 1, **kwargs) -> None:
        """
        interpolates missing data in the tracking data, and sets likelihood to np.nan
        uses pandas.DataFrame.interpolate() with kwargs
        """
        if "interpolation" in self.meta.keys():
            raise Exception(
                "data already interpolated. re-load the raw data to interpolate again"
            )

        # interpolate only the position columns, and set likelihood to np.nan
        position_columns = self.data.columns[
            self.data.columns.str.endswith(".x")
            | self.data.columns.str.endswith(".y")
            | self.data.columns.str.endswith(".z")
        ]
        self.data.loc[:, position_columns] = self.data.loc[
            :, position_columns
        ].interpolate(method=method, limit=limit, **kwargs)
        self.data.loc[:, self.data.columns.str.endswith(".likelihood")] = np.nan

        self.meta["interpolation"] = {
            "method": method,
            "limit": limit,
            "kwargs": kwargs,
        }

    @property
    def loc(self):
        return _Indexer(self, self._loc)

    @property
    def iloc(self):
        return _Indexer(self, self._iloc)

    def _loc(self, idx):
        new_data = self.data.loc[idx].copy()
        new_meta = copy.deepcopy(self.meta)
        return self.__class__(new_data, new_meta, self.handle)

    def _iloc(self, idx):
        new_data = self.data.iloc[idx].copy()
        new_meta = copy.deepcopy(self.meta)
        return self.__class__(new_data, new_meta, self.handle)

    def __getitem__(self, idx):
        return self.loc[idx]

    def plot(
        self,
        trajectories=None,
        static=None,
        lines=None,
        dims=("x", "y"),
        ax=None,
        title=None,
        show=True,
        elev=30,
        azim=45,
    ):
        """
        Plot trajectories and static points for this Tracking object.
        Args:
            trajectories: list of point names or dict {point: color_series}
            static: list of point names to plot as static (median)
            lines: list of (point1, point2) pairs to join with a line
            dims: tuple of dimension names (default ('x','y'); use ('x','y','z') for 3D)
            ax: matplotlib axis (optional)
            title: plot title (default: self.handle)
            show: whether to call plt.show()
        Returns: fig, ax
        """
        import numpy as np

        is3d = len(dims) == 3
        if len(dims) > 3:
            raise ValueError("dims must be a tuple of length 2 or 3")
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            if is3d:
                ax = fig.add_subplot(111, projection="3d")
                ax.view_init(elev=elev, azim=azim)
            else:
                ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        # Prepare trajectories
        if trajectories is None:
            trajectories = []
        if static is None:
            static = []
        if lines is None:
            lines = []
        # If dict, allow color series for each trajectory
        if isinstance(trajectories, dict):
            traj_points = list(trajectories.keys())
        else:
            traj_points = list(trajectories)
        # Plot trajectories
        for point in traj_points:
            cols = [f"{point}.{d}" for d in dims]
            for c in cols:
                if c not in self.data.columns:
                    raise ValueError(f"Column {c} not in data for point {point}")
            arrs = [self.data[f"{point}.{d}"].values for d in dims]
            mask = np.all([np.isfinite(a) for a in arrs], axis=0)
            arrs = [a[mask] for a in arrs]
            if isinstance(trajectories, dict) and isinstance(
                trajectories[point], pd.Series
            ):
                cvals = trajectories[point].values[mask]
                sc = ax.scatter(*arrs, c=cvals, cmap="viridis", label=point, s=8)
                plt.colorbar(sc, ax=ax, label=f"{point} color")
            else:
                if is3d:
                    ax.plot(*arrs, label=point)
                else:
                    ax.plot(*arrs, label=point)
        # Plot static points (median)
        for point in static:
            cols = [f"{point}.{d}" for d in dims]
            for c in cols:
                if c not in self.data.columns:
                    raise ValueError(f"Column {c} not in data for point {point}")
            med = [np.nanmedian(self.data[f"{point}.{d}"]) for d in dims]
            if is3d:
                ax.scatter(*med, marker="o", s=60)
            else:
                ax.scatter(*med, marker="o", s=60)
        # Plot lines between static points
        for p1, p2 in lines:
            cols1 = [f"{p1}.{d}" for d in dims]
            cols2 = [f"{p2}.{d}" for d in dims]
            for c in cols1 + cols2:
                if c not in self.data.columns:
                    raise ValueError(f"Column {c} not in data for line {p1}-{p2}")
            med1 = [np.nanmedian(self.data[f"{p1}.{d}"]) for d in dims]
            med2 = [np.nanmedian(self.data[f"{p2}.{d}"]) for d in dims]
            if is3d:
                ax.plot(
                    [med1[0], med2[0]],
                    [med1[1], med2[1]],
                    [med1[2], med2[2]],
                    "k",
                    lw=1,
                )
            else:
                ax.plot([med1[0], med2[0]], [med1[1], med2[1]], "k", lw=1)
        if title is None:
            title = self.handle
        # label axes with dims
        ax.set_xlabel(dims[0])
        ax.set_ylabel(dims[1])
        if is3d:
            ax.set_zlabel(dims[2])
        ax.set_title(title)
        ax.legend()
        if show:
            plt.show()
        return fig, ax

    def save_3d_tracking_video_multi_view(
        self,
        out_path: str,
        lines: list[tuple[str, str]] = None,
        point_size=40,
        line_width=2,
        point_color="b",
        line_color="k",
        dpi=150,
        writer="pillow",
        startframe=None,
        endframe=None,
        xlim=None,
        ylim=None,
        zlim=None,
        robust_percentile=1,
        invert_z=True,
    ):
        """
        Save a 3D animation of tracked points to a video file, with 4 subplots per frame:
        - azim=0, elev=0, ortho
        - azim=90, elev=0, ortho
        - azim=0, elev=90, ortho
        - azim=45, elev=30, persp
        Optionally, set axis limits manually or use robust percentiles to ignore outliers.
        Enforces equal aspect ratio for all axes.
        """
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from matplotlib import animation
        import matplotlib.pyplot as plt

        def get_robust_limits(data, lower=1, upper=99):
            return float(np.percentile(data, lower)), float(np.percentile(data, upper))

        def set_axes_equal(ax, xlim, ylim, zlim):
            xmid = np.mean(xlim)
            ymid = np.mean(ylim)
            zmid = np.mean(zlim)
            max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]) / 2
            ax.set_xlim(xmid - max_range, xmid + max_range)
            ax.set_ylim(ymid - max_range, ymid + max_range)
            ax.set_zlim(zmid - max_range, zmid + max_range)

        if lines is None:
            lines = []
        frames = self.data.index
        fps = self.meta["fps"]
        # Determine frame range
        if startframe is not None:
            if startframe in frames:
                start_idx = np.where(frames == startframe)[0][0]
            else:
                start_idx = int(startframe)
        else:
            start_idx = 0
        if endframe is not None:
            if endframe in frames:
                end_idx = np.where(frames == endframe)[0][0] + 1
            else:
                end_idx = int(endframe) + 1
        else:
            end_idx = len(frames)
        selected_frames = frames[start_idx:end_idx]

        point_names = self.get_point_names()

        # Precompute all coordinates for efficiency
        coords_per_frame = []
        total_frames = len(selected_frames)
        try:
            from tqdm import tqdm

            use_tqdm = True
        except ImportError:
            use_tqdm = False
        if use_tqdm:
            frame_iter = tqdm(
                selected_frames, desc="Precomputing 3D coordinates", unit="frame"
            )
        else:
            frame_iter = selected_frames
            print("Precomputing 3D coordinates...")
        for idx, frame in enumerate(frame_iter):
            coords = {}
            for point in point_names:
                try:
                    x = self.data.loc[frame, point + ".x"]
                    y = self.data.loc[frame, point + ".y"]
                    z = self.data.loc[frame, point + ".z"]
                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                        coords[point] = (x, y, -z)  # Reverse z
                except KeyError:
                    continue
            coords_per_frame.append(coords)
            if (
                not use_tqdm
                and total_frames > 0
                and idx % max(1, total_frames // 10) == 0
            ):
                print(f"  {idx + 1}/{total_frames} frames processed...")
        if not use_tqdm:
            print("Precompute done.")
        if invert_z:
            for coords in coords_per_frame:
                for point in coords:
                    coords[point] = (
                        coords[point][0],
                        coords[point][1],
                        -coords[point][2],
                    )

        # Set up figure and axes
        fig = plt.figure(figsize=(12, 10))
        axs = [
            fig.add_subplot(221, projection="3d"),
            fig.add_subplot(222, projection="3d"),
            fig.add_subplot(223, projection="3d"),
            fig.add_subplot(224, projection="3d"),
        ]

        # View settings: (elev, azim, proj_type)
        views = [
            (30, 135, "persp"),  # front
            (30, 225, "persp"),  # side
            (90, 0, "ortho"),  # top
            (30, 45, "persp"),  # isometric
        ]
        titles = [
            "Isometric (azim=135, elev=30, persp)",
            "Isometric (azim=225, elev=30, persp)",
            "Top (azim=0, elev=90, ortho)",
            "Isometric (azim=45, elev=30, persp)",
        ]

        # Set up plot elements (scatter and lines) for each axis
        scatters = []
        line_objs = []
        for ax, (elev, azim, proj_type), title in zip(axs, views, titles):
            ax.view_init(elev=elev, azim=azim)
            ax.set_proj_type(proj_type)
            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            scatters.append(ax.scatter([], [], [], s=point_size, c=point_color))
            line_objs.append(
                [
                    ax.plot([], [], [], color=line_color, linewidth=line_width)[0]
                    for _ in lines
                ]
            )

        # Set axis limits based on all data (robust to outliers)
        all_x, all_y, all_z = [], [], []
        for coords in coords_per_frame:
            for x, y, z in coords.values():
                all_x.append(x)
                all_y.append(y)
                all_z.append(z)
        if not all_x or not all_y or not all_z:
            raise ValueError("No valid 3D points found for plotting.")

        if xlim is None:
            xlim = get_robust_limits(all_x, robust_percentile, 100 - robust_percentile)
        if ylim is None:
            ylim = get_robust_limits(all_y, robust_percentile, 100 - robust_percentile)
        if zlim is None:
            zlim = get_robust_limits(all_z, robust_percentile, 100 - robust_percentile)

        for ax in axs:
            set_axes_equal(ax, xlim, ylim, zlim)

        # Progress bar for animation saving
        save_progress = None
        save_total = len(coords_per_frame)
        try:
            from tqdm import tqdm as tqdm_save

            use_tqdm_save = True
        except ImportError:
            use_tqdm_save = False
        if use_tqdm_save:
            save_progress = tqdm_save(
                total=save_total, desc="Rendering animation frames", unit="frame"
            )
        else:
            print("Rendering animation frames...")
            save_progress = None
            save_last_print = -1

        def update(frame_idx):
            coords = coords_per_frame[frame_idx]
            xs, ys, zs = zip(*coords.values()) if coords else ([], [], [])
            for i, ax in enumerate(axs):
                scatters[i]._offsets3d = (xs, ys, zs)
                # Update lines
                for j, (p1, p2) in enumerate(lines):
                    if p1 in coords and p2 in coords:
                        xline = [coords[p1][0], coords[p2][0]]
                        yline = [coords[p1][1], coords[p2][1]]
                        zline = [coords[p1][2], coords[p2][2]]
                        line_objs[i][j].set_data(xline, yline)
                        line_objs[i][j].set_3d_properties(zline)
                        line_objs[i][j].set_visible(True)
                    else:
                        line_objs[i][j].set_visible(False)
                ax.set_title(f"{titles[i]}\nFrame {selected_frames[frame_idx]}")
            # Progress update
            if save_progress is not None:
                save_progress.update(1)
            else:
                nonlocal save_last_print
                if (
                    save_total > 0
                    and frame_idx % max(1, save_total // 10) == 0
                    and frame_idx != save_last_print
                ):
                    print(f"  {frame_idx + 1}/{save_total} frames rendered...")
                    save_last_print = frame_idx
            return [item for sublist in line_objs for item in sublist] + scatters

        anim = animation.FuncAnimation(
            fig, update, frames=len(coords_per_frame), interval=1000 / fps, blit=False
        )

        # Save animation
        if writer == "ffmpeg":
            Writer = animation.FFMpegWriter
        elif writer == "pillow":
            Writer = animation.PillowWriter
        else:
            raise ValueError("writer must be 'ffmpeg' or 'pillow'")
        anim.save(out_path, writer=Writer(fps=fps), dpi=dpi)
        if save_progress is not None:
            save_progress.close()
        else:
            print("Rendering done.")
        plt.close(fig)
        print(f"Saved 3D tracking video to {out_path}")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.data)} rows, fps={self.meta.get('fps', 'unknown')}>"
