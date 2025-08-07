from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Type, TypeVar
import pandas as pd
import re
import warnings
import os
import json
import numpy as np
from py3r.behaviour.exceptions import BatchProcessError
from collections import defaultdict
import copy
from py3r.behaviour.util.collection_utils import _Indexer, BatchResult
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from py3r.behaviour.util.dev_utils import dev_mode

Self = TypeVar("Self", bound="Tracking")


@dataclass(kw_only=True)
class LoadOptions:
    fps: float
    aspectratio_correction: float = 1.0
    usermeta: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        # Validate fps
        if not isinstance(self.fps, (int, float)):
            raise TypeError(
                f"fps must be a number (int or float), got {type(self.fps).__name__}"
            )
        self.fps = float(self.fps)

        # Validate aspectratio_correction
        if not isinstance(self.aspectratio_correction, (int, float)):
            raise TypeError(
                f"aspectratio_correction must be a number (int or float), got {type(self.aspectratio_correction).__name__}"
            )
        self.aspectratio_correction = float(self.aspectratio_correction)

        # Validate usermeta
        if self.usermeta is not None:
            if not isinstance(self.usermeta, dict):
                raise TypeError(
                    f"usermeta must be a dictionary, got {type(self.usermeta).__name__}"
                )


class Tracking:
    data: pd.DataFrame
    meta: dict
    handle: str

    @classmethod
    def from_dlc(
        cls: Type[Self], filepath: str, *, handle: str, options: LoadOptions
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
            "fps": options.fps,
            "aspectratio_correction": options.aspectratio_correction,
            "network": scorer,
            "usermeta": options.usermeta,
        }

        data = cls._apply_aspectratio_correction(data, options.aspectratio_correction)

        return cls(data, meta, handle)

    @classmethod
    def from_dlcma(
        cls: Type[Self], filepath: str, *, handle: str, options: LoadOptions
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
            "fps": options.fps,
            "aspectratio_correction": options.aspectratio_correction,
            "network": scorer,
            "usermeta": options.usermeta,
        }

        data = cls._apply_aspectratio_correction(data, options.aspectratio_correction)

        return cls(data, meta, handle)

    @classmethod
    def from_yolo3r(
        cls: Type[Self], filepath: str, *, handle: str, options: LoadOptions
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
            "fps": options.fps,
            "aspectratio_correction": options.aspectratio_correction,
            "usermeta": options.usermeta,
        }

        data = cls._apply_aspectratio_correction(data, options.aspectratio_correction)

        return cls(data, meta, handle)

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

    def __init__(self, data: pd.DataFrame, meta: Dict[str, Any], handle: str) -> None:
        if not isinstance(meta, dict):
            raise TypeError(f"meta must be a dictionary, got {type(meta).__name__}")
        if "fps" not in meta:
            raise ValueError("meta dictionary must contain 'fps' key")
        self.data = data
        self.meta = meta
        self.handle = handle

    # ----------- Instance methods -----------

    def __getitem__(self, key: str) -> pd.Series:
        """
        returns a single column of the tracking data
        """
        return self.data[key]

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

    def save(self, filepath: str) -> None:
        """saves .csv file and _meta.json file to disk at location specified by filepath"""
        basefilepath = filepath.split(".csv")[0]
        self.data.to_csv(basefilepath + ".csv")
        with open(os.path.expanduser(basefilepath + "_meta.json"), "w") as f:
            json.dump(self.meta, f)

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
        frame_indices = range(start_idx, end_idx)
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
            (0, 0, "ortho"),  # front
            (0, 90, "ortho"),  # side
            (90, 0, "ortho"),  # top
            (30, 45, "persp"),  # isometric
        ]
        titles = [
            "Front (azim=0, elev=0, ortho)",
            "Side (azim=90, elev=0, ortho)",
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


class TrackingCollection:
    """
    Collection of Tracking objects, keyed by name (e.g. for grouping individuals)
    note: type-hints refer to Tracking, but factory methods allow for other classes
    these are intended ONLY for subclasses of Tracking, and this is enforced
    """

    tracking_dict: dict[str, Tracking]

    def __init__(self, tracking_dict: dict[str, Tracking]):
        for key, obj in tracking_dict.items():
            if obj.handle != key:
                raise ValueError(
                    f"Key '{key}' does not match object's handle '{obj.handle}'"
                )
        self.tracking_dict = tracking_dict

    def __getattr__(self, name):
        def batch_method(*args, **kwargs):
            results = {}
            for key, obj in self.tracking_dict.items():
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
    def from_dlc(
        cls,
        handles_and_filepaths: dict[str, str],
        options: LoadOptions,
        tracking_cls=Tracking,
    ):
        """
        Loads a TrackingCollection from a dict of DLC tracking csvs.
        handles_and_filepaths: dict mapping handles to file paths.
        """
        if not issubclass(tracking_cls, Tracking):
            raise TypeError(
                f"tracking_cls must be Tracking or a subclass, got {tracking_cls}"
            )
        trackings = {}
        for handle, fp in handles_and_filepaths.items():
            trackings[handle] = tracking_cls.from_dlc(
                fp, handle=handle, options=options
            )
        return cls(trackings)

    @classmethod
    def from_yolo3r(
        cls,
        handles_and_filepaths: dict[str, str],
        options: LoadOptions,
        tracking_cls=Tracking,
    ):
        """
        Loads a TrackingCollection from a dict of yolo3r tracking csvs.
        handles_and_filepaths: dict mapping handles to file paths.
        """
        if not issubclass(tracking_cls, Tracking):
            raise TypeError(
                f"tracking_cls must be Tracking or a subclass, got {tracking_cls}"
            )
        trackings = {}
        for handle, fp in handles_and_filepaths.items():
            trackings[handle] = tracking_cls.from_yolo3r(
                fp, handle=handle, options=options
            )
        return cls(trackings)

    @classmethod
    def from_dlcma(
        cls,
        handles_and_filepaths: dict[str, str],
        options: LoadOptions,
        tracking_cls=Tracking,
    ):
        """
        Loads a TrackingCollection from a dict of DLC multi-animal tracking csvs.
        handles_and_filepaths: dict mapping handles to file paths.
        """
        if not issubclass(tracking_cls, Tracking):
            raise TypeError(
                f"tracking_cls must be Tracking or a subclass, got {tracking_cls}"
            )
        trackings = {}
        for handle, fp in handles_and_filepaths.items():
            trackings[handle] = tracking_cls.from_dlcma(
                fp, handle=handle, options=options
            )
        return cls(trackings)

    @classmethod
    def from_list(cls, tracking_list: list[Tracking]):
        """
        creates a TrackingCollection from a list of Tracking objects, keyed by handle
        """
        handles = [obj.handle for obj in tracking_list]
        if len(handles) != len(set(handles)):
            raise Exception("handles must be unique")
        trackings = {obj.handle: obj for obj in tracking_list}
        return cls(trackings)

    @dev_mode
    @classmethod
    def from_dogfeather(
        cls,
        handles_and_filepaths: dict[str, str],
        options: LoadOptions,
        tracking_cls=Tracking,
    ):
        """
        Loads a TrackingCollection from a dict of dogfeather tracking csvs.
        handles_and_filepaths: dict mapping handles to file paths.
        """

        trackings = {}
        for handle, fp in handles_and_filepaths.items():
            trackings[handle] = tracking_cls.from_dogfeather(
                fp, handle=handle, options=options
            )
        return cls(trackings)

    @classmethod
    def from_dlc_folder(
        cls, folder_path: str, options: "LoadOptions", tracking_cls: type = Tracking
    ) -> "TrackingCollection":
        tracking_dict = {}
        bookkeeping = cls._collect_tracking_files(
            folder_path, tracking_cls, options=options
        )
        for handle, kwargs in bookkeeping.items():
            tracking_obj = tracking_cls.from_dlc(**kwargs)
            tracking_dict[handle] = tracking_obj
        return cls(tracking_dict)

    @classmethod
    def from_yolo3r_folder(
        cls, folder_path: str, options: "LoadOptions", tracking_cls: type = Tracking
    ) -> "TrackingCollection":
        tracking_dict = {}
        bookkeeping = cls._collect_tracking_files(
            folder_path, tracking_cls, options=options
        )
        for handle, kwargs in bookkeeping.items():
            tracking_obj = tracking_cls.from_yolo3r(**kwargs)
            tracking_dict[handle] = tracking_obj
        return cls(tracking_dict)

    @classmethod
    def from_dlcma_folder(
        cls, folder_path: str, options: "LoadOptions", tracking_cls: type = Tracking
    ) -> "TrackingCollection":
        tracking_dict = {}
        bookkeeping = cls._collect_tracking_files(
            folder_path, tracking_cls, options=options
        )
        for handle, kwargs in bookkeeping.items():
            tracking_obj = tracking_cls.from_dlcma(**kwargs)
            tracking_dict[handle] = tracking_obj
        return cls(tracking_dict)

    def stereo_triangulate(self):
        """
        Triangulate all TrackingMV objects in the collection.
        Returns a new TrackingCollection of triangulated Tracking objects.
        """
        triangulated = {}
        for handle, obj in self.tracking_dict.items():
            if hasattr(obj, "stereo_triangulate"):
                triangulated[handle] = obj.stereo_triangulate()
            else:
                raise TypeError(
                    f"Object {handle} does not support stereo_triangulate()"
                )
        return TrackingCollection(triangulated)

    @staticmethod
    def _collect_tracking_files(folder_path, tracking_cls, options):
        """
        Returns a dict of handle -> kwargs dict for tracking_cls factory method.
        For TrackingMV: handle -> {filepaths, handle, options, calibration}
        For Tracking: handle -> {filepath, handle, options}
        """
        result = {}
        if tracking_cls is TrackingMV:
            # Each subfolder is a recording
            print(f"Scanning {folder_path} for recordings...")
            for recording in sorted(os.listdir(folder_path)):
                recording_path = os.path.join(folder_path, recording)
                print(f"  Checking {recording_path}...")
                if not os.path.isdir(recording_path):
                    print("    Not a directory, skipping.")
                    continue
                # Find all view csvs
                filepaths = {}
                for fname in os.listdir(recording_path):
                    if fname.endswith(".csv") and not fname.startswith("."):
                        view = os.path.splitext(fname)[0]
                        filepaths[view] = os.path.join(recording_path, fname)
                # Load calibration
                calib_path = os.path.join(recording_path, "calibration.json")
                if not os.path.exists(calib_path):
                    raise FileNotFoundError(
                        f"Missing calibration.json in {recording_path}"
                    )
                with open(calib_path, "r") as f:
                    calibration = json.load(f)
                result[recording] = {
                    "filepaths": filepaths,
                    "handle": recording,
                    "options": options,
                    "calibration": calibration,
                }
        else:
            # Single-view: treat each csv as a single-view Tracking
            for fname in os.listdir(folder_path):
                if fname.endswith(".csv") and not fname.startswith("."):
                    handle = os.path.splitext(fname)[0]
                    fpath = os.path.join(folder_path, fname)
                    result[handle] = {
                        "filepath": fpath,
                        "handle": handle,
                        "options": options,
                    }
        return result

    @property
    def loc(self):
        return _Indexer(self, self._loc)

    @property
    def iloc(self):
        return _Indexer(self, self._iloc)

    def _loc(self, idx):
        return self.__class__({k: v.loc[idx] for k, v in self.tracking_dict.items()})

    def _iloc(self, idx):
        return self.__class__({k: v.iloc[idx] for k, v in self.tracking_dict.items()})

    def __getitem__(self, key):
        """
        Get Tracking by handle (str), by integer index, or by slice.
        """
        if isinstance(key, int):
            handle = list(self.tracking_dict)[key]
            return self.tracking_dict[handle]
        elif isinstance(key, slice):
            handles = list(self.tracking_dict)[key]
            return self.__class__({h: self.tracking_dict[h] for h in handles})
        else:
            return self.tracking_dict[key]

    def keys(self):
        """Return the keys of the tracking_dict."""
        return self.tracking_dict.keys()

    def values(self):
        """Return the values of the tracking_dict."""
        return self.tracking_dict.values()

    def items(self):
        """Return the items of the tracking_dict."""
        return self.tracking_dict.items()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.tracking_dict)} Tracking objects>"

    def plot(self, *args, **kwargs):
        print(f"\nCollection: {getattr(self, 'handle', 'unnamed')}")
        for handle, tracking in self.tracking_dict.items():
            tracking.plot(*args, title=handle, **kwargs)

    def __setitem__(self, key, value):
        """
        Set Tracking by handle (str).
        """
        if not isinstance(value, Tracking):
            raise TypeError(f"Value must be a Tracking, got {type(value).__name__}")
        warnings.warn(
            "Direct assignment to TrackingCollection is deprecated and may be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.tracking_dict[key] = value


class MultipleTrackingCollection:
    """
    Collection of TrackingCollection objects, keyed by name (e.g. for comparison between groups)
    """

    def __init__(self, tracking_collections: dict[str, TrackingCollection]):
        self.tracking_collections = tracking_collections

    def __getattr__(self, name):
        def batch_method(*args, **kwargs):
            results = {}
            for key, obj in self.tracking_collections.items():
                try:
                    method = getattr(obj, name)
                    results[key] = method(*args, **kwargs)
                except Exception as e:
                    raise BatchProcessError(
                        collection_name=key,
                        object_name=getattr(e, "object_name", None),
                        method=getattr(e, "method", None),
                        original_exception=getattr(e, "original_exception", e),
                    ) from e
            return BatchResult(results, self)

        return batch_method

    @classmethod
    def from_dict(cls, trackingcollections: dict[str, TrackingCollection]):
        """
        creates a MultipleTrackingCollection from a dict of TrackingCollection objects
        """
        trackings = {key: obj for key, obj in trackingcollections.items()}
        return cls(trackings)

    @classmethod
    def from_dlc_folder(
        cls, parent_folder: str, options: "LoadOptions", tracking_cls: type = Tracking
    ) -> "MultipleTrackingCollection":
        """
        Create a MultipleTrackingCollection from a folder of subfolders, each containing DLC csv files.
        Args:
            parent_folder: Path to parent folder containing subfolders.
            load_options: LoadOptions instance (required).
            tracking_cls: Tracking class to use (default: Tracking).
        Returns:
            MultipleTrackingCollection with keys as subfolder names, each value a TrackingCollection.
        """
        import os

        tracking_collections = {}
        for subfolder in sorted(os.listdir(parent_folder)):
            subfolder_path = os.path.join(parent_folder, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            tc = TrackingCollection.from_dlc_folder(
                subfolder_path, options=options, tracking_cls=tracking_cls
            )
            tracking_collections[subfolder] = tc
        return cls(tracking_collections)

    @classmethod
    def from_yolo3r_folder(
        cls, parent_folder: str, options: "LoadOptions", tracking_cls: type = Tracking
    ) -> "MultipleTrackingCollection":
        """
        Create a MultipleTrackingCollection from a folder of subfolders, each containing YOLO3R csv files.
        Args:
            parent_folder: Path to parent folder containing subfolders.
            options: LoadOptions instance (required).
            tracking_cls: Tracking class to use (default: Tracking).
        Returns:
            MultipleTrackingCollection with keys as subfolder names, each value a TrackingCollection.
        """
        import os

        tracking_collections = {}
        for subfolder in sorted(os.listdir(parent_folder)):
            subfolder_path = os.path.join(parent_folder, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            tc = TrackingCollection.from_yolo3r_folder(
                subfolder_path, options=options, tracking_cls=tracking_cls
            )
            tracking_collections[subfolder] = tc
        return cls(tracking_collections)

    @classmethod
    def from_dlcma_folder(
        cls, parent_folder: str, options: "LoadOptions", tracking_cls: type = Tracking
    ) -> "MultipleTrackingCollection":
        """
        Create a MultipleTrackingCollection from a folder of subfolders, each containing DLCMA csv files.
        Args:
            parent_folder: Path to parent folder containing subfolders.
            options: LoadOptions instance (required).
            tracking_cls: Tracking class to use (default: Tracking).
        Returns:
            MultipleTrackingCollection with keys as subfolder names, each value a TrackingCollection.
        """
        import os

        tracking_collections = {}
        for subfolder in sorted(os.listdir(parent_folder)):
            subfolder_path = os.path.join(parent_folder, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            tc = TrackingCollection.from_dlcma_folder(
                subfolder_path, options=options, tracking_cls=tracking_cls
            )
            tracking_collections[subfolder] = tc
        return cls(tracking_collections)

    def stereo_triangulate(self):
        """
        Triangulate all TrackingMV objects in all collections.
        Returns a new MultipleTrackingCollection of triangulated TrackingCollections.
        """
        triangulated = {}
        for group, collection in self.tracking_collections.items():
            triangulated[group] = collection.stereo_triangulate()
        return MultipleTrackingCollection(triangulated)

    @property
    def loc(self):
        return _Indexer(self, self._loc)

    @property
    def iloc(self):
        return _Indexer(self, self._iloc)

    def _loc(self, idx):
        return self.__class__(
            {k: v.loc[idx] for k, v in self.tracking_collections.items()}
        )

    def _iloc(self, idx):
        return self.__class__(
            {k: v.iloc[idx] for k, v in self.tracking_collections.items()}
        )

    def __getitem__(self, key):
        """
        Get TrackingCollection by handle (str), by integer index, or by slice.
        """
        if isinstance(key, int):
            handle = list(self.tracking_collections)[key]
            return self.tracking_collections[handle]
        elif isinstance(key, slice):
            handles = list(self.tracking_collections)[key]
            return self.__class__({h: self.tracking_collections[h] for h in handles})
        else:
            return self.tracking_collections[key]

    def keys(self):
        """Return the keys of the tracking_collections."""
        return self.tracking_collections.keys()

    def values(self):
        """Return the values of the tracking_collections."""
        return self.tracking_collections.values()

    def items(self):
        """Return the items of the tracking_collections."""
        return self.tracking_collections.items()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.tracking_collections)} TrackingCollection objects>"

    def plot(self, *args, **kwargs):
        for handle, collection in self.tracking_collections.items():
            print(f"\n=== Group: {handle} ===")
            collection.plot(*args, **kwargs)

    def __setitem__(self, key, value):
        """
        Set TrackingCollection by handle (str).
        """
        if not isinstance(value, TrackingCollection):
            raise TypeError(
                f"Value must be a TrackingCollection, got {type(value).__name__}"
            )
        warnings.warn(
            "Direct assignment to MultipleTrackingCollection is deprecated and may be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.tracking_collections[key] = value


class TrackingMV:
    """
    multi-view tracking object for stereo or multi-camera setups
    can be used as a drop-in replacement for Tracking in TrackingCollection
    stores dict of view name -> Tracking, calibration, and handle
    """

    def __init__(self, views: dict[str, Tracking], calibration: dict, handle: str):
        self.views = views  # e.g., {'left': Tracking, 'right': Tracking}
        self.calibration = calibration
        self.handle = handle

    @classmethod
    def from_dlc(
        cls,
        filepaths: dict[str, str],
        handle: str,
        options: LoadOptions,
        calibration: dict,
    ):
        """
        Load a TrackingMV from a dict of view name -> csv filepath.
        """
        tracks = {
            view: Tracking.from_dlc(fp, handle=f"{handle}_{view}", options=options)
            for view, fp in filepaths.items()
        }
        return cls(tracks, calibration, handle)

    @classmethod
    def from_dlcma(
        cls,
        filepaths: dict[str, str],
        handle: str,
        options: LoadOptions,
        calibration: dict,
    ):
        """
        Load a TrackingMV from a dict of view name -> csv filepath.
        """
        tracks = {
            view: Tracking.from_dlcma(fp, handle=f"{handle}_{view}", options=options)
            for view, fp in filepaths.items()
        }
        return cls(tracks, calibration, handle)

    @classmethod
    def from_yolo3r(
        cls,
        filepaths: dict[str, str],
        handle: str,
        options: LoadOptions,
        calibration: dict,
    ):
        """
        Load a TrackingMV from a dict of view name -> csv filepath.
        """
        tracks = {
            view: Tracking.from_yolo3r(fp, handle=f"{handle}_{view}", options=options)
            for view, fp in filepaths.items()
        }
        return cls(tracks, calibration, handle)

    def stereo_triangulate(self, invert_z: bool = True) -> "Tracking":
        """
        Triangulate the two views to produce a 3D Tracking object.
        Returns a new Tracking object with .x, .y, .z columns.
        invert_z is true by default to align with typical top-down behaviour tracking setups
        """
        import cv2
        import numpy as np

        calib = self.calibration

        # validation
        views = list(self.views.keys())
        if len(views) != 2 or len(views) != len(set(views)):
            raise ValueError("Exactly two unique views are required for triangulation")
        if not all(v in calib["view_order"] for v in views):
            raise ValueError("Views must correspond to calibration")

        v1, v2 = calib["view_order"][:2]
        K1, dist1 = (
            np.array(calib["views"][v1]["K"]),
            np.array(calib["views"][v1]["dist"]),
        )
        K2, dist2 = (
            np.array(calib["views"][v2]["K"]),
            np.array(calib["views"][v2]["dist"]),
        )
        R, T = (
            np.array(calib["relative_pose"]["R"]),
            np.array(calib["relative_pose"]["T"]).reshape(3, 1),
        )

        # data extraction
        df1, df2 = self.views[v1].data, self.views[v2].data
        if not df1.columns.equals(df2.columns):
            raise ValueError("Views have different columns")
        point_names = self.views[v1].get_point_names()
        frames = df1.index.intersection(df2.index)

        # projection matrices
        P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K2 @ np.hstack((R, T))

        # helper for triangulation
        def triangulate_point(xs1, ys1, xs2, ys2, l1, l2):
            valid = (
                np.isfinite(xs1)
                & np.isfinite(ys1)
                & np.isfinite(xs2)
                & np.isfinite(ys2)
            )
            n = len(xs1)
            x3d, y3d, z3d, likelihood = [np.full(n, np.nan) for _ in range(4)]
            if np.any(valid):
                pts1 = np.stack([xs1[valid], ys1[valid]], axis=1).astype(np.float32)
                pts2 = np.stack([xs2[valid], ys2[valid]], axis=1).astype(np.float32)
                pts1_ud = cv2.undistortPoints(
                    pts1.reshape(-1, 1, 2), K1, dist1, P=K1
                ).reshape(-1, 2)
                pts2_ud = cv2.undistortPoints(
                    pts2.reshape(-1, 1, 2), K2, dist2, P=K2
                ).reshape(-1, 2)
                pts4d = cv2.triangulatePoints(P1, P2, pts1_ud.T, pts2_ud.T)
                pts3d = (pts4d[:3] / pts4d[3]).T
                x3d[valid], y3d[valid], z3d[valid] = (
                    pts3d[:, 0],
                    pts3d[:, 1],
                    pts3d[:, 2],
                )
                likelihood[valid] = np.minimum(l1[valid], l2[valid])
            return x3d, y3d, z3d, likelihood

        # triangulate all points
        triangulated = {}
        l1_all = df1.loc[frames, [p + ".likelihood" for p in point_names]].values
        l2_all = df2.loc[frames, [p + ".likelihood" for p in point_names]].values
        for i, point in enumerate(point_names):
            xs1, ys1 = (
                df1.loc[frames, point + ".x"].values,
                df1.loc[frames, point + ".y"].values,
            )
            xs2, ys2 = (
                df2.loc[frames, point + ".x"].values,
                df2.loc[frames, point + ".y"].values,
            )
            l1, l2 = l1_all[:, i], l2_all[:, i]
            x3d, y3d, z3d, likelihood = triangulate_point(xs1, ys1, xs2, ys2, l1, l2)
            triangulated[point + ".x"] = x3d
            triangulated[point + ".y"] = y3d
            if invert_z:
                triangulated[point + ".z"] = -z3d
            else:
                triangulated[point + ".z"] = z3d
            triangulated[point + ".likelihood"] = likelihood

        triangulated_df = pd.DataFrame(triangulated, index=frames)
        triangulated_meta = self.views[v1].meta.copy()
        triangulated_meta["calibration"] = calib
        triangulated_meta["views"] = views
        return Tracking(triangulated_df, triangulated_meta, self.handle)

    @staticmethod
    def _get_animals_and_bodyparts(df):
        animals = set()
        bodyparts = set()
        for col in df.columns:
            parts = col.split(".")
            if len(parts) >= 2:
                animals.add(parts[0])
                bodyparts.add(parts[1])
        return sorted(animals), sorted(bodyparts)

    @staticmethod
    def _extract_coords(df, animals, keypoints, frames):
        import numpy as np

        coords = np.full(
            (len(frames), len(animals), len(keypoints), 2), np.nan, dtype=np.float32
        )
        for ai, animal in enumerate(animals):
            for ki, kp in enumerate(keypoints):
                xcol = f"{animal}.{kp}.x"
                ycol = f"{animal}.{kp}.y"
                if xcol in df.columns and ycol in df.columns:
                    coords[:, ai, ki, 0] = df.loc[frames, xcol].values
                    coords[:, ai, ki, 1] = df.loc[frames, ycol].values
        return coords

    @staticmethod
    def _compute_cost_matrices(coords1, coords2):
        import numpy as np

        # coords1, coords2: (n_frames, n_animals, n_keypoints, 2)
        coords1_exp = coords1[
            :, :, None, :, :
        ]  # (n_frames, n_animals, 1, n_keypoints, 2)
        coords2_exp = coords2[
            :, None, :, :, :
        ]  # (n_frames, 1, n_animals, n_keypoints, 2)
        diffs = (
            coords1_exp - coords2_exp
        )  # (n_frames, n_animals, n_animals, n_keypoints, 2)
        sqdist = np.sum(
            diffs**2, axis=-1
        )  # (n_frames, n_animals, n_animals, n_keypoints)
        dists = np.sqrt(sqdist)  # (n_frames, n_animals, n_animals, n_keypoints)
        mask1 = np.isfinite(coords1).all(axis=-1)  # (n_frames, n_animals, n_keypoints)
        mask2 = np.isfinite(coords2).all(axis=-1)  # (n_frames, n_animals, n_keypoints)
        mask1_exp = mask1[:, :, None, :]  # (n_frames, n_animals, 1, n_keypoints)
        mask2_exp = mask2[:, None, :, :]  # (n_frames, 1, n_animals, n_keypoints)
        valid_mask = (
            mask1_exp & mask2_exp
        )  # (n_frames, n_animals, n_animals, n_keypoints)
        dists_masked = np.where(valid_mask, dists, 0.0)
        n_valid = np.sum(valid_mask, axis=-1)  # (n_frames, n_animals, n_animals)
        sum_dists = np.sum(dists_masked, axis=-1)  # (n_frames, n_animals, n_animals)
        large_cost = 1e6
        cost_matrices = np.where(
            n_valid > 0, sum_dists, large_cost
        )  # (n_frames, n_animals, n_animals)
        return cost_matrices

    @staticmethod
    def _reorder_view2_data(
        df2, animals1, animals2, keypoints, aligned_indices, frames
    ):
        import numpy as np
        import pandas as pd

        n_frames = len(frames)
        new_data_dict = {}
        for ai, animal1 in enumerate(animals1):
            for ki, kp in enumerate(keypoints):
                for coord in ["x", "y", "likelihood"]:
                    colname = f"{animal1}.{kp}.{coord}"
                    new_data_dict[colname] = np.full(n_frames, np.nan, dtype=np.float32)
        for fi, frame in enumerate(frames):
            for ai, animal1 in enumerate(animals1):
                aj = aligned_indices[fi, ai]
                for ki, kp in enumerate(keypoints):
                    for coord in ["x", "y", "likelihood"]:
                        src_col = f"{animals2[aj]}.{kp}.{coord}"
                        dst_col = f"{animal1}.{kp}.{coord}"
                        if src_col in df2.columns:
                            new_data_dict[dst_col][fi] = df2.loc[frame, src_col]
        new_df2 = pd.DataFrame(new_data_dict, index=frames)
        extra_cols = [c for c in df2.columns if not any(bp in c for bp in keypoints)]
        if extra_cols:
            new_df2 = pd.concat([new_df2, df2.loc[frames, extra_cols]], axis=1)
        new_df2 = new_df2[
            [c for c in df2.columns if c in new_df2.columns]
            + [c for c in new_df2.columns if c not in df2.columns]
        ]
        return new_df2

    def align_ids_by_keypoints(
        self,
        keypoints: list[str],
        views: list[str] | None = None,
    ) -> "TrackingMV":
        """
        Align animal IDs between two specified views by minimizing the sum of distances between specified keypoints.
        Returns a new TrackingMV with aligned IDs in the second view.
        Args:
            keypoints: list of bodypart names (e.g., ["nose", "tailbase"])
            views: list or tuple of two view names to align (default: first two views in self.views)
        """
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        all_views = list(self.views.keys())
        if views is not None:
            if len(views) != 2:
                raise ValueError(
                    "views argument must be a list or tuple of two view names."
                )
            v1, v2 = views
            if v1 not in self.views or v2 not in self.views:
                raise ValueError(
                    f"Specified views {views} not found in TrackingMV object."
                )
        else:
            if len(all_views) < 2:
                raise ValueError("TrackingMV must have at least two views to align.")
            v1, v2 = all_views[:2]

        df1, df2 = self.views[v1].data, self.views[v2].data
        animals1, _ = self._get_animals_and_bodyparts(df1)
        animals2, _ = self._get_animals_and_bodyparts(df2)
        if len(animals1) != len(animals2):
            raise ValueError("Number of animals differs between views.")
        n_animals = len(animals1)
        frames = df1.index.intersection(df2.index)
        n_frames = len(frames)

        coords1 = self._extract_coords(df1, animals1, keypoints, frames)
        coords2 = self._extract_coords(df2, animals2, keypoints, frames)
        cost_matrices = self._compute_cost_matrices(coords1, coords2)

        aligned_indices = np.zeros((n_frames, n_animals), dtype=int)
        for fi in range(n_frames):
            row_ind, col_ind = linear_sum_assignment(cost_matrices[fi])
            aligned_indices[fi] = col_ind

        new_df2 = self._reorder_view2_data(
            df2, animals1, animals2, keypoints, aligned_indices, frames
        )
        new_views = self.views.copy()
        new_views[v1] = Tracking(
            df1.loc[frames], self.views[v1].meta.copy(), self.views[v1].handle
        )
        new_views[v2] = Tracking(
            new_df2, self.views[v2].meta.copy(), self.views[v2].handle
        )
        return TrackingMV(new_views, self.calibration, self.handle)

    def plot(
        self,
        trajectories=None,
        static=None,
        lines=None,
        dims=("x", "y"),
        ax=None,
        title=None,
        show=True,
    ):
        n = len(self.views)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
        axes = axes[0]  # flatten to 1D
        for i, (view, track) in enumerate(self.views.items()):
            track.plot(
                trajectories=trajectories,
                static=static,
                lines=lines,
                dims=dims,
                ax=axes[i],
                title=view,
                show=False,
            )
        if title is None:
            title = self.handle
        fig.suptitle(title)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def __getattr__(self, name):
        """
        batch method: call method on all underlying Tracking objects
        returns dict of view name -> result
        """

        def batch_method(*args, **kwargs):
            results = {}
            for view, track in self.views.items():
                method = getattr(track, name)
                results[view] = method(*args, **kwargs)
            return results

        return batch_method

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.views)} views: {', '.join(self.views.keys())}>"
