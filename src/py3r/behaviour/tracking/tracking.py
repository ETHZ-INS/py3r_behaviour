from __future__ import annotations

import copy
import re
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from py3r.behaviour.util.collection_utils import _Indexer
from py3r.behaviour.util.dataframe_utils import (
    euclidean_distance,
    filter_by_threshold,
    scale_columns,
)
from py3r.behaviour.util.io_utils import (
    SchemaVersion,
    begin_save,
    read_dataframe,
    read_manifest,
    write_dataframe,
    write_manifest,
)
from py3r.behaviour.util.smoothing import apply_smoothing

Self = TypeVar("Self", bound="Tracking")


class Tracking:
    """
    Represent frame-by-frame tracked keypoints with convenience loaders and tools.

    A `Tracking` holds a pandas DataFrame of columns like `p1.x`, `p1.y`,
    `p1.z`, `p1.likelihood` with index named `frame`. Most users create
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
    Minimal DLC example:

    ```pycon
    >>> from py3r.behaviour.util.docdata import data_path
    >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
    ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
    >>> len(t.data), t.meta['fps'], t.handle
    (5, 30.0, 'ex')
    >>> t.data[['p1.x','p1.y','p1.z','p1.likelihood']].head(2).reset_index().values.tolist()
    [[0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 2.0, 3.0, 0.75]]

    ```

    Load from DLC multi-animal (DLCMA):

    ```pycon
    >>> with data_path('py3r.behaviour.tracking._data', 'dlcma_multi.csv') as p_ma:
    ...     tma = Tracking.from_dlcma(str(p_ma), handle='ma', fps=30)
    >>> tma.meta['fps'], tma.handle
    (30.0, 'ma')

    ```

    Load from YOLO3R (3D columns present):

    ```pycon
    >>> with data_path('py3r.behaviour.tracking._data', 'yolo3r.csv') as p_y:
    ...     ty = Tracking.from_yolo3r(str(p_y), handle='y3r', fps=30)
    >>> 'p1.z' in ty.data.columns and 'p1.likelihood' in ty.data.columns
    True
    >>> ty.data[['p1.x','p1.y','p1.z','p1.likelihood']].head(2).reset_index().values.tolist()
    [[0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 2.0, 3.0, 0.9]]

    ```

    Inspect points and distances:

    ```pycon
    >>> names = t.get_point_names()
    >>> sorted(names)[:3]
    ['p1', 'p2', 'p3']
    >>> d = t.distance_between('p1', 'p2')
    >>> len(d) == len(t.data)
    True

    ```

    Filter low-likelihood positions and interpolate:

    ```pycon
    >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
    ...     t2 = Tracking.from_dlc(str(p), handle='ex2', fps=30)
    >>> _ = t2.filter_likelihood(0.2)
    >>> import numpy as np
    >>> bool(np.isnan(t2.data['p1.x']).any())
    True
    >>> _ = t2.interpolate(method='nearest', limit=1)
    >>> has_lik = t2.data.columns.str.endswith('.likelihood').any()
    >>> interp_ok = t2.meta['interpolation']['method'] == 'nearest'
    >>> has_lik and interp_ok
    True

    ```

    Smooth all points with default window=3 rolling mean, and optional exception for point 'p1':

    ```pycon
    >>> _ = t.smooth_all(3, 'mean',[(['p1'],'median',4)])
    >>> 'smoothing' in t.meta
    True

    ```

    Rescale by known distance between two points (uniform across dims):

    ```pycon
    >>> _ = t.rescale_by_known_distance('p1', 'p2', distance_in_metres=2.0)
    >>> t.meta['distance_units']
    'm'

    ```

    Trim frames and verify time window:

    ```pycon
    >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
    ...     t3 = Tracking.from_dlc(str(p), handle='ex3', fps=30)
    >>> _ = t3.trim(startframe=2, endframe=4)
    >>> bool(t3.data.index[0] == 2 and t3.data.index[-1] == 4)
    True
    >>> bool(t3.time_as_expected(mintime=0.0, maxtime=10.0))
    True

    ```

    Save to a directory (parquet backend) and load back:

    ```pycon
    >>> import os, tempfile
    >>> with tempfile.TemporaryDirectory() as d:
    ...     _ = t.save(d, data_format='csv',overwrite=True)
    ...     t_loaded = Tracking.load(d)
    >>> isinstance(t_loaded, Tracking) and len(t_loaded.data) == len(t.data)
    True

    ```

    Slice with loc/iloc and keep handle:

    ```pycon
    >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
    ...     t4 = Tracking.from_dlc(str(p), handle='ex4', fps=30)
    >>> t4s = t4.loc[0:3]
    >>> isinstance(t4s, Tracking) and t4s.handle == 'ex4'
    True
    >>> t4s2 = t4.iloc[0:2]
    >>> isinstance(t4s2, Tracking) and len(t4s2.data) == 2
    True

    ```


    Minimal plotting (no display):

    ```pycon
    >>> _ = t.plot(show=False)

    ```

    Tagging and user metadata:

    ```pycon
    >>> t.add_tag('session', 'S1')
    >>> t.tags['session']
    'S1'
    >>> t.add_usermeta({'group': 'G1'}, overwrite=True)
    >>> t.meta['usermeta']['group']
    'G1'

    ```
    """

    data: pd.DataFrame
    meta: dict
    handle: str
    tags: dict[str, str]

    @classmethod
    def from_dlc(
        cls: type[Self],
        filepath: str | Path,
        *,
        handle: str,
        fps: float,
        aspectratio_correction: float = 1.0,
        tags: dict[str, str] | None = None,
    ) -> Self:
        """
        loads a Tracking object from a (single animal) deeplabcut tracking csv

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> len(t.data), t.meta['fps'], t.handle
        (5, 30.0, 'ex')

        ```
        """
        # normalize path
        filepath = Path(filepath)
        # read header
        header = pd.read_csv(filepath, header=None, nrows=3)
        cols = [
            ".".join(i)
            for i in zip(
                list(header.iloc[1, 1:].astype(str)),
                list(header.iloc[2, 1:].astype(str)),
                strict=True,
            )
        ]
        scorer = header.iloc[0, 1]

        # setup data
        data = pd.read_csv(filepath, skiprows=3, header=None)
        data.set_index(0, inplace=True)
        data.index.rename("frame", inplace=True)
        data.columns = cols

        meta = {
            "filepath": str(filepath),
            "fps": float(fps),
            "aspectratio_correction": float(aspectratio_correction),
            "network": scorer,
        }

        data = cls._apply_aspectratio_correction(data, float(aspectratio_correction))

        return cls(data, meta, handle, tags)

    @classmethod
    def from_dlcma(
        cls: type[Self],
        filepath: str | Path,
        *,
        handle: str,
        fps: float,
        aspectratio_correction: float = 1.0,
        tags: dict[str, str] | None = None,
    ) -> Self:
        """
        loads a Tracking object from a multi-animal deeplabcut tracking csv

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlcma_multi.csv') as p:
        ...     t = Tracking.from_dlcma(str(p), handle='ma', fps=30)
        >>> len(t.data), t.meta['fps'], t.handle
        (4, 30.0, 'ma')

        ```
        """
        # normalize path
        filepath = Path(filepath)
        # read header
        header = pd.read_csv(filepath, header=None, nrows=4)
        cols = [
            ".".join(i)
            for i in zip(
                list(header.iloc[1, 1:].astype(str)),
                list(header.iloc[2, 1:].astype(str)),
                list(header.iloc[3, 1:].astype(str)),
                strict=True,
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
            "filepath": str(filepath),
            "fps": float(fps),
            "aspectratio_correction": float(aspectratio_correction),
            "network": scorer,
        }

        data = cls._apply_aspectratio_correction(data, float(aspectratio_correction))

        return cls(data, meta, handle, tags)

    @classmethod
    def from_yolo3r(
        cls: type[Self],
        filepath: str | Path,
        *,
        handle: str,
        fps: float,
        aspectratio_correction: float = 1.0,
        tags: dict[str, str] | None = None,
    ) -> Self:
        """
        loads a Tracking object from a single- or multi-animal yolo csv in 3R hub format

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'yolo3r.csv') as p:
        ...     t = Tracking.from_yolo3r(str(p), handle='y3r', fps=30)
        >>> 'p1.z' in t.data.columns and 'p1.likelihood' in t.data.columns
        True

        ```
        """
        # normalize path
        filepath = Path(filepath)
        # setup data
        data = pd.read_csv(filepath, index_col="frame_index")
        data.index.rename("frame", inplace=True)
        newcols = [re.sub(".conf$", ".likelihood", col) for col in data.columns]
        data.columns = newcols

        # drop only bounding-box corner coordinates; keep everything else.
        # remove columns ending with .x1, .y1, .x2, .y2 and their .likelihood pairs
        drop_column_suffixes = (".x1", ".y1", ".x2", ".y2")
        bbox_cols = [col for col in data.columns if col.endswith(drop_column_suffixes)]
        if bbox_cols:
            bbox_bases = {col.rsplit(".", 1)[0] for col in bbox_cols}
            likelihood_to_drop = [
                f"{base}.likelihood" for base in bbox_bases if f"{base}.likelihood" in data.columns
            ]
            to_drop = list(set(bbox_cols).union(likelihood_to_drop))
            data.drop(columns=to_drop, inplace=True)

        # drop max_dim columns
        max_dim_cols = [col for col in data.columns if col == "max_dim.x" or col == "max_dim.y"]
        data.drop(columns=max_dim_cols, inplace=True)

        meta = {
            "filepath": str(filepath),
            "fps": float(fps),
            "aspectratio_correction": float(aspectratio_correction),
        }

        data = cls._apply_aspectratio_correction(data, float(aspectratio_correction))

        return cls(data, meta, handle, tags)

    def copy(self) -> Self:
        """Creates a copy of an existing tracking object

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlcma_multi.csv') as p:
        ...     t = Tracking.from_dlcma(str(p), handle='ma', fps=30)
        >>> t_copy = t.copy()
        >>> len(t_copy.data), t_copy.meta['fps'], t_copy.handle
        (4, 30.0, 'ma')

        ```
        """
        return type(self)(
            data=self.data.copy(),
            meta=copy.deepcopy(self.meta),
            handle=self.handle,
            tags=copy.deepcopy(self.tags),
        )

    @classmethod
    def concat(
        cls: type[Self],
        trackings: list[Self],
        *,
        handle: str | None = None,
        reindex: Literal["rezero", "follow_previous", "keep_original"] = "follow_previous",
    ) -> Self:
        """
        Concatenate multiple Tracking objects along the time (frame) axis.

        All Tracking objects must have:
        - Matching fps
        - Identical column names (same tracked points and dimensions)

        Parameters
        ----------
        trackings : list[Tracking]
            List of Tracking objects to concatenate, in temporal order.
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
        Tracking
            A new Tracking object containing all frames from input objects.

        Raises
        ------
        ValueError
            If trackings is empty, fps values don't match, or columns differ.

        Examples
        --------
        Concatenate two tracking objects:

        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t1 = Tracking.from_dlc(str(p), handle='ex1', fps=30)
        ...     t2 = Tracking.from_dlc(str(p), handle='ex2', fps=30)
        >>> combined = Tracking.concat([t1, t2], handle='combined')
        >>> len(combined.data) == len(t1.data) + len(t2.data)
        True
        >>> combined.handle
        'combined'
        >>> combined.meta['fps']
        30.0

        ```

        Verify column preservation:

        ```pycon
        >>> list(combined.data.columns) == list(t1.data.columns)
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
        if not trackings:
            raise ValueError("Cannot concatenate empty list of Tracking objects")

        if len(trackings) == 1:
            result = trackings[0].copy()
            if handle is not None:
                result.handle = handle
            return result

        # Keys in meta that are expected to differ between chunks (not validated)
        _meta_ignore_keys = {"filepath", "concat"}

        # Validate fps consistency
        fps_values = [t.meta["fps"] for t in trackings]
        if len(set(fps_values)) > 1:
            raise ValueError(f"All Tracking objects must have the same fps. Got: {fps_values}")

        # Validate column consistency
        reference_cols = list(trackings[0].data.columns)
        for i, t in enumerate(trackings[1:], start=1):
            if list(t.data.columns) != reference_cols:
                raise ValueError(
                    f"Column mismatch: Tracking[0] has columns {reference_cols}, "
                    f"but Tracking[{i}] has columns {list(t.data.columns)}"
                )

        # Validate meta consistency (excluding ignored keys)
        ref_meta = trackings[0].meta
        for i, t in enumerate(trackings[1:], start=1):
            ref_keys = set(ref_meta.keys()) - _meta_ignore_keys
            t_keys = set(t.meta.keys()) - _meta_ignore_keys
            if ref_keys != t_keys:
                raise ValueError(
                    f"Meta key mismatch: Tracking[0] has keys {ref_keys}, "
                    f"but Tracking[{i}] has keys {t_keys}"
                )
            for key in ref_keys:
                if ref_meta[key] != t.meta[key]:
                    raise ValueError(
                        f"Meta value mismatch for key '{key}': "
                        f"Tracking[0] has {ref_meta[key]!r}, "
                        f"but Tracking[{i}] has {t.meta[key]!r}"
                    )

        # Check handle consistency - warn if differs, use first
        handles = [t.handle for t in trackings]
        if len(set(handles)) > 1 and handle is None:
            warnings.warn(
                f"Handles differ across Tracking objects: {handles}. "
                f"Using first handle '{handles[0]}'. "
                f"Pass handle= parameter to specify explicitly.",
                stacklevel=2,
            )

        # Check tags consistency - warn if differs, use first
        first_tags = trackings[0].tags
        tags_differ = any(t.tags != first_tags for t in trackings[1:])
        if tags_differ:
            warnings.warn(
                f"Tags differ across Tracking objects. Using tags from first object: {first_tags}",
                stacklevel=2,
            )

        # Build concatenated DataFrame with adjusted indices
        dfs = []
        chunk_boundaries = []
        # For "follow_previous", start from first object's starting index
        # For "rezero", start from 0
        if reindex == "rezero":
            current_offset = 0
        else:
            current_offset = trackings[0].data.index[0]

        for i, t in enumerate(trackings):
            df = t.data.copy()
            original_start = df.index[0]
            original_end = df.index[-1]
            n_frames = len(df)

            if reindex == "rezero":
                # Contiguous reindexing starting from 0
                df.index = pd.RangeIndex(current_offset, current_offset + n_frames)
            elif reindex == "follow_previous":
                # Each chunk continues from previous end + 1
                df.index = pd.RangeIndex(current_offset, current_offset + n_frames)
            # else reindex == "keep_original": leave df.index untouched

            chunk_boundaries.append(
                {
                    "chunk_index": i,
                    "original_handle": t.handle,
                    "original_start_frame": int(original_start),
                    "original_end_frame": int(original_end),
                    "concat_start_frame": int(df.index[0]),
                    "concat_end_frame": int(df.index[-1]),
                    "n_frames": n_frames,
                }
            )

            dfs.append(df)
            # Update offset for next chunk (only matters for rezero/follow_previous)
            current_offset = df.index[-1] + 1

        combined_data = pd.concat(dfs, axis=0)
        combined_data.index.name = "frame"

        # Build metadata (from first, add concat info)
        combined_meta = copy.deepcopy(trackings[0].meta)
        combined_meta["concat"] = {
            "n_chunks": len(trackings),
            "chunk_boundaries": chunk_boundaries,
            "reindexed": reindex,
            "source_handles": handles,
        }

        # Use first object's tags
        combined_tags = copy.deepcopy(first_tags)

        result_handle = handle if handle is not None else trackings[0].handle

        return cls(combined_data, combined_meta, result_handle, combined_tags)

    @staticmethod
    def _apply_aspectratio_correction(df: pd.DataFrame, correction: float) -> pd.DataFrame:
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
        meta: dict[str, Any],
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

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> t.add_usermeta({'group': 'G1'}, overwrite=True)
        >>> t.meta['usermeta']['group']
        'G1'

        ```
        """
        if not isinstance(usermeta, dict):
            raise TypeError(f"usermeta must be a dictionary, got {type(usermeta).__name__}")

        if "usermeta" in self.meta and not overwrite:
            raise Exception("user defined metadata already stored, set overwrite=True to overwrite")

        self.meta["usermeta"] = usermeta
        if overwrite:
            warnings.warn("usermeta may be overwritten", stacklevel=2)

    def add_tag(self, tagname: str, tagvalue: str, overwrite: bool = False) -> None:
        """
        adds or updates a tag

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> t.add_tag('session', 'S1', overwrite=True)
        >>> t.tags['session']
        'S1'

        ```
        """
        if not isinstance(tagname, str):
            raise TypeError(f"tagname must be a string, got {type(tagname).__name__}")
        if tagname in self.tags and not overwrite:
            raise Exception(f"tag {tagname} already exists, set overwrite=True to overwrite")
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

        Examples
        --------
        ```pycon
        >>> import tempfile, os
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     t.save(d, data_format='csv', overwrite=True)
        ...     os.path.exists(os.path.join(d, 'manifest.json'))
        True

        ```
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
    def load(cls: type[Self], dirpath: str) -> Self:
        """
        Load a Tracking (or subclass) previously saved with save().

        Examples
        --------
        ```pycon
        >>> import tempfile
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     t.save(d, data_format='csv', overwrite=True)
        ...     t2 = Tracking.load(d)
        >>> isinstance(t2, Tracking) and len(t2.data) == len(t.data)
        True

        ```
        """
        manifest = read_manifest(dirpath)
        df = read_dataframe(dirpath, manifest["data"])
        handle = manifest["handle"]
        meta = manifest["meta"]
        tags = manifest.get("tags", {})
        return cls(df, meta, handle, tags)

    def strip_column_names(self, *, inplace: bool = True) -> Tracking | None:
        """strip column names to the last two dot-delimited sections

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> before = list(t.data.columns)[:3]
        >>> t.strip_column_names()
        >>> after = list(t.data.columns)[:3]
        >>> all(len(c.split('.')) == 2 for c in after)
        True

        ```
        """
        if not inplace:
            new = self.copy()
            new.strip_column_names(inplace=True)
            return new
        stripped_colnames = [".".join(col.split(".")[-2:]) for col in self.data.columns]
        self.data.columns = stripped_colnames

    def time_as_expected(self, mintime: float, maxtime: float) -> bool:
        """
        check that total tracking duration (seconds) is between mintime and maxtime.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> bool(t.time_as_expected(0.0, 1.0)) # between 0 and 1 second
        True
        >>> bool(t.time_as_expected(0.0, 0.1)) # less than 0.1 seconds
        False

        ```
        """
        if "trim" in self.meta.keys():
            warnings.warn("tracking data have been trimmed", stacklevel=2)
        totalframes = self.data.index[-1] - self.data.index[0]
        totaltime = totalframes / self.meta["fps"]

        return (mintime <= totaltime) & (maxtime >= totaltime)

    def trim(
        self,
        startframe: int | None = None,
        endframe: int | None = None,
        *,
        inplace: bool = True,
    ) -> Tracking | None:
        """
        trims the tracking data object between startframe and endframe

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> _ = t.trim(1, 3)
        >>> int(t.data.index[0]), int(t.data.index[-1])
        (1, 3)

        ```
        """
        if startframe is not None:
            if (self.data.index[0] > startframe) or (self.data.index[-1] < startframe):
                raise Exception("startframe not in data")
        if endframe is not None:
            if endframe < 0:
                endframe = self.data.index[-1] + endframe
            if (self.data.index[0] > endframe) or (self.data.index[-1] < endframe):
                raise Exception("endframe not in data")

        if not inplace:
            new = self.copy()
            new.trim(startframe, endframe, inplace=True)
            return new
        datatrim = self.data.loc[startframe:endframe, :].copy()
        self.data = datatrim
        self.meta["trim"] = {"startframe": startframe, "endframe": endframe}

    def filter_likelihood(self, threshold: float, *, inplace: bool = True) -> Tracking | None:
        """set position values with likelihood below threshold to np.nan.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> import numpy as np
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> t.filter_likelihood(0.5)
        >>> bool(np.isnan(t.data.filter(like='.x')).any().any())
        True
        >>> bool(np.isnan(t.data['p1.x'].values[-1]))
        True
        >>> float(t.data['p1.likelihood'].values[0])
        1.0

        ```
        """
        if "filter_likelihood_threshold" in self.meta.keys():
            raise Exception("likelihood already filtered. re-load the raw data to change filter.")
        if "smoothing" in self.meta.keys():
            warnings.warn(
                "these data have been smoothed. you should filter likelihood before smoothing",
                stacklevel=2,
            )

        if not inplace:
            new = self.copy()
            new.filter_likelihood(threshold, inplace=True)
            return new
        for point in self.get_point_names():
            df = self.get_point_data(point)
            df = filter_by_threshold(df=df, reference_col="likelihood", threshold=threshold)
            self.set_point_data(df, point)

        self.meta["filter_likelihood_threshold"] = threshold

    def distance_between(self, point1: str, point2: str, dims=("x", "y")) -> pd.Series:
        """framewise distance between two points

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> d = t.distance_between('p1', 'p2')
        >>> len(d) == len(t.data)
        True

        ```
        """
        distance = euclidean_distance(
            self.get_point_data(point1, dims),
            self.get_point_data(point2, dims),
            method="element_wise",
        )
        assert isinstance(distance, pd.Series)  # euclidean_distance can return float or pd.Series!
        return distance

    def get_point_names(self) -> list:
        """list of tracked point names, sorted alphabetically (ascending)

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> names = t.get_point_names()
        >>> set(['p1','p2','p3']).issubset(names)
        True
        >>> sorted_names = sorted(names)
        >>> sorted_names == names
        True

        ```
        """
        tracked_points = list(set([".".join(i.split(".")[:-1]) for i in self.data.columns]))
        tracked_points.sort()
        return tracked_points

    def _assert_valid_point(self, point: str):
        """Return viable dimension names associated with a point.

        Examples
        --------
        ```pycon
        >>> import pytest
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> t._assert_valid_point('p1')
        >>> with pytest.raises(KeyError):
        ...     t._assert_valid_point("nonexisting")

        ```
        """
        valid_points = self.get_point_names()
        if point not in valid_points:
            raise KeyError(f"point {point} not in tracking data; valid: {valid_points}")

    def get_point_dimensions(self, point: str) -> list[str]:
        """Return viable dimension names associated with a point.

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> t.get_point_dimensions('p1')
        ['x', 'y', 'z', 'likelihood']

        ```
        """

        self._assert_valid_point(point)
        prefix = f"{point}."
        dimensions = self.data.columns.str.removeprefix(prefix)[
            self.data.columns.str.startswith(prefix)
        ]
        return list(dimensions)

    def get_point_data(self, point: str, dims: Iterable[str] | None = None) -> pd.DataFrame:
        """For a specific point, returns the DataFrame with all dimensions data.
        colnames are reformated to drop the pointname (i.e p1.x -> x)

        Args:
            point (str): name of the point for which data should be exteracted
            dims (optional(tuple(str))): dimensons which should exclusively be returned

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> df = t.get_point_data('p1')
        >>> df['x'].values
        array([0, 1, 2, 3, 4])

        ```
        """
        self._assert_valid_point(point)
        prefix = f"{point}."
        df = self.data.loc[:, self.data.columns.str.startswith(prefix)].copy()
        df.columns = df.columns.str.removeprefix(prefix)

        if dims is not None:
            return df.loc[:, dims]

        return df

    def set_point_data(self, df: pd.DataFrame, point: str, target_df: pd.DataFrame = None):
        """Sets the data of a point to the values of an external df.

        Args:
            df (pd.DataFrame):  the dataframe containing the point data that should be
                writen into the trackingobject. colnames should reflect the dimension
                name (i.e 'x', 'y' etc.)
            point (str): the name of the point to overwrite
            target_df (pd.DataFrame, Optional): An external copy of the self.data
                dataframe can be specified. Usefull for operations that are not in
                place. defaults to None = write into self.data


        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> df = t.get_point_data('p1')
        >>> df['x'].values
        array([0, 1, 2, 3, 4])
        >>> df['x'] += 1
        >>> t.set_point_data(df,'p1')
        >>> t.data['p1.x'].values
        array([1, 2, 3, 4, 5])

        >>> external_df = t.data.copy()
        >>> external_df['p1.x'].values
        array([1, 2, 3, 4, 5])

        >>> df['x'] += 1
        >>> t.set_point_data(df = df, point = 'p1', target_df = external_df)
        >>> external_df['p1.x'].values
        array([2, 3, 4, 5, 6])

        ```
        """
        self._assert_valid_point(point)

        if target_df is None:
            target_df = self.data

        prefix = f"{point}."
        target_cols = target_df.columns[target_df.columns.str.startswith(prefix)]
        point_dimensions = self.get_point_dimensions(point)
        original_shape = len(target_df), len(target_cols)
        if df.shape != original_shape:
            raise ValueError(
                f"Shape mismatch between input df {df.shape} with dimensions "
                f"{df.columns} and target point data {original_shape} with "
                f"dimensions {point_dimensions}"
            )
        if list(df.columns) != point_dimensions:
            raise ValueError(
                f"Dimension names of df {list(df.columns)} do not match point "
                f"{point} dimensions {point_dimensions}"
            )
        if not df.index.equals(target_df.index):
            raise ValueError("Index mismatch between input df and target data")

        # Assign with guaranteed column order
        target_df.loc[:, target_cols] = df.to_numpy()

    def rescale_by_known_distance(
        self,
        point1: str,
        point2: str,
        distance_in_metres: float,
        dims=("x", "y"),
        *,
        inplace: bool = True,
    ) -> Tracking | None:
        """rescale all dims by known distance between two points

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> t.rescale_by_known_distance('p1','p2', 2.0)
        >>> t.meta['distance_units']
        'm'

        ```
        """
        if "rescale_distance_method" in self.meta.keys():
            if self.meta["rescale_distance_method"] == "two_point_scalar_uniform":
                if any(d in self.meta["rescale_factor"].keys() for d in dims):
                    raise Exception(
                        "distance already rescaled in this dim. re-load the raw "
                        "data to change scaling"
                    )
            else:
                raise Exception("distance already rescaled. re-load the raw data to change scaling")

        if not inplace:
            new = self.copy()
            new.rescale_by_known_distance(
                point1, point2, distance_in_metres, dims=dims, inplace=True
            )
            return new

        tracking_distance = euclidean_distance(
            self.get_point_data(point1, dims),
            self.get_point_data(point2, dims),
            method="median",
        )
        assert isinstance(
            tracking_distance, float
        )  # euclidean_distance can return float or pd.Series!
        if tracking_distance == 0:
            raise Exception(f"observed distance between '{point1}' and '{point2}' is 0")
        if np.isnan(tracking_distance):
            raise Exception(f"observed distance between '{point1}' and '{point2}' is NaN")

        rescale_factor = distance_in_metres / tracking_distance

        tracked_points = self.get_point_names()

        for point in tracked_points:
            df = self.get_point_data(point)
            df = scale_columns(df, rescale_factor, dims)
            self.set_point_data(df, point)

        self.meta["rescale_distance_method"] = "two_point_scalar_uniform"
        self.meta["rescale_factor"] = {dim: rescale_factor for dim in dims}
        self.meta["distance_units"] = "m"

    def _generate_partial_smoothdict(self, points: list, window: int, smoothtype: str) -> dict:
        """make partial smoothdict for points"""
        smoothdict = dict()
        for key in points:
            smoothdict[key] = {"window": window, "type": smoothtype}
        return smoothdict

    def generate_smoothdict(self, pointslists: list, windows: list, smoothtypes: list) -> dict:
        """
        deprecated, use smooth_all instead
        """
        raise NotImplementedError(
            "Tracking.generate_smoothdict() was removed; use Tracking.smooth_all() instead."
        )

    def smooth(self, smoothing_params: dict) -> None:
        """
        deprecated, use smooth_all instead
        """
        raise NotImplementedError(
            "Tracking.smooth() was removed; use Tracking.smooth_all() instead."
        )

    def smooth_all(
        self,
        window: int | None = 11,
        method: Literal["mean", "median", "savgol"] = "savgol",
        overrides: list[tuple[list[str] | tuple[str, ...] | str, str, int | None]] | None = None,
        dims: tuple[str, ...] = ("x", "y"),
        strict: bool = False,
        inplace: bool = True,
        smoother=None,
        smoother_kwargs: dict | None = None,
        method_kwargs: dict | None = None,
        **kwargs,
    ) -> Tracking | None:
        """
        Smooth all tracked points using a default method/window, with optional
        override groups.

        - window/method: default applied to any point without override
        - overrides: optional list of (points, method, window) tuples, where
            - points: list/tuple of point names (or a single str)
            - method: 'median' or 'mean'
            - window: int (or None to skip smoothing for those points)
        - dims: coordinate dimensions to smooth
        - strict: require an effective window for every point
        - inplace: mutate or return a new object

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> t.smooth_all(3, 'mean', overrides=[(['p1'], 'median', 4)])
        >>> 'smoothing' in t.meta
        True

        ```
        """
        # Normalize override groups into a point->spec dict
        overrides_dict: dict[str, dict] = {}
        if overrides:
            for grp in overrides:
                if not (isinstance(grp, tuple) and len(grp) == 3):
                    raise ValueError("each override must be a tuple: (points, method, window)")
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
        # Apply global per-method kwargs (e.g., nan_policy for savgol) if provided
        if method_kwargs:
            for p in specs:
                for k, v in method_kwargs.items():
                    if k not in specs[p]:
                        specs[p][k] = v
        # Also merge any free-form kwargs (e.g., polyorder=3) for convenience,
        # allowing callers (including batch mixins) to pass smoothing params directly.
        if kwargs:
            for p in specs:
                for k, v in kwargs.items():
                    if k not in specs[p]:
                        specs[p][k] = v
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
            raise Exception("data already smoothed. load again to use different smoothing")
        if method not in {"median", "mean", "savgol"}:
            raise ValueError("method must be one of {'median','mean','savgol'}")
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
                    f"Invalid override for {p}: expected dict with keys "
                    f"'method'/'window', got {type(spec)}"
                )
            if strict and (w is None or w <= 0):
                raise ValueError(f"No valid window resolved for point '{p}' with strict=True")
            specs[p] = {"method": m, "window": None if not w or w <= 0 else int(w)}
        return specs

    def _build_smoothing_meta(self, specs: dict[str, dict], dims: tuple[str, ...]) -> dict:
        return {"spec": specs, "dims": list(dims)}

    def interpolate(
        self, method: str = "linear", limit: int = 1, *, inplace: bool = True, **kwargs
    ) -> Tracking | None:
        """
        interpolates missing data in the tracking data, and sets likelihood to np.nan
        uses pandas.DataFrame.interpolate() with kwargs

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> import numpy as np
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> t.filter_likelihood(0.5)
        >>> t.interpolate(method='linear', limit=1)
        >>> 'interpolation' in t.meta
        True

        ```
        """
        if "interpolation" in self.meta.keys():
            raise Exception("data already interpolated. re-load the raw data to interpolate again")

        if not inplace:
            new = self.copy()
            new.interpolate(method=method, limit=limit, inplace=True, **kwargs)
            return new
        # interpolate only the position columns, and set likelihood to np.nan
        position_columns = self.data.columns[
            self.data.columns.str.endswith(".x")
            | self.data.columns.str.endswith(".y")
            | self.data.columns.str.endswith(".z")
        ]
        self.data.loc[:, position_columns] = self.data.loc[:, position_columns].interpolate(
            method=method, limit=limit, **kwargs
        )
        self.data.loc[:, self.data.columns.str.endswith(".likelihood")] = np.nan

        self.meta["interpolation"] = {
            "method": method,
            "limit": limit,
            "kwargs": kwargs,
        }

    @property
    def loc(self):
        """
        Return a new Tracking object with self.data sliced by np.loc

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> t.data.shape
        (5, 12)
        >>> t.loc[0:2,'p1.x'].data.shape
        (3,)
        >>> t.loc[0:2].handle
        'ex'

        ```
        """
        return _Indexer(self, self._loc)

    @property
    def iloc(self):
        """
        Return a new Tracking object with self.data sliced by np.iloc

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> t.data.shape
        (5, 12)
        >>> t.iloc[0:2,0].data.shape
        (2,)
        >>> t.iloc[0:2,0].handle
        'ex'

        ```
        """
        return _Indexer(self, self._iloc)

    def _loc(self, idx):
        new_data = self.data.loc[idx].copy()
        new_meta = copy.deepcopy(self.meta)
        return self.__class__(new_data, new_meta, self.handle, tags=copy.deepcopy(self.tags))

    def _iloc(self, idx):
        new_data = self.data.iloc[idx].copy()
        new_meta = copy.deepcopy(self.meta)
        return self.__class__(new_data, new_meta, self.handle, tags=copy.deepcopy(self.tags))

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
        savedir: str | None = None,
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
            savedir: optional directory path to save the plot image. If provided,
                     figure is saved as '<handle>_plot.png' inside this directory.
        Returns: fig, ax

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> _ = t.plot(show=False)

        ```
        """
        import numpy as np

        is3d = len(dims) == 3
        if len(dims) > 3:
            raise ValueError("dims must be a tuple of length 2 or 3")
        created_fig = False
        if ax is None:
            created_fig = True
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
            if isinstance(trajectories, dict) and isinstance(trajectories[point], pd.Series):
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
            # safe median without warnings on all-NaN slices
            med = []
            for d in dims:
                arr = self.data[f"{point}.{d}"].to_numpy()
                finite = arr[np.isfinite(arr)]
                med.append(float(np.median(finite)) if finite.size > 0 else np.nan)
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

        # place legend to the right of the axes
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=False,
        )
        try:
            fig.tight_layout()
        except Exception:
            pass

        # Enforce 1:1 aspect ratio for 2D plots
        if not is3d:
            try:
                ax.set_aspect("equal", adjustable="box")
            except Exception:
                pass
        # Optional save to disk, named by handle
        if savedir is not None:
            import os

            os.makedirs(savedir, exist_ok=True)
            out_path = os.path.join(savedir, f"{self.handle}_plot.png")
            fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        if show:
            plt.show()
        # Close figure if we created it and we're not showing,
        # to avoid accumulating open figures
        if created_fig and not show:
            import matplotlib.pyplot as _plt

            _plt.close(fig)
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
        Save a 3D animation of tracked points to a video file, with 4 subplots
        per frame:
        - azim=0, elev=0, ortho
        - azim=90, elev=0, ortho
        - azim=0, elev=90, ortho
        - azim=45, elev=30, persp
        Optionally, set axis limits manually or use robust percentiles to
        ignore outliers. Enforces equal aspect ratio for all axes.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import animation
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
            frame_iter = tqdm(selected_frames, desc="Precomputing 3D coordinates", unit="frame")
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
            if not use_tqdm and total_frames > 0 and idx % max(1, total_frames // 10) == 0:
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
        for ax, (elev, azim, proj_type), title in zip(axs, views, titles, strict=True):
            ax.view_init(elev=elev, azim=azim)
            ax.set_proj_type(proj_type)
            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            scatters.append(ax.scatter([], [], [], s=point_size, c=point_color))
            line_objs.append(
                [ax.plot([], [], [], color=line_color, linewidth=line_width)[0] for _ in lines]
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
            xs, ys, zs = zip(*coords.values(), strict=True) if coords else ([], [], [])
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

    def animation_stream(
        self,
        *,
        points: list[str],
        lines: list[tuple[str, str]] | None = None,
        features: list[str | None] | dict[str | None, str | None] | None = None,
        dims: tuple[str, ...] = ("x", "y"),
        view: dict | None = None,
        canvas_size: tuple[int, int] = (800, 800),
        bg_color: tuple[int, int, int] = (0, 0, 0),
        style: dict | None = None,
        pixel_coords: bool = False,
        undo_meta_scaling: bool = False,
    ):
        """
        Build an OpenCV-backed frame stream for animated point/line overlays.

        This method precomputes the selected point coordinates (and optional 3D
        projection) once, then returns a stream object that can:

        - fetch individual rendered frames via ``get_frame(i)``
        - iterate sequentially via ``read()`` / ``next()``
        - play live via ``stream.play(...)``
        - save video via ``stream.save(...)``

        Parameters
        ----------
        points : list[str]
            Point names to render as circles.
        lines : list[tuple[str, str]], optional
            Line segments connecting point pairs. Endpoints can include points
            not listed in ``points``.
        features : list[str | None] | dict[str | None, str | None], optional
            Per-frame scalar columns to render as text overlays. If a list is
            provided, each column is shown as ``name: value``. If a dict is
            provided, keys are display labels and values are source column names.
            ``None`` or ``""`` entries insert a blank spacer line.
        dims : tuple[str, ...], default ("x", "y")
            Coordinate dimensions. Use 2D (``("x","y")``) or 3D
            (``("x","y","z")`` with ``view``).
        view : dict, optional
            3D camera options used only when ``dims`` has length 3.
            Supported keys include ``azim``, ``elev``, ``proj`` (``"ortho"``
            or ``"persp"``), ``camera_distance``, ``focal_length``, and
            ``pad``.
        canvas_size : tuple[int, int], default (800, 800)
            Canvas size as ``(width, height)``.
        bg_color : tuple[int, int, int], default (0, 0, 0)
            Background color in BGR.
        style : dict, optional
            Style overrides for points/lines/boundaries.
        pixel_coords : bool, default False
            If True, interpret coordinates as absolute pixel locations.
            If False, auto-fit projected coordinates to the canvas.
        undo_meta_scaling : bool, default False
            If True, invert ``aspectratio_correction`` and
            ``meta["rescale_factor"]`` before rendering.

        Returns
        -------
        GeometryAnimationStream
            Stream object with ``get_frame()``, ``read()``, ``play()``, and
            ``save()``.

        Examples
        --------
        Build a simple 2D stream and render one frame:

        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> df = pd.DataFrame(
        ...     {
        ...         "nose.x": [10.0, 11.0, 12.0],
        ...         "nose.y": [20.0, 21.0, 22.0],
        ...         "neck.x": [8.0, 9.0, 10.0],
        ...         "neck.y": [19.0, 20.0, 21.0],
        ...     },
        ...     index=pd.Index([0, 1, 2], name="frame"),
        ... )
        >>> t = Tracking(df, meta={"fps": 30.0}, handle="demo")
        >>> stream = t.animation_stream(
        ...     points=["nose", "neck"],
        ...     lines=[("nose", "neck")],
        ...     pixel_coords=True,
        ...     canvas_size=(64, 48),
        ... )
        >>> stream.frame_count
        3
        >>> frame0 = stream.get_frame(0)
        >>> frame0.shape
        (48, 64, 3)

        ```

        Build a 3D projected stream:

        ```pycon
        >>> df3 = pd.DataFrame(
        ...     {
        ...         "p1.x": [0.0, 1.0],
        ...         "p1.y": [0.0, 0.0],
        ...         "p1.z": [0.0, 0.5],
        ...         "p2.x": [1.0, 2.0],
        ...         "p2.y": [0.0, 0.0],
        ...         "p2.z": [0.0, 0.5],
        ...     },
        ...     index=pd.Index([10, 11], name="frame"),
        ... )
        >>> t3 = Tracking(df3, meta={"fps": 30.0}, handle="demo3d")
        >>> s3 = t3.animation_stream(
        ...     points=["p1", "p2"],
        ...     lines=[("p1", "p2")],
        ...     dims=("x", "y", "z"),
        ...     view={"azim": 45, "elev": 30, "proj": "ortho"},
        ... )
        >>> s3.frame_count
        2

        ```
        """
        from py3r.behaviour.animation.geometry_stream import (
            build_geometry_stream_from_points,
            collect_dynamic_source_names_from_style,
        )

        line_points = {p for line in (lines or []) for p in line}
        all_points = sorted(set(points) | line_points)
        point_names, points_arr = self.points_to_numpy(
            all_points, dims=dims, undo_meta_scaling=undo_meta_scaling
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
                    raise ValueError(f"Column {col} not found for text overlay")
                text_overlays.append((str(label), self.data[col].to_numpy(copy=True)))
        style_sources = None
        if style is not None:
            needed = collect_dynamic_source_names_from_style(style)
            if needed:
                style_sources = {}
                for name in needed:
                    if name not in self.data.columns:
                        raise ValueError(f"Column {name} not found for dynamic style source")
                    style_sources[name] = self.data[name].to_numpy(copy=True)

        return build_geometry_stream_from_points(
            points=points_arr,
            point_names=point_names,
            draw_points=points,
            lines=lines,
            view=view,
            frame_ids=self.data.index.to_numpy(copy=True),
            fps=float(self.meta.get("fps", 30.0)),
            canvas_size=canvas_size,
            bg_color=bg_color,
            style=style,
            style_sources=style_sources,
            text_overlays=text_overlays,
            pixel_coords=pixel_coords,
            bounds_pad=float((view or {}).get("pad", 0.05)),
        )

    def points_to_numpy(
        self,
        points: list[str],
        dims: tuple[str, ...] = ("x", "y"),
        *,
        undo_meta_scaling: bool = False,
    ) -> tuple[list[str], np.ndarray]:
        """
        Resolve selected point coordinates to a NumPy array.

        Parameters
        ----------
        points : list[str]
            Point names to extract.
        dims : tuple[str, ...], default ("x", "y")
            Coordinate dimensions to extract (2D or 3D).
        undo_meta_scaling : bool, default False
            If True, invert ``aspectratio_correction`` and ``rescale_factor``
            before extraction.

        Returns
        -------
        tuple[list[str], np.ndarray]
            ``(point_names, array)`` where array has shape
            ``(n_frames, n_points, len(dims))``.

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> df = pd.DataFrame(
        ...     {
        ...         "nose.x": [1.0, 2.0],
        ...         "nose.y": [3.0, 4.0],
        ...         "tail.x": [5.0, 6.0],
        ...         "tail.y": [7.0, 8.0],
        ...     }
        ... )
        >>> t = Tracking(df, meta={"fps": 30.0}, handle="demo")
        >>> names, arr = t.points_to_numpy(["nose", "tail"], dims=("x", "y"))
        >>> names
        ['nose', 'tail']
        >>> arr.shape
        (2, 2, 2)

        ```
        """
        from py3r.behaviour.animation.geometry_stream import undo_meta_scaling_for_geometry

        source_df = (
            undo_meta_scaling_for_geometry(self.data, self.meta, dims=dims)
            if undo_meta_scaling
            else self.data
        )
        if len(points) == 0:
            return (
                [],
                np.empty(
                    (len(source_df), 0, len(dims)),
                    dtype=float,
                ),
            )
        point_arrays = []
        for point in points:
            cols = []
            for dim in dims:
                col = f"{point}.{dim}"
                if col not in source_df.columns:
                    raise ValueError(f"Column {col} not found in tracking data")
                cols.append(source_df[col].to_numpy(dtype=float, copy=True))
            point_arrays.append(np.column_stack(cols))
        return list(points), np.stack(point_arrays, axis=1)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        fps = self.meta.get("fps", "unknown")
        return f"<{cn} with {len(self.data)} rows, fps={fps}>"
