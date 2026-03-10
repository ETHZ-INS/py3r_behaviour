from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import pandas as pd

from py3r.behaviour.tracking.tracking import Tracking
from py3r.behaviour.tracking.tracking_mv import TrackingMV
from py3r.behaviour.util.base_collection import BaseCollection
from py3r.behaviour.util.collection_utils import _Indexer
from py3r.behaviour.util.dev_utils import dev_mode

if TYPE_CHECKING:
    from py3r.behaviour.features.features_collection import FeaturesCollection


class TrackingCollection(BaseCollection):
    """
    Collection of Tracking objects, keyed by name (e.g. for grouping individuals)
    note: type-hints refer to Tracking, but factory methods allow for other classes
    these are intended ONLY for subclasses of Tracking, and this is enforced
    """

    _element_type = Tracking
    each: Tracking
    each_forcebatch: Tracking

    def __init__(self, tracking_dict: dict[str, Tracking]):
        # Only validate handle mapping when values are leaf Tracking objects.
        # Grouped views (values are sub-collections) should skip this check.
        values = list(tracking_dict.values())
        if values and all(isinstance(v, Tracking) for v in values):
            for key, obj in tracking_dict.items():
                if obj.handle != key:
                    raise ValueError(f"Key '{key}' does not match object's handle '{obj.handle}'")
        super().__init__(tracking_dict)

    @property
    def tracking_dict(self):
        return self._obj_dict

    @classmethod
    def from_mapping(
        cls,
        handles_and_filepaths: dict[str, str],
        *,
        tracking_loader,
        tracking_cls=Tracking,
        **loader_kwargs,
    ):
        """
        Generic constructor from a mapping of handle -> filepath using a loader callable.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     # create two files for demonstration
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         f1 = d / 'a.csv'; f2 = d / 'b.csv'
        ...         _ = shutil.copy(p, f1); _ = shutil.copy(p, f2)
        ...     mapping = {'A': str(f1), 'B': str(f2)}
        ...     coll = TrackingCollection.from_mapping(
        ...         mapping, tracking_loader=Tracking.from_dlc, fps=30)
        >>> sorted(coll.keys())
        ['A', 'B']

        ```
        """
        if not issubclass(tracking_cls, Tracking):
            raise TypeError(f"tracking_cls must be Tracking or a subclass, got {tracking_cls}")
        trackings = {}
        for handle, fp in handles_and_filepaths.items():
            trackings[handle] = tracking_loader(fp, handle=handle, **loader_kwargs)
        return cls(trackings)

    each: Tracking

    @classmethod
    def from_dlc(
        cls,
        handles_and_filepaths: dict[str, str],
        *,
        fps: float,
        aspectratio_correction: float = 1.0,
        tracking_cls=Tracking,
    ):
        """
        Load a collection from DLC CSVs.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'a.csv'; b = d / 'b.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        >>> len(coll)
        2

        ```
        """
        return cls.from_mapping(
            handles_and_filepaths,
            tracking_loader=tracking_cls.from_dlc,
            tracking_cls=tracking_cls,
            fps=fps,
            aspectratio_correction=aspectratio_correction,
        )

    @classmethod
    def from_yolo3r(
        cls,
        handles_and_filepaths: dict[str, str],
        *,
        fps: float,
        aspectratio_correction: float = 1.0,
        tracking_cls=Tracking,
    ):
        """
        Load a collection from YOLO3R CSVs.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'yolo3r.csv') as p:
        ...         a = d / 'a.csv'; b = d / 'b.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_yolo3r({'A': str(a), 'B': str(b)}, fps=30)
        >>> set(coll.tracking_dict.keys()) == {'A','B'}
        True

        ```
        """
        return cls.from_mapping(
            handles_and_filepaths,
            tracking_loader=tracking_cls.from_yolo3r,
            tracking_cls=tracking_cls,
            fps=fps,
            aspectratio_correction=aspectratio_correction,
        )

    @classmethod
    def from_dlcma(
        cls,
        handles_and_filepaths: dict[str, str],
        *,
        fps: float,
        aspectratio_correction: float = 1.0,
        tracking_cls=Tracking,
    ):
        """
        Load a collection from DLC multi-animal CSVs.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlcma_multi.csv') as p:
        ...         a = d / 'a.csv'; b = d / 'b.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlcma({'A': str(a), 'B': str(b)}, fps=30)
        >>> len(coll) == 2
        True

        ```
        """
        return cls.from_mapping(
            handles_and_filepaths,
            tracking_loader=tracking_cls.from_dlcma,
            tracking_cls=tracking_cls,
            fps=fps,
            aspectratio_correction=aspectratio_correction,
        )

    @dev_mode
    @classmethod
    def from_dogfeather(
        cls,
        handles_and_filepaths: dict[str, str],
        *,
        fps: float,
        aspectratio_correction: float = 1.0,
        tracking_cls=Tracking,
    ):
        """
        Loads a TrackingCollection from a dict of dogfeather tracking csvs.
        handles_and_filepaths: dict mapping handles to file paths.
        """

        trackings = {}
        for handle, fp in handles_and_filepaths.items():
            trackings[handle] = tracking_cls.from_dogfeather(
                fp,
                handle=handle,
                fps=fps,
                aspectratio_correction=aspectratio_correction,
            )
        return cls(trackings)

    @classmethod
    def from_folder(
        cls,
        folder_path: str,
        *,
        tracking_loader,
        tracking_cls: type = Tracking,
        **loader_kwargs,
    ) -> TrackingCollection:
        """
        Build a collection by scanning a folder for CSVs (or multi-view subfolders).

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv')
        ...         _ = shutil.copy(p, d / 'B.csv')
        ...     coll = TrackingCollection.from_folder(
        ...         str(d), tracking_loader=Tracking.from_dlc, fps=30)
        >>> sorted(coll.keys())
        ['A', 'B']

        ```
        """
        tracking_dict = {}
        if issubclass(tracking_cls, TrackingMV):
            # Each subfolder is a multi-view recording; delegate to loader on the folder
            for recording in sorted(os.listdir(folder_path)):
                recording_path = os.path.join(folder_path, recording)
                if not os.path.isdir(recording_path):
                    continue
                tracking_obj = tracking_loader(recording_path, handle=recording, **loader_kwargs)
                tracking_dict[recording] = tracking_obj
        else:
            for fname in os.listdir(folder_path):
                if fname.endswith(".csv") and not fname.startswith("."):
                    handle = os.path.splitext(fname)[0]
                    fpath = os.path.join(folder_path, fname)
                    tracking_obj = tracking_loader(fpath, handle=handle, **loader_kwargs)
                    tracking_dict[handle] = tracking_obj
        return cls(tracking_dict)

    @classmethod
    def from_yolo3r_folder(
        cls,
        folder_path: str,
        *,
        fps: float,
        aspectratio_correction: float = 1.0,
        tracking_cls: type = Tracking,
    ) -> TrackingCollection:
        """
        Convenience for from_folder using YOLO3R loader.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'yolo3r.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv')
        ...         _ = shutil.copy(p, d / 'B.csv')
        ...     coll = TrackingCollection.from_yolo3r_folder(str(d), fps=30)
        >>> len(coll)
        2

        ```
        """
        return cls.from_folder(
            folder_path,
            tracking_loader=tracking_cls.from_yolo3r,
            tracking_cls=tracking_cls,
            fps=fps,
            aspectratio_correction=aspectratio_correction,
        )

    @classmethod
    def from_dlc_folder(
        cls,
        folder_path: str,
        *,
        fps: float,
        aspectratio_correction: float = 1.0,
        tracking_cls: type = Tracking,
    ) -> TrackingCollection:
        """
        Convenience for from_folder using DLC loader.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv')
        ...         _ = shutil.copy(p, d / 'B.csv')
        ...     coll = TrackingCollection.from_dlc_folder(str(d), fps=30)
        >>> set(coll.keys()) == {'A','B'}
        True

        ```
        """
        return cls.from_folder(
            folder_path,
            tracking_loader=tracking_cls.from_dlc,
            tracking_cls=tracking_cls,
            fps=fps,
            aspectratio_correction=aspectratio_correction,
        )

    @classmethod
    def from_dlcma_folder(
        cls,
        folder_path: str,
        *,
        fps: float,
        aspectratio_correction: float = 1.0,
        tracking_cls: type = Tracking,
    ) -> TrackingCollection:
        """
        Convenience for from_folder using DLCMA loader.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlcma_multi.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv')
        ...         _ = shutil.copy(p, d / 'B.csv')
        ...     coll = TrackingCollection.from_dlcma_folder(str(d), fps=30)
        >>> len(coll) == 2
        True

        ```
        """
        return cls.from_folder(
            folder_path,
            tracking_loader=tracking_cls.from_dlcma,
            tracking_cls=tracking_cls,
            fps=fps,
            aspectratio_correction=aspectratio_correction,
        )

    @classmethod
    def concat(
        cls,
        collections: list[TrackingCollection],
        *,
        reindex: Literal["rezero", "follow_previous", "keep_original"] = "follow_previous",
    ) -> TrackingCollection:
        """
        Concatenate multiple TrackingCollections along the time (frame) axis.

        Each collection must have the same handles (keys). For each handle,
        the corresponding Tracking objects are concatenated in order.
        Supports both flat and grouped collections.

        Parameters
        ----------
        collections : list[TrackingCollection]
            List of TrackingCollection objects to concatenate, in temporal order.
            All must have matching keys (handles).
        reindex : {"rezero", "follow_previous", "keep_original"}, default "follow_previous"
            How to handle frame indices:
            - "rezero": Reindex all frames starting from 0 (0, 1, 2, ...).
            - "follow_previous": Each chunk continues from where the previous
              ended. If chunk 1 ends at frame n, chunk 2 starts at n+1.
            - "keep_original": Leave indices untouched; duplicates are allowed.

        Returns
        -------
        TrackingCollection
            A new collection with concatenated Tracking objects for each handle.

        Raises
        ------
        ValueError
            If collections is empty, keys don't match, or grouping structure differs.

        Examples
        --------
        Concatenate two flat collections:

        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc1 = TrackingCollection.from_dlc({'A': str(d/'A.csv'),
        ...                                       'B': str(d/'B.csv')}, fps=30)
        ...     tc2 = TrackingCollection.from_dlc({'A': str(d/'A.csv'),
        ...                                        'B': str(d/'B.csv')}, fps=30)
        >>> combined = TrackingCollection.concat([tc1, tc2])
        >>> len(combined['A'].data) == len(tc1['A'].data) + len(tc2['A'].data)
        True
        >>> 'concat' in combined['A'].meta
        True

        ```
        """
        if not collections:
            raise ValueError("Cannot concatenate empty list of TrackingCollections")

        if len(collections) == 1:
            # Return a copy
            return cls({k: v.copy() for k, v in collections[0].items()})

        # Check grouping consistency
        is_grouped = [getattr(c, "is_grouped", False) for c in collections]
        if len(set(is_grouped)) > 1:
            raise ValueError(
                "Cannot concatenate mixed grouped/ungrouped collections. "
                f"Grouping states: {is_grouped}"
            )

        first = collections[0]

        if first.is_grouped:
            # Grouped collections: validate group keys match
            group_keys = [set(c.keys()) for c in collections]
            if not all(gk == group_keys[0] for gk in group_keys):
                raise ValueError(
                    f"Group key mismatch across collections. "
                    f"First has {group_keys[0]}, others have {group_keys[1:]}"
                )

            # For each group, validate handles match and concatenate
            result_dict = {}
            for group_key in first.keys():
                sub_collections = [c[group_key] for c in collections]
                # Validate handles within group
                handle_sets = [set(sc.keys()) for sc in sub_collections]
                if not all(hs == handle_sets[0] for hs in handle_sets):
                    raise ValueError(
                        f"Handle mismatch in group '{group_key}'. "
                        f"First has {handle_sets[0]}, others differ."
                    )
                # Concatenate each handle within this group
                group_result = {}
                for handle in sub_collections[0].keys():
                    trackings = [sc[handle] for sc in sub_collections]
                    group_result[handle] = Tracking.concat(
                        trackings, handle=handle, reindex=reindex
                    )
                result_dict[group_key] = cls(group_result)

            result = cls(result_dict)
            result._is_grouped = True
            result._groupby_tags = getattr(first, "_groupby_tags", None)
            return result

        else:
            # Flat collections: validate handles match
            handle_sets = [set(c.keys()) for c in collections]
            if not all(hs == handle_sets[0] for hs in handle_sets):
                raise ValueError(
                    f"Handle mismatch across collections. "
                    f"First has {handle_sets[0]}, others have {handle_sets[1:]}"
                )

            # Concatenate each handle
            result_dict = {}
            for handle in first.keys():
                trackings = [c[handle] for c in collections]
                result_dict[handle] = Tracking.concat(trackings, handle=handle, reindex=reindex)

            return cls(result_dict)

    def add_tags_from_csv(self, csv_path: str) -> None:
        """
        Adds tags to all Tracking objects in the collection from a csv file.
        csv_path: path to a csv file with first column: "handle"
        and other columns with tagnames as titles and tagvalues as values

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil, pandas as pd
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     # build a small collection
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        ...     # tags csv
        ...     tagcsv = d / 'tags.csv'
        ...     tagdf = pd.DataFrame([{'handle':'A','group':'G1'},{'handle':'B','group':'G2'}])
        ...     tagdf.to_csv(tagcsv, index=False)
        ...     coll.add_tags_from_csv(str(tagcsv))
        >>> coll['A'].tags
        {'group': 'G1'}
        >>> coll['B'].tags
        {'group': 'G2'}

        ```
        """
        df = pd.read_csv(csv_path)

        missing_handles = []
        handles_updated = set()
        num_tags_added = 0

        for _, row in df.iterrows():
            handle = row["handle"]
            if handle not in self.tracking_dict:
                missing_handles.append(handle)
                continue

            for tagname in df.columns[1:]:
                tagvalue = row[tagname]
                self.tracking_dict[handle].add_tag(tagname, tagvalue)
                num_tags_added += 1
                handles_updated.add(handle)

        print(f"added {num_tags_added} tags to {len(handles_updated)} elements in collection.")
        if len(missing_handles) > 0:
            missing_str = ", ".join(sorted(set(map(str, missing_handles))))
            print("the following handles were not found in collection: " + missing_str)

    def stored_info(self) -> pd.DataFrame:
        """
        Summarize stored tracked points across the collection's leaf Tracking objects.

        Returns a DataFrame indexed by `point_name` with columns:
        - `attached_to`: number of recordings containing the point
        - `missing_from`: number of recordings not containing the point
        - `dims`: point dimensions (e.g. `['x', 'y', 'z', 'likelihood']`), or a list
          of such dimension-sets when mixed across recordings.
        """
        leaves = list(self.flatten().values())
        total = len(leaves)
        if total == 0:
            return pd.DataFrame(
                columns=["point_name", "attached_to", "missing_from", "dims"]
            ).set_index("point_name")

        point_names = sorted({p for t in leaves for p in t.get_point_names()})
        records = []
        for point_name in point_names:
            attached = 0
            dims_seen: set[tuple[str, ...]] = set()
            for tracking in leaves:
                if point_name in tracking.get_point_names():
                    attached += 1
                    dims = [
                        d for d in tracking.get_point_dimensions(point_name) if d != "likelihood"
                    ]
                    dims_seen.add(tuple(dims))

            if len(dims_seen) == 1:
                dims_value = list(next(iter(dims_seen)))
            else:
                dims_value = [list(d) for d in sorted(dims_seen)]

            records.append(
                {
                    "point_name": point_name,
                    "attached_to": attached,
                    "missing_from": total - attached,
                    "dims": dims_value,
                }
            )

        out = pd.DataFrame.from_records(records).set_index("point_name")
        out["attached_to"] = out["attached_to"].astype("int64")
        out["missing_from"] = out["missing_from"].astype("int64")
        return out

    def stereo_triangulate(self) -> TrackingCollection:
        """
        Triangulate all TrackingMV objects and return a new TrackingCollection.
        The new collection will have the same grouping as the original.

        Notes
        -----
        This requires multi-view `TrackingMV` elements;
        typical `Tracking` elements do not support stereo triangulation.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil, json
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_mv import TrackingMV
        >>> # Create a collection with a single multi-view recording
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d) / 'rec1'
        ...     d.mkdir(parents=True, exist_ok=True)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p_csv:
        ...         _ = shutil.copy(p_csv, d / 'left.csv')
        ...         _ = shutil.copy(p_csv, d / 'right.csv')
        ...     # write a minimal synthetic calibration.json
        ...     calib = {
        ...         'view_order': ['left', 'right'],
        ...         'views': {
        ...             'left':  {'K': [[1,0,0],[0,1,0],[0,0,1]], 'dist': [0,0,0,0,0]},
        ...             'right': {'K': [[1,0,0],[0,1,0],[0,0,1]], 'dist': [0,0,0,0,0]},
        ...         },
        ...         'relative_pose': {'R': [[1,0,0],[0,1,0],[0,0,1]], 'T': [0.1, 0.0, 0.0]},
        ...     }
        ...     (d / 'calibration.json').write_text(json.dumps(calib))
        ...     # Build collection by scanning the parent folder with TrackingMV
        ...     parent = str(d.parent)
        ...     coll_mv = TrackingCollection.from_dlc_folder(
        ...         parent, tracking_cls=TrackingMV, fps=30)
        ...     coll_3d = coll_mv.stereo_triangulate()
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> isinstance(next(iter(coll_3d.values())), Tracking)
        True
        >>> next(iter(coll_3d.keys()))
        'rec1'

        ```
        """
        return self.map_leaves(lambda t: t.stereo_triangulate())

    def to_features(self) -> FeaturesCollection:
        """
        Create a `FeaturesCollection` from this `TrackingCollection`.

        This is a convenience wrapper around
        `FeaturesCollection.from_tracking_collection(self)` and preserves grouped
        structure when the collection is grouped.

        Returns:
            FeaturesCollection: Collection containing one `Features` object per
                tracking object in this collection.

        Examples:
            ```pycon
            >>> import tempfile, shutil
            >>> from pathlib import Path
            >>> from py3r.behaviour.tracking.tracking import Tracking
            >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
            >>> from py3r.behaviour.util.docdata import data_path
            >>> with tempfile.TemporaryDirectory() as d:
            ...     d = Path(d)
            ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
            ...         a = d / 'A.csv'; b = d / 'B.csv'
            ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
            ...     tc = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
            ...     fc = tc.to_features()
            >>> from py3r.behaviour.features.features_collection import FeaturesCollection
            >>> isinstance(fc, FeaturesCollection)
            True
            >>> sorted(fc.keys())
            ['A', 'B']

            ```
        """
        from py3r.behaviour.features.features_collection import FeaturesCollection

        return FeaturesCollection.from_tracking_collection(self)

    @property
    def loc(self):
        """
        Slice all elements with Tracking object .loc and return a new collection.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        >>> sub = coll.loc[0:2]
        >>> all(len(t.data) == 3 for t in sub.values())
        True

        ```
        """
        return _Indexer(self, self._loc)

    @property
    def iloc(self):
        """
        Slice all elements with Tracking object .iloc and return a new collection.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        >>> sub = coll.iloc[0:2]
        >>> all(len(t.data) == 2 for t in sub.values())
        True

        ```
        """
        return _Indexer(self, self._iloc)

    def _loc(self, idx):
        return self.__class__({k: v.loc[idx] for k, v in self.tracking_dict.items()})

    def _iloc(self, idx):
        return self.__class__({k: v.iloc[idx] for k, v in self.tracking_dict.items()})

    def plot(self, *args, **kwargs):
        """
        Plot all elements in the collection (or per group if grouped).

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         a = d / 'A.csv'; b = d / 'B.csv'
        ...         _ = shutil.copy(p, a); _ = shutil.copy(p, b)
        ...     coll = TrackingCollection.from_dlc({'A': str(a), 'B': str(b)}, fps=30)
        >>> _ = coll.plot(show=False)

        ```
        """
        if getattr(self, "is_grouped", False):
            for gkey, sub in self.items():
                print(f"\n=== Group: {gkey} ===")
                sub.plot(*args, **kwargs)
            return
        print(f"\nCollection: {getattr(self, 'handle', 'unnamed')}")
        for handle, tracking in self.tracking_dict.items():
            tracking.plot(*args, title=handle, **kwargs)
