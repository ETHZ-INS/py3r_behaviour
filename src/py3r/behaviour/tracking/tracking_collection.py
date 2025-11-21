from __future__ import annotations
import os
import pandas as pd
from py3r.behaviour.tracking.tracking import Tracking
from py3r.behaviour.tracking.tracking_collection_batch_mixin import (
    TrackingCollectionBatchMixin,
)
from py3r.behaviour.tracking.tracking_mv import TrackingMV
from py3r.behaviour.util.base_collection import BaseCollection
from py3r.behaviour.util.collection_utils import _Indexer
from py3r.behaviour.util.dev_utils import dev_mode


class TrackingCollection(BaseCollection, TrackingCollectionBatchMixin):
    """
    Collection of Tracking objects, keyed by name (e.g. for grouping individuals)
    note: type-hints refer to Tracking, but factory methods allow for other classes
    these are intended ONLY for subclasses of Tracking, and this is enforced
    """

    _element_type = Tracking

    def __init__(self, tracking_dict: dict[str, Tracking]):
        # Only validate handle mapping when values are leaf Tracking objects.
        # Grouped views (values are sub-collections) should skip this check.
        values = list(tracking_dict.values())
        if values and all(isinstance(v, Tracking) for v in values):
            for key, obj in tracking_dict.items():
                if obj.handle != key:
                    raise ValueError(
                        f"Key '{key}' does not match object's handle '{obj.handle}'"
                    )
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
        ...     coll = TrackingCollection.from_mapping(mapping, tracking_loader=Tracking.from_dlc, fps=30)
        >>> sorted(coll.keys())
        ['A', 'B']

        ```
        """
        if not issubclass(tracking_cls, Tracking):
            raise TypeError(
                f"tracking_cls must be Tracking or a subclass, got {tracking_cls}"
            )
        trackings = {}
        for handle, fp in handles_and_filepaths.items():
            trackings[handle] = tracking_loader(fp, handle=handle, **loader_kwargs)
        return cls(trackings)

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
        ...     coll = TrackingCollection.from_folder(str(d), tracking_loader=Tracking.from_dlc, fps=30)
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
                tracking_obj = tracking_loader(
                    recording_path, handle=recording, **loader_kwargs
                )
                tracking_dict[recording] = tracking_obj
        else:
            for fname in os.listdir(folder_path):
                if fname.endswith(".csv") and not fname.startswith("."):
                    handle = os.path.splitext(fname)[0]
                    fpath = os.path.join(folder_path, fname)
                    tracking_obj = tracking_loader(
                        fpath, handle=handle, **loader_kwargs
                    )
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
        ...     pd.DataFrame([{'handle':'A','group':'G1'},{'handle':'B','group':'G2'}]).to_csv(tagcsv, index=False)
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

        print(
            f"added {num_tags_added} tags to {len(handles_updated)} elements in collection."
        )
        if len(missing_handles) > 0:
            missing_str = ", ".join(sorted(set(map(str, missing_handles))))
            print("the following handles were not found in collection: " + missing_str)

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
        ...     coll_mv = TrackingCollection.from_dlc_folder(parent, tracking_cls=TrackingMV, fps=30)
        ...     coll_3d = coll_mv.stereo_triangulate()
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> isinstance(next(iter(coll_3d.values())), Tracking)
        True
        >>> next(iter(coll_3d.keys()))
        'rec1'

        ```
        """
        return self.map_leaves(lambda t: t.stereo_triangulate())

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
