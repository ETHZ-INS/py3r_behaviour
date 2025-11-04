from __future__ import annotations
import json
import os
import pandas as pd
from py3r.behaviour.tracking.tracking import (
    Tracking,
    LoadOptions,
)

from py3r.behaviour.tracking.tracking_mv import TrackingMV
from py3r.behaviour.util.base_collection import BaseCollection
from py3r.behaviour.util.collection_utils import _Indexer
from py3r.behaviour.util.dev_utils import dev_mode


class TrackingCollection(BaseCollection):
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
        cls, folder_path: str, options: LoadOptions, tracking_cls: type = Tracking
    ) -> TrackingCollection:
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
        cls, folder_path: str, options: LoadOptions, tracking_cls: type = Tracking
    ) -> TrackingCollection:
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
        cls, folder_path: str, options: LoadOptions, tracking_cls: type = Tracking
    ) -> TrackingCollection:
        tracking_dict = {}
        bookkeeping = cls._collect_tracking_files(
            folder_path, tracking_cls, options=options
        )
        for handle, kwargs in bookkeeping.items():
            tracking_obj = tracking_cls.from_dlcma(**kwargs)
            tracking_dict[handle] = tracking_obj
        return cls(tracking_dict)

    def add_tags_from_csv(self, csv_path: str) -> None:
        """
        Adds tags to all Tracking objects in the collection from a csv file.
        csv_path: path to a csv file with first column: "handle"
        and other columns with tagnames as titles and tagvalues as values
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
        result = {}
        if issubclass(tracking_cls, TrackingMV):
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

    def plot(self, *args, **kwargs):
        print(f"\nCollection: {getattr(self, 'handle', 'unnamed')}")
        for handle, tracking in self.tracking_dict.items():
            tracking.plot(*args, title=handle, **kwargs)
