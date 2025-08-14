from __future__ import annotations
import json
import os
import warnings

from py3r.behaviour.tracking.tracking import Tracking, LoadOptions
from py3r.behaviour.tracking.tracking_mv import TrackingMV
from py3r.behaviour.exceptions import BatchProcessError
from py3r.behaviour.util.collection_utils import _Indexer, BatchResult
from py3r.behaviour.util.dev_utils import dev_mode


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

    def __getitem__(self, key):
        if isinstance(key, int):
            handle = list(self.tracking_dict)[key]
            return self.tracking_dict[handle]
        elif isinstance(key, slice):
            handles = list(self.tracking_dict)[key]
            return self.__class__({h: self.tracking_dict[h] for h in handles})
        else:
            return self.tracking_dict[key]

    def keys(self):
        return self.tracking_dict.keys()

    def values(self):
        return self.tracking_dict.values()

    def items(self):
        return self.tracking_dict.items()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.tracking_dict)} Tracking objects>"

    def plot(self, *args, **kwargs):
        print(f"\nCollection: {getattr(self, 'handle', 'unnamed')}")
        for handle, tracking in self.tracking_dict.items():
            tracking.plot(*args, title=handle, **kwargs)

    def __setitem__(self, key, value):
        if not isinstance(value, Tracking):
            raise TypeError(f"Value must be a Tracking, got {type(value).__name__}")
        warnings.warn(
            "Direct assignment to TrackingCollection is deprecated and may be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.tracking_dict[key] = value
