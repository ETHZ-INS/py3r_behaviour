from __future__ import annotations
import json
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


class TrackingCollection(TrackingCollectionBatchMixin, BaseCollection):
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
        tracking_dict = {}
        if issubclass(tracking_cls, TrackingMV):
            for recording in sorted(os.listdir(folder_path)):
                recording_path = os.path.join(folder_path, recording)
                if not os.path.isdir(recording_path):
                    continue
                filepaths = {}
                for fname in os.listdir(recording_path):
                    if fname.endswith(".csv") and not fname.startswith("."):
                        view = os.path.splitext(fname)[0]
                        filepaths[view] = os.path.join(recording_path, fname)
                calib_path = os.path.join(recording_path, "calibration.json")
                if not os.path.exists(calib_path):
                    raise FileNotFoundError(
                        f"Missing calibration.json in {recording_path}"
                    )
                with open(calib_path, "r") as f:
                    calibration = json.load(f)
                tracking_obj = tracking_cls.from_views(
                    filepaths,
                    handle=recording,
                    calibration=calibration,
                    tracking_loader=tracking_loader,
                    **loader_kwargs,
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
        """
        return self.map_leaves(lambda t: t.stereo_triangulate())

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
        if getattr(self, "is_grouped", False):
            for gkey, sub in self.items():
                print(f"\n=== Group: {gkey} ===")
                sub.plot(*args, **kwargs)
            return
        print(f"\nCollection: {getattr(self, 'handle', 'unnamed')}")
        for handle, tracking in self.tracking_dict.items():
            tracking.plot(*args, title=handle, **kwargs)
