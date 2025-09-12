from __future__ import annotations

from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.tracking.tracking import LoadOptions
from py3r.behaviour.util.collection_utils import _Indexer
from py3r.behaviour.util.base_collection import BaseMultipleCollection


class MultipleTrackingCollection(BaseMultipleCollection):
    """
    Collection of TrackingCollection objects, keyed by name (e.g. for comparison between groups)
    """

    _element_type = TrackingCollection
    _multiple_collection_type = "MultipleTrackingCollection"

    def __init__(self, tracking_collections: dict[str, TrackingCollection]):
        super().__init__(tracking_collections)

    @property
    def tracking_collections(self):
        return self._obj_dict

    @classmethod
    def from_dict(cls, trackingcollections: dict[str, TrackingCollection]):
        """
        creates a MultipleTrackingCollection from a dict of TrackingCollection objects
        """
        trackings = {key: obj for key, obj in trackingcollections.items()}
        return cls(trackings)

    @classmethod
    def from_dlc_folder(
        cls, parent_folder: str, options: LoadOptions, tracking_cls: type = None
    ) -> MultipleTrackingCollection:
        import os

        if tracking_cls is None:
            from py3r.behaviour.tracking.tracking import Tracking

            tracking_cls = Tracking
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
        cls, parent_folder: str, options: LoadOptions, tracking_cls: type = None
    ) -> MultipleTrackingCollection:
        import os

        if tracking_cls is None:
            from py3r.behaviour.tracking.tracking import Tracking

            tracking_cls = Tracking
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
        cls, parent_folder: str, options: LoadOptions, tracking_cls: type = None
    ) -> MultipleTrackingCollection:
        import os

        if tracking_cls is None:
            from py3r.behaviour.tracking.tracking import Tracking

            tracking_cls = Tracking
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

    def add_tags_from_csv(self, csv_path: str) -> None:
        self.flatten().add_tags_from_csv(csv_path)

    def stereo_triangulate(self):
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

    def plot(self, *args, **kwargs):
        for handle, collection in self.tracking_collections.items():
            print(f"\n=== Group: {handle} ===")
            collection.plot(*args, **kwargs)
