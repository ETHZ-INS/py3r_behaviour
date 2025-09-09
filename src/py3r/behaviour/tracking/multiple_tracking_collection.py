from __future__ import annotations
import warnings

from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.tracking.tracking import LoadOptions
from py3r.behaviour.exceptions import BatchProcessError
from py3r.behaviour.util.collection_utils import _Indexer, BatchResult


class MultipleTrackingCollection:
    """
    Collection of TrackingCollection objects, keyed by name (e.g. for comparison between groups)
    """

    def __init__(self, tracking_collections: dict[str, TrackingCollection]):
        super().__init__(tracking_collections)

    def tracking_collections(self):
        return self._obj_dict

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

    def flatten(self):
        """
        Flatten a MultipleCollection into a single Collection containing all elements.
        Returns:
            Collection: a flat collection of all elements.
        """
        all_objs = []
        for (
            group_collection
        ) in self.tracking_collections.values():  # .values() yields each sub-collection
            all_objs.extend(group_collection.values())  # .values() yields the elements
        # Use from_list to create a new flat collection
        return TrackingCollection.from_list(all_objs)  # or TrackingCollection, etc.

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

    def __getitem__(self, key):
        if isinstance(key, int):
            handle = list(self.tracking_collections)[key]
            return self.tracking_collections[handle]
        elif isinstance(key, slice):
            handles = list(self.tracking_collections)[key]
            return self.__class__({h: self.tracking_collections[h] for h in handles})
        else:
            return self.tracking_collections[key]

    def keys(self):
        return self.tracking_collections.keys()

    def values(self):
        return self.tracking_collections.values()

    def items(self):
        return self.tracking_collections.items()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.tracking_collections)} TrackingCollection objects>"

    def plot(self, *args, **kwargs):
        for handle, collection in self.tracking_collections.items():
            print(f"\n=== Group: {handle} ===")
            collection.plot(*args, **kwargs)

    def __setitem__(self, key, value):
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
