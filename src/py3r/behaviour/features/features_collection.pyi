from __future__ import annotations

from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection_batch_mixin import FeaturesCollectionBatchMixin
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.util.base_collection import BaseCollection

class FeaturesCollection(BaseCollection, FeaturesCollectionBatchMixin):
    @property
    def each(self) -> Features: ...
    @property
    def features_dict(self) -> dict[str, Features]: ...
    @classmethod
    def from_tracking_collection(
        cls, tracking_collection: TrackingCollection, feature_cls=Features
    ) -> FeaturesCollection: ...
    @classmethod
    def from_list(cls, features_list: list[Features]) -> FeaturesCollection: ...
