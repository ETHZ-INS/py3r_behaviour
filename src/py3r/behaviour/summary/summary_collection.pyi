from __future__ import annotations

from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.summary.summary import Summary
from py3r.behaviour.summary.summary_collection_batch_mixin import SummaryCollectionBatchMixin
from py3r.behaviour.summary.summary_collection_plot_mixin import SummaryCollectionPlotMixin
from py3r.behaviour.util.base_collection import BaseCollection

class SummaryCollection(BaseCollection, SummaryCollectionBatchMixin, SummaryCollectionPlotMixin):
    @property
    def each(self) -> Summary: ...
    @property
    def summary_dict(self) -> dict[str, Summary]: ...
    @classmethod
    def from_features_collection(
        cls, features_collection: FeaturesCollection, summary_cls=Summary
    ) -> SummaryCollection: ...
    @classmethod
    def from_list(cls, summary_list: list[Summary]) -> SummaryCollection: ...
