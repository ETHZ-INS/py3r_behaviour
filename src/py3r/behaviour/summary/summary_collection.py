from __future__ import annotations

from py3r.behaviour.summary.summary import Summary
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.summary.summary_result import SummaryResult
from py3r.behaviour.util.base_collection import BaseCollection


class SummaryCollection(BaseCollection):
    """
    collection of Summary objects
    (e.g. for grouping individuals)
    note: type-hints refer to Summary, but factory methods allow for other classes
    these are intended ONLY for subclasses of Summary, and this is enforced
    """

    _element_type = Summary
    _multiple_collection_type = "MultipleSummaryCollection"

    def __init__(self, summary_dict: dict[str, Summary]):
        super().__init__(summary_dict)

    @property
    def summary_dict(self):
        return self._obj_dict

    @classmethod
    def from_features_collection(
        cls, features_collection: FeaturesCollection, summary_cls=Summary
    ):
        """
        creates a SummaryCollection from a FeaturesCollection
        """
        if not issubclass(summary_cls, Summary):
            raise TypeError(
                f"summary_cls must be Summary or a subclass, got {summary_cls}"
            )
        # check that dict handles match tracking handles
        for handle, f in features_collection.features_dict.items():
            if handle != f.handle:
                raise ValueError(
                    f"Key '{handle}' does not match object's handle '{f.handle}'"
                )
        return cls(
            {
                handle: summary_cls(f)
                for handle, f in features_collection.features_dict.items()
            }
        )

    @classmethod
    def from_list(cls, summary_list: list[Summary]):
        """
        creates a SummaryCollection from a list of Summary objects, keyed by handle
        """
        handles = [obj.handle for obj in summary_list]
        if len(handles) != len(set(handles)):
            raise Exception("handles must be unique")
        summary_dict = {obj.handle: obj for obj in summary_list}
        return cls(summary_dict)

    def make_bin(self, startframe, endframe):
        # returns a new SummaryCollection with binned summaries
        binned = {
            k: v.make_bin(startframe, endframe) for k, v in self.summary_dict.items()
        }
        return SummaryCollection(binned)

    def make_bins(self, numbins):
        # returns a list of SummaryCollection, one per bin
        bins = {
            k: v.make_bins(numbins) for k, v in self.summary_dict.items()
        }  # {k: [Summary, ...]}
        # Transpose: for each bin index, collect {k: Summary}
        nbins = len(next(iter(bins.values())))
        return [SummaryCollection({k: bins[k][i] for k in bins}) for i in range(nbins)]

    def store(
        self,
        results_dict: dict[str, SummaryResult],
        name: str = None,
        meta: dict = None,
        overwrite: bool = False,
    ):
        """
        Store all SummaryResult objects in a one-layer dict (as returned by batch methods).
        Example:
            results = summary_collection.time_true('is_running')
            summary_collection.store(results)
        """
        for v in results_dict.values():
            if hasattr(v, "store"):
                v.store(name=name, meta=meta, overwrite=overwrite)
            else:
                raise ValueError(f"{v} is not a SummaryResult object")
