from __future__ import annotations
import warnings

from py3r.behaviour.summary.summary import Summary
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.summary.summary_result import SummaryResult
from py3r.behaviour.exceptions import BatchProcessError
from py3r.behaviour.util.collection_utils import BatchResult


class SummaryCollection:
    """
    collection of Summary objects
    (e.g. for grouping individuals)
    note: type-hints refer to Summary, but factory methods allow for other classes
    these are intended ONLY for subclasses of Summary, and this is enforced
    """

    summary_dict: dict[str, Summary]

    def __init__(self, summary_dict: dict[str, Summary]):
        self.summary_dict = summary_dict

    def __getattr__(self, name):
        def batch_method(*args, **kwargs):
            results = {}
            for key, obj in self.summary_dict.items():
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

    def keys(self):
        """Return the keys of the summary_dict."""
        return self.summary_dict.keys()

    def values(self):
        """Return the values of the summary_dict."""
        return self.summary_dict.values()

    def items(self):
        """Return the items of the summary_dict."""
        return self.summary_dict.items()

    def __getitem__(self, key):
        """
        Get Summary by handle (str), by integer index, or by slice.
        """
        if isinstance(key, int):
            handle = list(self.summary_dict)[key]
            return self.summary_dict[handle]
        elif isinstance(key, slice):
            handles = list(self.summary_dict)[key]
            return self.__class__({h: self.summary_dict[h] for h in handles})
        else:
            return self.summary_dict[key]

    def __setitem__(self, key, value):
        if not isinstance(value, Summary):
            raise TypeError(f"Value must be a Summary, got {type(value).__name__}")
        warnings.warn(
            "Direct assignment to SummaryCollection is deprecated and may be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.summary_dict[key] = value

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} with {len(self.summary_dict)} Summary objects>"
        )
