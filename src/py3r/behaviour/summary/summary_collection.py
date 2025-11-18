from __future__ import annotations

import pandas as pd

from py3r.behaviour.summary.summary import Summary
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.summary.summary_result import SummaryResult
from py3r.behaviour.util.base_collection import BaseCollection
from py3r.behaviour.summary.summary_collection_batch_mixin import (
    SummaryCollectionBatchMixin,
)


class SummaryCollection(BaseCollection, SummaryCollectionBatchMixin):
    """
    collection of Summary objects
    (e.g. for grouping individuals)
    note: type-hints refer to Summary, but factory methods allow for other classes
    these are intended ONLY for subclasses of Summary, and this is enforced

    Examples
    --------
    ```pycon
    >>> import tempfile, shutil
    >>> from pathlib import Path
    >>> import pandas as pd
    >>> from py3r.behaviour.util.docdata import data_path
    >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
    >>> from py3r.behaviour.features.features_collection import FeaturesCollection
    >>> from py3r.behaviour.summary.summary_collection import SummaryCollection
    >>> with tempfile.TemporaryDirectory() as d:
    ...     d = Path(d)
    ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
    ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
    ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
    >>> fc = FeaturesCollection.from_tracking_collection(tc)
    >>> # add a simple boolean feature to each Features for summaries to consume
    >>> for f in fc.values():
    ...     s = pd.Series([True, False] * (len(f.tracking.data)//2 + 1))[:len(f.tracking.data)]
    ...     s.index = f.tracking.data.index
    ...     f.store(s, 'flag', meta={})
    >>> sc = SummaryCollection.from_features_collection(fc)
    >>> list(sorted(sc.keys()))
    ['A', 'B']

    ```
    """

    _element_type = Summary

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
        creates a SummaryCollection from a FeaturesCollection (flat or grouped)

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> from py3r.behaviour.features.features_collection import FeaturesCollection
        >>> from py3r.behaviour.summary.summary_collection import SummaryCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        >>> fc = FeaturesCollection.from_tracking_collection(tc)
        >>> # add numeric scalar per Features via a quick summary to test to_df later
        >>> for f in fc.values():
        ...     import numpy as np, pandas as pd
        ...     f.store(pd.Series(range(len(f.tracking.data)), index=f.tracking.data.index), 'counter', meta={})
        >>> sc = SummaryCollection.from_features_collection(fc)
        >>> isinstance(sc['A'], Summary) and isinstance(sc['B'], Summary)
        True

        ```
        """
        if not issubclass(summary_cls, Summary):
            raise TypeError(
                f"summary_cls must be Summary or a subclass, got {summary_cls}"
            )
        # Grouped case: preserve grouping
        if getattr(features_collection, "is_grouped", False):
            grouped_dict = {}
            for gkey, sub_fc in features_collection.items():
                for handle, f in sub_fc.features_dict.items():
                    if handle != f.handle:
                        raise ValueError(
                            f"Key '{handle}' does not match object's handle '{f.handle}'"
                        )
                grouped_dict[gkey] = cls(
                    {
                        handle: summary_cls(f)
                        for handle, f in sub_fc.features_dict.items()
                    }
                )
            grouped_sc = cls(grouped_dict)
            grouped_sc._is_grouped = True
            grouped_sc._groupby_tags = getattr(
                features_collection, "groupby_tags", None
            )
            return grouped_sc
        # Flat case
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

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> from py3r.behaviour.summary.summary_collection import SummaryCollection
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t1 = Tracking.from_dlc(str(p), handle='A', fps=30)
        ...     t2 = Tracking.from_dlc(str(p), handle='B', fps=30)
        >>> f1, f2 = Features(t1), Features(t2)
        >>> # store simple scalar summaries
        >>> s1, s2 = Summary(f1), Summary(f2)
        >>> s1.store(1, 'count'); s2.store(2, 'count')
        >>> sc = SummaryCollection.from_list([s1, s2])
        >>> list(sorted(sc.keys()))
        ['A', 'B']

        ```
        """
        handles = [obj.handle for obj in summary_list]
        if len(handles) != len(set(handles)):
            raise Exception("handles must be unique")
        summary_dict = {obj.handle: obj for obj in summary_list}
        return cls(summary_dict)

    def to_df(self, include_tags: bool = False, tag_prefix: str = "tag_"):
        """
        Collate scalar values (numeric, string, bool) from each Summary.data into a pandas DataFrame.

        - Index: handles of the Summary objects
        - Columns: keys from each Summary.data (simple scalar values)
        - If include_tags is True, include tag columns with the given prefix

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> from py3r.behaviour.summary.summary_collection import SummaryCollection
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t1 = Tracking.from_dlc(str(p), handle='A', fps=30)
        ...     t2 = Tracking.from_dlc(str(p), handle='B', fps=30)
        >>> s1, s2 = Summary(Features(t1)), Summary(Features(t2))
        >>> s1.store(1.0, 'score'); s2.store(2.0, 'score')
        >>> s1.features.tracking.add_tag('group', 'G1'); s2.features.tracking.add_tag('group', 'G2')
        >>> sc = SummaryCollection.from_list([s1, s2])
        >>> df = sc.to_df(include_tags=True)
        >>> set(df.columns) >= {'score', 'tag_group'}
        True

        ```
        """
        import numbers

        rows = {}
        for handle, summary in self.summary_dict.items():
            row = {}
            for key, value in summary.data.items():
                if isinstance(value, (numbers.Number, str, bool)):
                    row[key] = value
            if include_tags and getattr(summary, "tags", None):
                for tag_key, tag_val in summary.tags.items():
                    row[f"{tag_prefix}{tag_key}"] = tag_val
            rows[handle] = row

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "handle"
        return df

    def make_bin(self, startframe, endframe):
        """
        returns a new SummaryCollection with binned summaries

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> from py3r.behaviour.summary.summary_collection import SummaryCollection
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='A', fps=30)
        >>> s = Summary(Features(t))
        >>> sc = SummaryCollection.from_list([s])
        >>> b = sc.make_bin(0, 2)
        >>> isinstance(b, SummaryCollection)
        True

        ```
        """
        binned = {
            k: v.make_bin(startframe, endframe) for k, v in self.summary_dict.items()
        }
        return SummaryCollection(binned)

    def make_bins(self, numbins):
        """
        returns a list of SummaryCollection, one per bin

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> from py3r.behaviour.summary.summary_collection import SummaryCollection
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='A', fps=30)
        >>> sc = SummaryCollection.from_list([Summary(Features(t))])
        >>> bins = sc.make_bins(3)
        >>> len(bins) == 3 and all(isinstance(b, SummaryCollection) for b in bins)
        True

        ```
        """
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

        Examples
        --------
        ```pycon
        >>> import pandas as pd, tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> from py3r.behaviour.features.features_collection import FeaturesCollection
        >>> from py3r.behaviour.summary.summary_collection import SummaryCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        >>> fc = FeaturesCollection.from_tracking_collection(tc)
        >>> # add a boolean column for summaries
        >>> for f in fc.values():
        ...     m = pd.Series([True, False] * (len(f.tracking.data)//2 + 1))[:len(f.tracking.data)]
        ...     m.index = f.tracking.data.index
        ...     f.store(m, 'mask', meta={})
        >>> sc = SummaryCollection.from_features_collection(fc)
        >>> rd = {h: s.time_true('mask') for h, s in sc.items()}
        >>> sc.store(rd, name='t_mask')
        >>> all('t_mask' in s.data for s in sc.values())
        True

        ```
        """
        for v in results_dict.values():
            if hasattr(v, "store"):
                v.store(name=name, meta=meta, overwrite=overwrite)
            else:
                raise ValueError(f"{v} is not a SummaryResult object")

    # ---- Cross-group analysis (formerly in MultipleSummaryCollection) ----
    def bfa(self, column: str, all_states=None, numshuffles: int = 1000):
        """
        Behaviour Flow Analysis between groups for a grouped SummaryCollection.

        Requires the collection to be grouped (via groupby). Computes transition
        matrices per Summary within each group, then computes Manhattan distances
        between group means and surrogate distributions via shuffling.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> from py3r.behaviour.features.features_collection import FeaturesCollection
        >>> from py3r.behaviour.summary.summary_collection import SummaryCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        >>> fc = FeaturesCollection.from_tracking_collection(tc)
        >>> # inject simple 2-state labels and tags to build groups
        >>> for i, (h, f) in enumerate(fc.items()):
        ...     states = pd.Series(['A','A','B','B','A'] * (len(f.tracking.data)//5 + 1))[:len(f.tracking.data)]
        ...     states.index = f.tracking.data.index
        ...     f.store(states, 'state', meta={})
        ...     f.tracking.add_tag('group', f'G{i+1}')
        >>> gfc = fc.groupby('group')
        >>> sc = SummaryCollection.from_features_collection(gfc)
        >>> res = sc.bfa('state', all_states=['A','B'], numshuffles=2)
        >>> isinstance(res, dict) and 'observed' in next(iter(res.values()))
        True

        ```
        """
        if not getattr(self, "is_grouped", False):
            raise ValueError(
                "bfa requires a grouped SummaryCollection (call groupby first)"
            )

        from itertools import combinations

        # batch calculate transition matrix for each summary object
        transition_matrices_result = self.transition_matrix(column, all_states)
        # Extract the .value from each SummaryResult in the nested dict
        transition_matrices = {
            group: {k: v.value for k, v in d.items()}
            for group, d in transition_matrices_result.items()
        }
        # calculate manhattan distance for each group pair
        distances = {}
        for group1, group2 in combinations(self.group_keys, 2):
            _ = {}
            list1 = list(transition_matrices[group1].values())
            list2 = list(transition_matrices[group2].values())
            _["observed"] = self._manhattan_distance_twogroups(list1, list2)
            _["surrogates"] = [
                self._shuffle_lists(*self._shuffle_lists(list1, list2))
                and self._manhattan_distance_twogroups(
                    *self._shuffle_lists(list1, list2)
                )
                for i in range(numshuffles)
            ]
            distances[f"{group1}_vs_{group2}"] = _
        return distances

    @staticmethod
    def bfa_stats(
        bfa_results: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """
        Compute simple statistics (percentile, zscore, right_tail_p) from bfa results.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> from py3r.behaviour.features.features_collection import FeaturesCollection
        >>> from py3r.behaviour.summary.summary_collection import SummaryCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        >>> fc = FeaturesCollection.from_tracking_collection(tc)
        >>> for i, (h, f) in enumerate(fc.items()):
        ...     states = pd.Series(['A','A','B','B','A'] * (len(f.tracking.data)//5 + 1))[:len(f.tracking.data)]
        ...     states.index = f.tracking.data.index
        ...     f.store(states, 'state', meta={})
        ...     f.tracking.add_tag('group', f'G{i+1}')
        >>> sc = SummaryCollection.from_features_collection(fc.groupby('group'))
        >>> bfa_out = sc.bfa('state', all_states=['A','B'], numshuffles=2)
        >>> stats = SummaryCollection.bfa_stats(bfa_out)
        >>> set(next(iter(stats.values())).keys()) >= {'percentile','zscore','right_tail_p'}
        True

        ```
        """
        import numpy as np
        import pandas as pd

        def percentile(observed: float, surrogates: list[float]) -> float:
            return sum(observed > pd.Series(surrogates)) / (len(surrogates) + 1)

        def zscore(observed: float, surrogates: list[float]) -> float:
            return (observed - np.mean(surrogates)) / np.std(surrogates)

        def right_tail_p(observed: float, surrogates: list[float]) -> float:
            from math import erf

            return 0.5 * (1 - erf(zscore(observed, surrogates) / np.sqrt(2)))

        stats = {}
        for group, result in bfa_results.items():
            observed = result["observed"]
            surrogates = result["surrogates"]
            stats[group] = {
                "percentile": percentile(observed, surrogates),
                "zscore": zscore(observed, surrogates),
                "right_tail_p": right_tail_p(observed, surrogates),
            }
        return stats

    @staticmethod
    def _manhattan_distance(
        transition_matrix1: pd.DataFrame, transition_matrix2: pd.DataFrame
    ) -> float:
        # check that transition_matrix1 and transition_matrix2 have the same index and columns
        if not transition_matrix1.index.equals(transition_matrix2.index):
            raise ValueError(
                "transition_matrix1 and transition_matrix2 must have the same index"
            )
        if not transition_matrix1.columns.equals(transition_matrix2.columns):
            raise ValueError(
                "transition_matrix1 and transition_matrix2 must have the same columns"
            )
        difference = transition_matrix1 - transition_matrix2
        return difference.abs().sum(axis=1).sum()

    @staticmethod
    def _mean_transition_matrix(matrices: list[pd.DataFrame]) -> pd.DataFrame:
        summed_matrix = sum(matrices)
        mean_matrix = summed_matrix / len(matrices)
        return mean_matrix

    def _manhattan_distance_twogroups(
        self, list1: list[pd.DataFrame], list2: list[pd.DataFrame]
    ) -> float:
        # calculate manhattan distance between two lists of transition matrices
        distance = self._manhattan_distance(
            self._mean_transition_matrix(list1), self._mean_transition_matrix(list2)
        )
        return distance

    @staticmethod
    def _shuffle_lists(group1: list, group2: list) -> tuple[list, list]:
        import random

        n1 = len(group1)
        combined = group1 + group2
        random.shuffle(combined)
        new_group1 = combined[:n1]
        new_group2 = combined[n1:]
        return new_group1, new_group2
