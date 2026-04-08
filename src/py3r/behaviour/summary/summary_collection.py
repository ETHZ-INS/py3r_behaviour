from __future__ import annotations

import random
import warnings
from itertools import combinations
from typing import Literal

import pandas as pd

from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.summary.summary import Summary
from py3r.behaviour.summary.summary_collection_plot_mixin import (
    SummaryCollectionPlotMixin,
)
from py3r.behaviour.util.base_collection import BaseCollection
from py3r.behaviour.util.collection_utils import resolve_single_store_name


class SummaryCollection(BaseCollection, SummaryCollectionPlotMixin):
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
    each: Summary
    each_forcebatch: Summary

    def __init__(self, summary_dict: dict[str, Summary]):
        super().__init__(summary_dict)

    @property
    def summary_dict(self):
        return self._obj_dict

    @classmethod
    def from_features_collection(cls, features_collection: FeaturesCollection, summary_cls=Summary):
        """
        Create a SummaryCollection from a FeaturesCollection.

        Parameters
        ----------
        features_collection : FeaturesCollection
            Source collection. Grouped structure is preserved.
        summary_cls : type, default=Summary
            ``Summary`` subclass to instantiate for each session.

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
        ...     s = pd.Series(range(len(f.tracking.data)), index=f.tracking.data.index)
        ...     f.store(s, 'counter', meta={})
        >>> sc = SummaryCollection.from_features_collection(fc)
        >>> isinstance(sc['A'], Summary) and isinstance(sc['B'], Summary)
        True

        ```
        """
        if not issubclass(summary_cls, Summary):
            raise TypeError(f"summary_cls must be Summary or a subclass, got {summary_cls}")
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
                    {handle: summary_cls(f) for handle, f in sub_fc.features_dict.items()}
                )
            grouped_sc = cls(grouped_dict)
            grouped_sc._is_grouped = True
            grouped_sc._groupby_tags = getattr(features_collection, "groupby_tags", None)
            return grouped_sc
        # Flat case
        for handle, f in features_collection.features_dict.items():
            if handle != f.handle:
                raise ValueError(f"Key '{handle}' does not match object's handle '{f.handle}'")
        return cls(
            {handle: summary_cls(f) for handle, f in features_collection.features_dict.items()}
        )

    @classmethod
    def from_list(cls, summary_list: list[Summary]):
        """
        Create a SummaryCollection from a list of Summary objects, keyed by handle.

        Parameters
        ----------
        summary_list : list[Summary]
            Summary objects to collect. All handles must be unique.

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

    def to_df(
        self,
        include_tags: bool = False,
        tag_prefix: str = "tag_",
        series: Literal["ignore", "separate"] = "ignore",
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Collate values from each Summary.data into tabular output.

        - Index: handles of the Summary objects
        - Scalar columns: keys from each Summary.data with scalar values
        - If include_tags is True, include tag columns with the given prefix
        - If series='ignore' (default), Series entries are skipped
        - If series='separate', return `(scalars_df, series_tables)` where
          `series_tables` is `{metric_name: dataframe}` and each dataframe has
          one row per handle and one column per Series index value.

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
        >>> s1.store(pd.Series([1.0, 2.0], index=['A', 'B']), 'speed_by_state')
        >>> s2.store(pd.Series([3.0, 4.0], index=['A', 'B']), 'speed_by_state')
        >>> scalars, series_tables = sc.to_df(series='separate')
        >>> isinstance(scalars, pd.DataFrame) and 'speed_by_state' in series_tables
        True

        ```
        """
        import numbers

        if series not in {"ignore", "separate"}:
            raise ValueError("series must be one of: 'ignore', 'separate'")

        rows = {}
        tags_by_handle: dict[str, dict] = {}
        series_rows: dict[str, dict[str, pd.Series]] = {}
        for handle, summary in self.summary_dict.items():
            row = {}
            for key, value in summary.data.items():
                if isinstance(value, (numbers.Number, str, bool)):
                    row[key] = value
                elif isinstance(value, pd.Series) and series == "separate":
                    series_rows.setdefault(key, {})[handle] = value
            if include_tags and getattr(summary, "tags", None):
                tags_by_handle[handle] = dict(summary.tags)
                for tag_key, tag_val in summary.tags.items():
                    row[f"{tag_prefix}{tag_key}"] = tag_val
            rows[handle] = row

        scalars_df = pd.DataFrame.from_dict(rows, orient="index")
        scalars_df.index.name = "handle"
        if series == "ignore":
            return scalars_df

        series_tables = {}
        for metric_name, metric_rows in series_rows.items():
            table = pd.DataFrame.from_dict(metric_rows, orient="index")
            table.index.name = "handle"
            if include_tags:
                for handle, tags in tags_by_handle.items():
                    if handle not in table.index:
                        continue
                    for tag_key, tag_val in tags.items():
                        table.loc[handle, f"{tag_prefix}{tag_key}"] = tag_val
            series_tables[metric_name] = table
        return scalars_df, series_tables

    def make_bin(self, startframe: int, endframe: int):
        """
        Return a new SummaryCollection restricted to frames in [startframe, endframe).

        Parameters
        ----------
        startframe : int
            First frame index of the bin (inclusive).
        endframe : int
            Last frame index of the bin (exclusive).

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
        binned = {k: v.make_bin(startframe, endframe) for k, v in self.summary_dict.items()}
        return SummaryCollection(binned)

    def make_bins(self, numbins):
        """
        Divide the collection into equal time bins and return one SummaryCollection per bin.

        Parameters
        ----------
        numbins : int
            Number of equal-length bins to split each session into.

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
        results_dict,
        name: str = None,
        meta: dict = None,
        overwrite: bool = False,
    ):
        """
        Store SummaryResult objects returned by batch methods.

        Parameters
        ----------
        results_dict : dict
            Batch results to store. Flat: ``{handle: SummaryResult}``.
            Grouped: ``{group_key: {handle: SummaryResult}}``.
        name : str | None, default=None
            Metric name to store under. If None, resolved automatically from
            the result objects (all must agree on a single name).
        meta : dict | None, default=None
            Metadata dict to attach alongside the stored metric.
        overwrite : bool, default=False
            If True, overwrite an existing metric with the same name.

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

        Returns
        -------
        str
            The resolved stored metric name. If auto-naming would resolve to
            multiple different names across leaves, raises ValueError.
        """

        def _resolve_leaf_name(v):
            if hasattr(v, "_func_name"):
                return v._func_name
            raise ValueError(f"{v} is not a SummaryResult object")

        resolved_name = resolve_single_store_name(results_dict, name, _resolve_leaf_name)

        if getattr(self, "is_grouped", False):
            for _, group_dict in results_dict.items():
                for _, v in group_dict.items():
                    if hasattr(v, "store"):
                        v.store(name=name, meta=meta, overwrite=overwrite)
                    else:
                        raise ValueError(f"{v} is not a SummaryResult object")
            return resolved_name

        for v in results_dict.values():
            if hasattr(v, "store"):
                v.store(name=name, meta=meta, overwrite=overwrite)
            else:
                raise ValueError(f"{v} is not a SummaryResult object")
        return resolved_name

    def stored_info(self) -> pd.DataFrame:
        """
        Summarize stored summary metrics across the collection's leaf Summary objects.

        Returns a DataFrame indexed by `summary` with columns:
        - `attached_to`: number of recordings containing the summary key
        - `missing_from`: number of recordings not containing the summary key
        - `type`: value datatype name when consistent, or a list of datatype names
          when mixed across recordings.
        """
        leaves = list(self.flatten().values())
        total = len(leaves)
        if total == 0:
            cols = ["summary", "attached_to", "missing_from", "type"]
            return pd.DataFrame(columns=cols).set_index("summary")

        summary_names = sorted({name for summary in leaves for name in summary.data.keys()})
        records = []
        for name in summary_names:
            attached = 0
            type_seen: set[str] = set()
            for summary in leaves:
                if name in summary.data:
                    attached += 1
                    type_seen.add(type(summary.data[name]).__name__)

            type_value: str | list[str]
            if len(type_seen) == 1:
                type_value = next(iter(type_seen))
            else:
                type_value = sorted(type_seen)

            records.append(
                {
                    "summary": name,
                    "attached_to": attached,
                    "missing_from": total - attached,
                    "type": type_value,
                }
            )

        out = pd.DataFrame.from_records(records).set_index("summary")
        out["attached_to"] = out["attached_to"].astype("int64")
        out["missing_from"] = out["missing_from"].astype("int64")
        return out

    # ---- Cross-group analysis (formerly in MultipleSummaryCollection) ----
    def bfa(
        self,
        column: str,
        all_states=None,
        numshuffles: int = 1000,
        pairs: list[tuple[str, str]] | None = None,
        random_state: int | None = 0,
        scale_by_transitions: bool = False,
    ):
        """
        Behaviour Flow Analysis between groups for a grouped SummaryCollection.

        Requires the collection to be grouped (via groupby). Computes transition
        matrices per Summary within each group, then computes Manhattan distances
        between group means and surrogate distributions via shuffling.

        If `pairs` is provided, only those group pairs are analyzed; otherwise all
        unique pairs in `self.group_keys` are evaluated.

        Parameters
        ----------
        column : str
            Name of the column containing discrete state labels.
        all_states : list | None
            Explicit state ordering for the transition matrix.  ``None`` infers
            states from the data.
        numshuffles : int
            Number of surrogate shuffles used to build the null distribution.
        pairs : list[tuple[str, str]] | None
            Group pairs to compare.  ``None`` evaluates all unique pairs.
        random_state : int | None
            Seed for reproducible surrogate shuffling.  ``None`` keeps
            non-deterministic behaviour.  Pass the same seed to each ``bfa()``
            call when combining scales so that surrogate shuffles are
            synchronised; see :meth:`combine_bfa_results`.
        scale_by_transitions : bool, default False
            If ``True``, each pairwise Manhattan distance (observed and all
            surrogates) is divided by the total number of transitions across
            both groups for that pair.  This rescales raw-count distances to a
            per-transition unit, making distances comparable across temporal
            resolutions with different numbers of observations.  Defaults to
            ``False`` to preserve legacy behaviour and retain the information
            contained in total transition counts.

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
        ...     pat = ['A','A','B','B','A'] * (len(f.tracking.data)//5 + 1)
        ...     states = pd.Series(pat[:len(f.tracking.data)], index=f.tracking.data.index)
        ...     f.store(states, 'state', meta={})
        ...     f.tracking.add_tag('group', f'G{i+1}')
        >>> gfc = fc.groupby('group')
        >>> sc = SummaryCollection.from_features_collection(gfc)
        >>> # compute all pairs (raw transition counts)
        >>> res = sc.bfa('state', all_states=['A','B'], numshuffles=2)
        >>> isinstance(res, dict) and 'observed' in next(iter(res.values()))
        True
        >>> # compute only specific pair(s)
        >>> res2 = sc.bfa('state', all_states=['A','B'], numshuffles=2, pairs=[('G1','G2')])
        >>> list(res2.keys()) == ['G1_vs_G2']
        True
        >>> # scale distances by total transition count (comparable across resolutions)
        >>> res3 = sc.bfa('state', all_states=['A','B'], numshuffles=2, scale_by_transitions=True)
        >>> isinstance(res3, dict) and 'observed' in next(iter(res3.values()))
        True

        ```
        """
        if not getattr(self, "is_grouped", False):
            raise ValueError("bfa requires a grouped SummaryCollection (call groupby first)")

        rng = random.Random(random_state)

        # batch calculate transition matrix for each summary object
        transition_matrices_result = self.each.transition_matrix(column, all_states)
        # Extract the .value from each SummaryResult in the nested dict
        transition_matrices = {
            group: {k: v.value for k, v in d.items()}
            for group, d in transition_matrices_result.items()
        }

        # helper to format group keys for human-friendly labels
        def _fmt_group(g):
            if isinstance(g, tuple) and len(g) == 1:
                return g[0]
            return g

        # map from formatted label back to original key for convenience
        label_to_key = {_fmt_group(g): g for g in self.group_keys}
        # determine group pairs to evaluate
        if pairs is None:
            pair_iter = combinations(self.group_keys, 2)
        else:
            # validate provided pairs
            group_set = set(self.group_keys)
            normalized_pairs: list[tuple] = []
            for g1, g2 in pairs:
                # allow passing either raw keys or formatted labels
                _g1 = g1 if g1 in group_set else label_to_key.get(g1, None)
                _g2 = g2 if g2 in group_set else label_to_key.get(g2, None)
                if _g1 is None or _g2 is None:
                    valid = sorted(map(_fmt_group, self.group_keys))
                    raise ValueError(f"Invalid group pair ({g1}, {g2}); valid groups: {valid}")
                normalized_pairs.append((_g1, _g2))
            pair_iter = normalized_pairs

        # calculate manhattan distance for each requested group pair
        distances = {}
        for group1, group2 in pair_iter:
            _ = {}
            list1 = list(transition_matrices[group1].values())
            list2 = list(transition_matrices[group2].values())
            if scale_by_transitions:
                total_T = float(sum(tm.to_numpy().sum() for tm in list1 + list2))
                scale = 1.0 / total_T if total_T > 0 else 1.0
            else:
                scale = 1.0
            _["observed"] = self._manhattan_distance_twogroups(list1, list2) * scale
            _["surrogates"] = [
                self._manhattan_distance_twogroups(*self._shuffle_lists(list1, list2, rng)) * scale
                for _ in range(numshuffles)
            ]
            # use formatted labels for result key
            distances[f"{_fmt_group(group1)}_vs_{_fmt_group(group2)}"] = _
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
        ...     pat = ['A','A','B','B','A'] * (len(f.tracking.data)//5 + 1)
        ...     states = pd.Series(pat[:len(f.tracking.data)], index=f.tracking.data.index)
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
    def plot_bfa_results(
        results: dict[str, dict[str, float]],
        compares: str | list[str] | None = None,
        add_stats: bool = True,
        stats: dict[str, dict[str, float]] | None = None,
        bins: int = 50,
        figsize: tuple[float, float] = (4, 3),
        save_dir: str | None = None,
        show: bool = True,
        # legacy: allow single 'compare' name
        compare: str | None = None,
    ):
        """
        Plot one or more BFA result comparisons as separate single-panel figures.

        - If `compares` is None and results contain a single comparison, that one is plotted.
        - If `compares` is a string, only that comparison is plotted.
        - If `compares` is a list of strings, each comparison is plotted separately.
        - If `add_stats` is True and `stats` not provided, statistics will be computed
          via `SummaryCollection.bfa_stats(results)` and annotated on each plot.

        Returns `(fig, ax)` for a single comparison, or a dict `{compare: (fig, ax)}`
        for multiple.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil, os
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
        >>> # add simple 2-state labels and tags to build two groups
        >>> for i, (h, f) in enumerate(fc.items()):
        ...     pat = ['A','A','B','B','A'] * (len(f.tracking.data)//5 + 1)
        ...     states = pd.Series(pat[:len(f.tracking.data)], index=f.tracking.data.index)
        ...     f.store(states, 'state', meta={})
        ...     f.tracking.add_tag('group', f'G{i+1}')
        >>> sc = SummaryCollection.from_features_collection(fc.groupby('group'))
        >>> bfa_out = sc.bfa('state', all_states=['A','B'], numshuffles=5)
        >>> # plot a single comparison and save it
        >>> with tempfile.TemporaryDirectory() as outdir:
        ...     fig, ax = SummaryCollection.plot_bfa_results(
        ...         bfa_out, compare='G1_vs_G2', show=False, save_dir=outdir)
        ...     os.path.exists(os.path.join(outdir, 'G1_vs_G2.png'))
        True

        ```
        """
        import os

        import matplotlib.pyplot as plt

        def _sanitize(name: str) -> str:
            return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(name))

        # selection
        if compares is None and compare is not None:
            compares = compare
        if compares is None:
            keys = list(results.keys())
        elif isinstance(compares, str):
            keys = [compares]
        elif isinstance(compares, list):
            keys = compares
        else:
            raise TypeError("compares must be None, str, or list[str]")

        if len(keys) == 0:
            raise ValueError("No comparisons to plot.")

        # compute stats once if requested and not provided
        if add_stats and stats is None:
            stats = SummaryCollection.bfa_stats(results)

        out: dict[str, tuple] = {}
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        for k in keys:
            if k not in results:
                continue
            fig, ax = plt.subplots(figsize=figsize)
            ax.hist(results[k]["surrogates"], color="gray", bins=bins)
            ax.axvline(results[k]["observed"], color="red")
            ax.set_xlabel("distance")
            ax.set_ylabel("count")
            ax.set_title(k, fontdict={"size": 10})
            if add_stats and stats is not None and k in stats:
                p_empirical = 1 - stats[k]["percentile"]
                if p_empirical < 0.0001:
                    sig = "****"
                elif p_empirical < 0.001:
                    sig = "***"
                elif p_empirical < 0.01:
                    sig = "**"
                elif p_empirical < 0.05:
                    sig = "*"
                else:
                    sig = "n.s."
                text = f"p={p_empirical:.3f}\n{sig}"
                ax.text(
                    0.95,
                    0.95,
                    text,
                    ha="right",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.8,
                        edgecolor="none",
                    ),
                    zorder=10,
                )
            plt.tight_layout()
            if save_dir:
                fig.savefig(
                    os.path.join(save_dir, f"{_sanitize(k)}.png"),
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0.02,
                )
            if show:
                plt.show()
            out[k] = (fig, ax)
        # return a single tuple when only one compare was requested to keep ergonomics
        if len(out) == 1:
            return next(iter(out.values()))
        return out

    def plot_transition_umap(
        self,
        column: str,
        all_states=None,
        groups: list[str | tuple[str, ...]] | list[list[str | tuple[str, ...]]] | None = None,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 0,
        figsize: tuple[float, float] = (4.5, 4),
        show: bool = True,
        save_dir: str | None = None,
    ):
        """
        Plot a UMAP embedding of per-subject transition matrices for selected groups.

        Transition matrices are computed for each subject within each group, flattened,
        scaled, and embedded with UMAP. The collection must already be grouped, for
        example via ``groupby``.

        Parameters
        ----------
        column
            Name of the categorical column used to compute transition matrices.
        all_states
            Optional explicit state ordering used when constructing transition matrices.
        groups
            Optional group selection. If omitted, all groups are included.

            This argument supports three forms:

            - A flat list of single-tag group labels, for example
              ``['control', 'treatment']``.
            - A flat list of multi-tag group keys (tuples), for example
              ``[('control', 'time1'), ('control', 'time2')]``.
            - A list of lists defining ordered sequences of groups, for example
              ``[[('control', 'time1'), ('control', 'time2')],
              [('treatment', 'time1'), ('treatment', 'time2')]]``.

            When sequences are provided, each sequence is plotted using a monochrome
            gradient to indicate progression within that sequence.
        n_neighbors
            Number of neighbors used by UMAP.
        min_dist
            Minimum distance parameter passed to UMAP.
        random_state
            Seed for reproducible UMAP embeddings.
        figsize
            Figure size passed to Matplotlib.
        show
            If True, display the figure.
        save_dir
            Optional directory in which to save the plot as
            ``transition_umap.png``.

        Returns
        -------
        fig, ax
            Matplotlib figure and axis.

        Raises
        ------
        ValueError
            If the collection is not grouped, or if no data are found for the
            requested groups.
        ImportError
            If ``umap-learn`` is not installed.

        Examples
        --------
        ```pycon
        >>> # xdoctest: +REQUIRES(module: umap)
        >>> import os, shutil, tempfile
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> from py3r.behaviour.features.features_collection import FeaturesCollection
        >>> from py3r.behaviour.summary.summary_collection import SummaryCollection

        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     paths = {}
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         for name in ['A', 'B', 'C', 'D']:
        ...             dst = d / f'{name}.csv'
        ...             _ = shutil.copy(p, dst)
        ...             paths[name] = str(dst)
        ...     tc = TrackingCollection.from_dlc(paths, fps=30)
        ...     fc = FeaturesCollection.from_tracking_collection(tc)
        ...
        ...     tags = {
        ...         'A': ('control', 'time1'),
        ...         'B': ('control', 'time2'),
        ...         'C': ('treatment', 'time1'),
        ...         'D': ('treatment', 'time2'),
        ...     }
        ...
        ...     for h, f in fc.items():
        ...         pat = ['A', 'A', 'B', 'B', 'A'] * (len(f.tracking.data) // 5 + 1)
        ...         states = pd.Series(pat[:len(f.tracking.data)], index=f.tracking.data.index)
        ...         f.store(states, 'state', meta={})
        ...         condition, time = tags[h]
        ...         f.tracking.add_tag('condition', condition)
        ...         f.tracking.add_tag('time', time)
        ...
        ...     sc = SummaryCollection.from_features_collection(fc.groupby(['condition', 'time']))
        ...
        ...     with tempfile.TemporaryDirectory() as outdir:
        ...         fig, ax = sc.plot_transition_umap(
        ...             column='state',
        ...             all_states=['A', 'B'],
        ...             groups=[('control', 'time1'), ('control', 'time2')],
        ...             show=False,
        ...             save_dir=outdir,
        ...         )
        ...         os.path.exists(os.path.join(outdir, 'transition_umap.png'))
        True

        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     paths = {}
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         for name in ['A', 'B', 'C', 'D']:
        ...             dst = d / f'{name}.csv'
        ...             _ = shutil.copy(p, dst)
        ...             paths[name] = str(dst)
        ...     tc = TrackingCollection.from_dlc(paths, fps=30)
        ...     fc = FeaturesCollection.from_tracking_collection(tc)
        ...
        ...     tags = {
        ...         'A': ('control', 'time1'),
        ...         'B': ('control', 'time2'),
        ...         'C': ('treatment', 'time1'),
        ...         'D': ('treatment', 'time2'),
        ...     }
        ...
        ...     for h, f in fc.items():
        ...         pat = ['A', 'A', 'B', 'B', 'A'] * (len(f.tracking.data) // 5 + 1)
        ...         states = pd.Series(pat[:len(f.tracking.data)], index=f.tracking.data.index)
        ...         f.store(states, 'state', meta={})
        ...         condition, time = tags[h]
        ...         f.tracking.add_tag('condition', condition)
        ...         f.tracking.add_tag('time', time)
        ...
        ...     sc = SummaryCollection.from_features_collection(fc.groupby(['condition', 'time']))
        ...
        ...     fig, ax = sc.plot_transition_umap(
        ...         column='state',
        ...         all_states=['A', 'B'],
        ...         groups=[
        ...             [('control', 'time1'), ('control', 'time2')],
        ...             [('treatment', 'time1'), ('treatment', 'time2')],
        ...         ],
        ...         show=False,
        ...     )
        ...     fig is not None and ax is not None
        True

        ```
        """
        import os

        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        try:
            import umap  # type: ignore
        except Exception as e:
            raise ImportError("UMAP is required for this plot. Please install 'umap-learn'.") from e

        if not getattr(self, "is_grouped", False):
            raise ValueError("UMAP plot requires a grouped SummaryCollection (call groupby first).")

        # Compute transition matrices per subject per group
        matrices_result = self.each.transition_matrix(column, all_states)
        matrices = {
            group: {k: v.value for k, v in d.items()} for group, d in matrices_result.items()
        }

        # Helpers to format group labels for nicer display/selection
        def _fmt_group(g):
            if isinstance(g, tuple) and len(g) == 1:
                return g[0]
            return g

        label_to_key = {_fmt_group(g): g for g in self.group_keys}

        # Determine which groups to include (supports sequential groups via nested lists)
        sequence_mode = False
        sequences: list[list[str]] = []
        if groups is None:
            selected = list(matrices.keys())
        else:
            if any(isinstance(g, (list)) for g in groups):
                # sequence mode
                sequence_mode = True
                sequences = [list(seq) for seq in groups]  # type: ignore[arg-type]
                selected = []
                for seq in sequences:
                    for lbl in seq:
                        key = label_to_key.get(lbl, None)
                        if key is None:
                            key = lbl
                        if key in matrices and key not in selected:
                            selected.append(key)
            else:
                # flat list of labels
                selected = []
                for lbl in groups:  # type: ignore[assignment]
                    key = label_to_key.get(lbl, None)
                    if key is None:
                        key = lbl
                    if key in matrices:
                        selected.append(key)

        # Flatten matrices and collect labels
        X, y = [], []
        for g in selected:
            for _, mat in matrices[g].items():
                X.append(mat.to_numpy().flatten())
                y.append(_fmt_group(g))
        if len(X) == 0:
            raise ValueError("No data found for the requested groups.")
        X = np.vstack(X)

        # Scale and embed
        X_scaled = StandardScaler().fit_transform(X)
        # guard for very small sample sizes to avoid eigsh issues in UMAP spectral init
        n_samples = X_scaled.shape[0]
        effective_neighbors = min(n_neighbors, max(2, n_samples - 1))
        reducer = umap.UMAP(
            n_neighbors=effective_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        import warnings

        # UMAP warns that random_state forces single-threaded execution.
        # That is expected here for reproducibility, so suppress this one warning.
        _umap_njobs_warn = (
            r"n_jobs value .* overridden to 1 by setting random_state\. "
            r"Use no seed for parallelism\."
        )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=_umap_njobs_warn,
                    category=UserWarning,
                )
                embedding = reducer.fit_transform(X_scaled)
        except TypeError:
            # fallback to random init if spectral layout fails for very small graphs
            reducer = umap.UMAP(
                n_neighbors=effective_neighbors,
                min_dist=min_dist,
                random_state=random_state,
                init="random",
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=_umap_njobs_warn,
                    category=UserWarning,
                )
                embedding = reducer.fit_transform(X_scaled)

        # Plot
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.set_facecolor("white")
        # Colors: either simple cycle (flat list) or per-sequence monochrome gradient
        unique_groups = list(dict.fromkeys(y))  # preserve order
        color_map = {}
        if not sequence_mode:
            base_colors = plt.cm.tab10.colors
            color_map = {g: base_colors[i % len(base_colors)] for i, g in enumerate(unique_groups)}
        else:
            base_colors = list(plt.cm.tab10.colors)
            # build a mapping from label -> color shade based on its position in its sequence
            label_to_color = {}
            for si, seq in enumerate(sequences):
                base = np.array(base_colors[si % len(base_colors)])
                L = max(1, len(seq))
                for pi, lbl in enumerate(seq):
                    # t from 0.0 to 0.8 across the sequence to produce a lightening gradient
                    t = 0.8 * (pi / max(L - 1, 1))
                    shade = (1 - t) * base + t * np.array([1.0, 1.0, 1.0])
                    label_to_color[lbl] = tuple(shade)
            # resolve colors for observed group labels (already formatted)
            for g in unique_groups:
                color_map[g] = label_to_color.get(g, base_colors[0])
        for g in unique_groups:
            mask = [gi == g for gi in y]
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                label=g,
                alpha=0.9,
                color=color_map[g],
            )
        # Group means and SEMs
        import pandas as pd  # local alias for clarity

        embedding_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
        embedding_df["group"] = y

        def _sem(arr):
            return np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0

        group_stats = (
            embedding_df.groupby("group")
            .agg(
                mean_x=("UMAP1", "mean"),
                mean_y=("UMAP2", "mean"),
                sem_x=("UMAP1", _sem),
                sem_y=("UMAP2", _sem),
            )
            .reset_index()
        )
        for _, row in group_stats.iterrows():
            color = color_map.get(row["group"], "gray")
            ax.errorbar(
                row["mean_x"],
                row["mean_y"],
                xerr=row["sem_x"],
                yerr=row["sem_y"],
                fmt="x",
                color=color,
                linewidth=2,
                capsize=5,
            )
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.legend(title="Group", loc="best")
        plt.tight_layout()
        if show:
            plt.show()
        if save_dir:
            fig.savefig(
                os.path.join(save_dir, "transition_umap.png"),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.02,
            )
        return fig, ax

    @staticmethod
    def _manhattan_distance(
        transition_matrix1: pd.DataFrame, transition_matrix2: pd.DataFrame
    ) -> float:
        # check that transition_matrix1 and transition_matrix2 have the same index and columns
        if not transition_matrix1.index.equals(transition_matrix2.index):
            raise ValueError("transition_matrix1 and transition_matrix2 must have the same index")
        if not transition_matrix1.columns.equals(transition_matrix2.columns):
            raise ValueError("transition_matrix1 and transition_matrix2 must have the same columns")
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
    def _shuffle_lists(group1: list, group2: list, rng) -> tuple[list, list]:
        n1 = len(group1)
        combined = group1 + group2
        rng.shuffle(combined)
        new_group1 = combined[:n1]
        new_group2 = combined[n1:]
        return new_group1, new_group2

    @staticmethod
    def combine_bfa_results(
        results_list: list[dict],
        *,
        scale_weights: list[float] | None = None,
        per_scale: bool = True,
    ) -> dict:
        """
        Combine BFA results from multiple temporal scales into a single result.

        .. note::
            **This is an escape-hatch for advanced workflows.**  If you are
            starting a multi-scale BFA from scratch, use :meth:`bfa_multiscale`
            instead — it handles scale generation, surrogate synchronisation,
            and result combination automatically.

            Only use this helper directly when you have already computed
            per-scale results through a custom pipeline and know that their
            surrogate shuffles are synchronised (same ``random_state``, same
            group/handle order, same ``pairs``, same ``numshuffles``).

        Each entry in ``results_list`` is a dict returned by :meth:`bfa`.  The
        ``observed`` distances and the per-surrogate surrogate distances are
        summed (optionally weighted) across scales, yielding a combined result
        in the same format as a single :meth:`bfa` call.

        For valid multi-scale statistics the surrogate shuffles must be
        synchronised across scales — pass the same ``random_state`` to every
        :meth:`bfa` call, use the same group structure and the same ``pairs``
        ordering, and the shuffles will be identical by construction.

        Parameters
        ----------
        results_list : list[dict]
            BFA result dicts, one per scale, in the same format returned by
            :meth:`bfa`.  All dicts must contain the same pair keys and the
            same number of surrogates.
        scale_weights : list[float] | None
            Optional per-scale multiplicative weights (must have the same length
            as ``results_list``).  Defaults to uniform weighting (all 1.0).
        per_scale : bool, default True
            If True, each combined pair entry includes a ``"per_scale_observed"``
            list containing the individual scale contributions.

        Returns
        -------
        dict
            Same structure as :meth:`bfa` output, with an optional extra key
            ``"per_scale_observed"`` per comparison when ``per_scale=True``.

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
        ...     pat = ['A','A','B','B','A'] * (len(f.tracking.data)//5 + 1)
        ...     states = pd.Series(pat[:len(f.tracking.data)], index=f.tracking.data.index)
        ...     f.store(states, 'state', meta={})
        ...     f.tracking.add_tag('group', f'G{i+1}')
        >>> gfc = fc.groupby('group')
        >>> sc1 = SummaryCollection.from_features_collection(gfc)
        >>> sc4 = SummaryCollection.from_features_collection(
        ...     gfc.each.coarse_grain(4, non_numeric='mode'))
        >>> res1 = sc1.bfa('state', all_states=['A','B'], numshuffles=2, random_state=0)
        >>> res4 = sc4.bfa('state', all_states=['A','B'], numshuffles=2, random_state=0)
        >>> combined = SummaryCollection.combine_bfa_results([res1, res4])
        >>> 'observed' in combined['G1_vs_G2'] and 'surrogates' in combined['G1_vs_G2']
        True
        >>> len(combined['G1_vs_G2']['surrogates']) == 2
        True
        >>> 'per_scale_observed' in combined['G1_vs_G2']
        True
        >>> len(combined['G1_vs_G2']['per_scale_observed']) == 2
        True

        ```
        """
        if not results_list:
            raise ValueError("results_list must not be empty")

        if scale_weights is not None:
            if len(scale_weights) != len(results_list):
                raise ValueError(
                    f"scale_weights has {len(scale_weights)} entries but "
                    f"results_list has {len(results_list)}"
                )
            weights = list(scale_weights)
        else:
            weights = [1.0] * len(results_list)

        pair_keys = list(results_list[0].keys())
        # validate all scales expose the same pair keys
        for i, r in enumerate(results_list[1:], start=1):
            if list(r.keys()) != pair_keys:
                raise ValueError(
                    f"results_list[{i}] has different pair keys than results_list[0]: "
                    f"{list(r.keys())} vs {pair_keys}"
                )

        n_surrogates = len(results_list[0][pair_keys[0]]["surrogates"])
        for i, r in enumerate(results_list):
            for key in pair_keys:
                n = len(r[key]["surrogates"])
                if n != n_surrogates:
                    raise ValueError(
                        f"results_list[{i}]['{key}'] has {n} surrogates; expected {n_surrogates}"
                    )

        combined = {}
        for key in pair_keys:
            scale_observed = [
                w * r[key]["observed"] for w, r in zip(weights, results_list, strict=True)
            ]
            combined_observed = sum(scale_observed)
            combined_surrogates = [
                sum(w * r[key]["surrogates"][i] for w, r in zip(weights, results_list, strict=True))
                for i in range(n_surrogates)
            ]
            entry: dict = {"observed": combined_observed, "surrogates": combined_surrogates}
            if per_scale:
                entry["per_scale_observed"] = scale_observed
            combined[key] = entry

        return combined

    @staticmethod
    def bfa_multiscale(
        scs: list[SummaryCollection],
        columns: list[str] | str,
        all_states: list[list | None] | list | None = None,
        numshuffles: int = 1000,
        pairs: list[tuple[str, str]] | None = None,
        random_state: int | None = 0,
        scale_by_transitions: bool = True,
        scale_weights: list[float] | None = None,
    ) -> dict:
        """
        Multi-scale Behaviour Flow Analysis across pre-built SummaryCollections.

        Each entry in ``scs`` is an independently prepared grouped
        ``SummaryCollection`` — typically derived from data at a different
        temporal resolution (e.g. raw, 4x coarse-grained, 16x coarse-grained).
        The relevant state column at each scale is specified via ``columns``.

        The state column is expected to have been computed directly on the data
        at that scale (e.g. via cluster labels computed on coarse-grained
        features), not simply aggregated from a finer scale.

        Surrogate shuffles are automatically synchronised: the same
        ``random_state`` is passed to every :meth:`bfa` call, which — given
        that QC has verified identical group/handle order — guarantees that
        surrogate *i* at scale A and surrogate *i* at scale B used the same
        animal shuffle.  The combined surrogate distribution is therefore the
        correct null for the combined statistic.

        Parameters
        ----------
        scs : list[SummaryCollection]
            Grouped ``SummaryCollection`` objects, one per scale, in the order
            they should be combined.
        columns : list[str] or str
            State column name to use for each scale's transition matrix.  Pass
            a single string to use the same column name for all scales.
        all_states : list[list | None] | list | None, default None
            Explicit state ordering for transition matrices.  Three forms are
            accepted:

            - ``None`` — states are inferred from the data at every scale.
            - A flat list (e.g. ``[0, 1, 2]``) — the same state set is used
              for all scales.
            - A list of lists / ``None`` values whose length equals ``len(scs)``
              (e.g. ``[[0,...,49], [0,...,9], None]``) — each scale uses its
              own state set, or ``None`` to infer for that scale.

            The per-scale form is detected when every element of the outer list
            is itself a list or ``None``.
        numshuffles : int
            Number of surrogate shuffles per scale.
        pairs : list[tuple[str, str]] | None
            Group pairs to compare.  ``None`` evaluates all unique pairs.
            Must be the same for all scales.
        random_state : int | None
            Seed for reproducible surrogate shuffling.  The same seed is used
            at every scale to synchronise surrogates.
        scale_by_transitions : bool, default True
            Divide each pairwise Manhattan distance by the total number of
            transitions across both groups for that pair.  This is enabled by
            default here because distances across scales must be on a common
            per-transition unit before they can be meaningfully combined.
            Set to ``False`` only if you need raw-count distances and are
            handling comparability yourself.
        scale_weights : list[float] | None
            Per-scale multipliers for the combined distance, in the same order
            as ``scs``.  Defaults to uniform weighting.

        Returns
        -------
        dict with keys:

        - ``"combined"`` : combined result in :meth:`bfa` format, with an
          additional ``"per_scale_observed"`` list per comparison.
        - ``"scales"`` : dict mapping integer index (0, 1, …) to the
          individual :meth:`bfa` result for that scale.

        Raises
        ------
        ValueError
            If any SC is not grouped, group keys / handle order differ across
            SCs, a requested column is missing, or ``columns`` length mismatches
            ``scs``.

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
        ...     pat = ['A','A','B','B','A'] * (len(f.tracking.data)//5 + 1)
        ...     states = pd.Series(pat[:len(f.tracking.data)], index=f.tracking.data.index)
        ...     f.store(states, 'state', meta={})
        ...     f.tracking.add_tag('group', f'G{i+1}')
        >>> gfc = fc.groupby('group')
        >>> sc1 = SummaryCollection.from_features_collection(gfc)
        >>> # sc2 would normally come from independently-clustered coarse-grained data;
        >>> # here we reuse sc1 with the same column purely for doctest purposes.
        >>> # shared all_states (broadcast form)
        >>> ms = SummaryCollection.bfa_multiscale(
        ...     [sc1, sc1], 'state', all_states=['A', 'B'], numshuffles=2)
        >>> # per-scale all_states (list-of-lists form)
        >>> ms2 = SummaryCollection.bfa_multiscale(
        ...     [sc1, sc1], 'state',
        ...     all_states=[['A', 'B'], ['A', 'B']],
        ...     numshuffles=2)
        >>> bool(ms['combined']['G1_vs_G2']['observed'] == ms2['combined']['G1_vs_G2']['observed'])
        True
        >>> set(ms.keys()) == {'combined', 'scales'}
        True
        >>> set(ms['scales'].keys()) == {0, 1}
        True
        >>> 'observed' in ms['combined']['G1_vs_G2']
        True
        >>> 'per_scale_observed' in ms['combined']['G1_vs_G2']
        True
        >>> len(ms['combined']['G1_vs_G2']['per_scale_observed']) == 2
        True

        ```
        """
        if not scs:
            raise ValueError("scs must not be empty")

        # Normalise columns to a per-scale list
        if isinstance(columns, str):
            columns_list = [columns] * len(scs)
        else:
            columns_list = list(columns)
            if len(columns_list) != len(scs):
                raise ValueError(f"columns has {len(columns_list)} entries but scs has {len(scs)}")

        # Normalise all_states to a per-scale list.
        # Per-scale form is detected when every element is a list or None.
        def _is_per_scale_states(v):
            return (
                isinstance(v, list)
                and len(v) > 0
                and all(isinstance(s, (list, type(None))) for s in v)
            )

        if _is_per_scale_states(all_states):
            all_states_list = list(all_states)
            if len(all_states_list) != len(scs):
                raise ValueError(
                    f"all_states has {len(all_states_list)} entries but scs has {len(scs)}"
                )
        else:
            all_states_list = [all_states] * len(scs)

        # --- QC ---
        # 1. All SCs must be grouped
        for i, sc in enumerate(scs):
            if not getattr(sc, "is_grouped", False):
                raise ValueError(
                    f"scs[{i}] is not grouped. "
                    "Call .groupby() on the FeaturesCollection before building the SC."
                )

        # 2. Group keys must be identical and in the same order
        ref_group_keys = list(scs[0].keys())
        for i, sc in enumerate(scs[1:], start=1):
            sc_keys = list(sc.keys())
            if sc_keys != ref_group_keys:
                raise ValueError(
                    f"scs[{i}] has different or differently-ordered group keys "
                    f"than scs[0].\n  scs[0]:   {ref_group_keys}\n  scs[{i}]: {sc_keys}"
                )

        # 3. Within each group, handles must be identical and in the same order
        for gkey in ref_group_keys:
            ref_handles = list(scs[0][gkey].keys())
            for i, sc in enumerate(scs[1:], start=1):
                sc_handles = list(sc[gkey].keys())
                if sc_handles != ref_handles:
                    raise ValueError(
                        f"scs[{i}]['{gkey}'] has different or differently-ordered handles "
                        f"than scs[0]['{gkey}'].\n"
                        f"  scs[0]:   {ref_handles}\n  scs[{i}]: {sc_handles}\n"
                        "Handle order must match for surrogate synchronisation to be valid."
                    )

        # 4. Each requested column must exist (spot-check first animal per group)
        for i, (sc, col) in enumerate(zip(scs, columns_list, strict=True)):
            gkey = ref_group_keys[0]
            handle = list(sc[gkey].keys())[0]
            summary = sc[gkey][handle]
            if col not in summary.features.data.columns:
                available = sorted(summary.features.data.columns)
                raise ValueError(
                    f"Column '{col}' not found in scs[{i}]['{gkey}']['{handle}'].features.data. "
                    f"Available columns: {available}"
                )

        # --- Run bfa at each scale with the same seed ---
        scale_results: dict[int, dict] = {}
        for idx, (sc, col, states) in enumerate(
            zip(scs, columns_list, all_states_list, strict=True)
        ):
            scale_results[idx] = sc.bfa(
                col,
                all_states=states,
                numshuffles=numshuffles,
                pairs=pairs,
                random_state=random_state,
                scale_by_transitions=scale_by_transitions,
            )

        combined = SummaryCollection.combine_bfa_results(
            [scale_results[i] for i in range(len(scs))],
            scale_weights=scale_weights,
            per_scale=True,
        )

        return {"combined": combined, "scales": scale_results}

    # ---- Chord diagram (state transitions) ----
    def plot_chord(
        self,
        column: str,
        all_states: list[str | int] | None = None,
        *,
        fromkey: str | None = None,
        plot_individual: bool = False,
        show: bool = True,
        save_dir: str | None = None,
        cmap: str | list | None = None,
        **kwargs,
    ):
        """
        Plot chord diagrams of state transitions using a minimal pattern.

        - If not grouped:
          - plot_individual=False: sum over the collection and plot a single chord.
          - plot_individual=True: plot one chord per recording.
        - If grouped:
          - plot_individual=False: sum within each group and plot one chord per group.
          - plot_individual=True: plot one chord per recording per group.

        Parameters
        ----------
        column:
            Name of the categorical column used to compute transitions.
        all_states:
            Optional explicit state ordering for transition matrices.
            Required when `fromkey` is not provided.
        fromkey:
            Optional key in each `Summary.data` containing a precomputed transition DataFrame.
            If provided, this key is used directly instead of computing transitions from `column`.
        plot_individual:
            If True, plot per recording; otherwise plot summed aggregate.
        show:
            If True, display figures.
        save_dir:
            Optional directory to save figures; created if missing.
        kwargs:
            Additional keyword arguments to pass to pycirclize.Circos.chord_diagram.

        Returns
        -------
        object:
            - flat & plot_individual=False: single fig
            - flat & plot_individual=True: dict {handle: fig}
            - grouped & plot_individual=False: dict {group: fig}
            - grouped & plot_individual=True: dict {group: {handle: fig}}

        Examples
        --------
        ```pycon
        >>> # xdoctest: +REQUIRES(module: pycirclize)
        >>> import tempfile, os, shutil
        >>> import pandas as pd
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> from py3r.behaviour.features.features_collection import FeaturesCollection
        >>> from py3r.behaviour.summary.summary_collection import SummaryCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     # create two recordings from the sample csv
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        ...     # build features and inject a simple 3-state sequence
        ...     fc = FeaturesCollection.from_tracking_collection(tc)
        ...     for _, f in fc.items():
        ...         pat = ['0','1','2','1','0'] * (len(f.tracking.data)//5 + 1)
        ...         seq = pd.Series(pat[:len(f.tracking.data)], index=f.tracking.data.index)
        ...         f.store(seq, 'state', meta={})
        ...     sc = SummaryCollection.from_features_collection(fc)
        ...     # plot flat aggregate and save it
        ...     with tempfile.TemporaryDirectory() as outdir:
        ...         _ = sc.plot_chord(
        ...             'state', all_states=['0','1','2'], show=False, save_dir=outdir)
        ...         os.path.exists(os.path.join(outdir, 'chord_state.png'))
        True

        ```
        """
        import os

        import matplotlib.pyplot as plt

        try:
            from pycirclize import Circos
        except ImportError as err:
            raise ImportError(
                "pycirclize is required for chord diagram plotting. "
                "Please install: 'pip install pycirclize'."
            ) from err

        def _sanitize(name: str) -> str:
            return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(name))

        # Build stable global label -> color mapping from chosen palette
        def _base_colors_for_n(n_states: int):
            import matplotlib.pyplot as _plt

            if n_states <= 10:
                return list(_plt.cm.tab10.colors)
            elif n_states <= 20:
                return list(_plt.cm.tab20.colors)
            else:
                base = list(_plt.cm.tab20.colors)
                return base

        if cmap is None:
            base_colors = _base_colors_for_n(len(all_states))
        elif isinstance(cmap, (list, tuple)):
            base_colors = list(cmap)
            if len(base_colors) == 0:
                base_colors = _base_colors_for_n(len(all_states))
        elif isinstance(cmap, str):
            import matplotlib.pyplot as _plt

            cm = _plt.cm.get_cmap(cmap)
            if hasattr(cm, "colors") and getattr(cm, "colors", None):
                base_colors = list(cm.colors)
            else:
                k = max(len(all_states), 10)
                base_colors = [cm(i / max(k - 1, 1)) for i in range(k)]
        else:
            base_colors = _base_colors_for_n(len(all_states))

        label_to_color = {
            str(lbl): base_colors[i % len(base_colors)] for i, lbl in enumerate(all_states)
        }

        def _matrix_from_summary(summary: Summary, handle: str) -> pd.DataFrame:
            if fromkey is None:
                return summary.transition_matrix(column, all_states=all_states).value
            if fromkey not in summary.data:
                raise KeyError(
                    f"fromkey '{fromkey}' not found in summary.data for handle '{handle}'"
                )
            matrix = summary.data[fromkey]
            if not isinstance(matrix, pd.DataFrame):
                raise TypeError(
                    f"summary.data['{fromkey}'] must be a pandas DataFrame for plot_chord, "
                    f"got {type(matrix).__name__} in handle '{handle}'"
                )
            return matrix

        if fromkey is None and all_states is None:
            raise ValueError("all_states must be provided when fromkey is not used.")
        if fromkey is not None and all_states is None:
            inferred_states = []
            seen = set()
            for handle, summary in self.flatten().items():
                df = _matrix_from_summary(summary, handle)
                for lbl in list(df.index) + list(df.columns):
                    if lbl not in seen:
                        seen.add(lbl)
                        inferred_states.append(lbl)
            all_states = inferred_states

        def _warn_if_misaligned(per: dict[str, pd.DataFrame], context_label: str) -> None:
            if len(per) <= 1:
                return
            first_handle, first_df = next(iter(per.items()))
            ref_index = first_df.index
            ref_columns = first_df.columns
            mismatched = []
            for h, df in per.items():
                if not ref_index.equals(df.index) or not ref_columns.equals(df.columns):
                    mismatched.append(h)
            if mismatched:
                warnings.warn(
                    f"plot_chord found mismatched transition matrix labels in {context_label}. "
                    f"Summing may align by labels and produce unexpected results. "
                    f"Reference handle: '{first_handle}'. Mismatched handles: {mismatched}",
                    stacklevel=2,
                )

        def _render(df: pd.DataFrame, title: str | None, path: str | None):
            # guard: empty matrix (zero total) -> placeholder
            if float(df.to_numpy().sum()) <= 0.0:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.axis("off")
                msg = "No transitions to plot"
                if title:
                    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=12)
                    ax.text(0.5, 0.38, msg, ha="center", va="center", fontsize=10)
                else:
                    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
                if path:
                    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
                if show:
                    plt.show()
                plt.close(fig)
                return fig
            # prune zero-only states and enforce global label order
            row_sums = df.sum(axis=1)
            col_sums = df.sum(axis=0)
            keep = (row_sums + col_sums) > 0
            present = [lbl for lbl in all_states if lbl in df.index and keep.get(lbl, False)]
            if len(present) == 0:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.axis("off")
                msg = "No transitions to plot"
                if title:
                    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=12)
                    ax.text(0.5, 0.38, msg, ha="center", va="center", fontsize=10)
                else:
                    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
                if path:
                    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
                if show:
                    plt.show()
                plt.close(fig)
                return fig
            df_pruned = df.reindex(index=present, columns=present, fill_value=0)

            # ensure labels are strings for mapping consistency
            df_pruned.index = df_pruned.index.astype(str)
            df_pruned.columns = df_pruned.columns.astype(str)

            cmap_present = {lbl: label_to_color[lbl] for lbl in df_pruned.index}
            kw = dict(kwargs)
            kw.pop("cmap", None)
            circos = Circos.chord_diagram(df_pruned, cmap=cmap_present, **kw)
            fig = circos.plotfig()
            if title:
                try:
                    plt.title(title)
                except Exception:
                    pass
            if path:
                fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
            if show:
                plt.show()
            # proactively close to avoid accumulating many open figures
            plt.close(fig)
            return fig

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        is_grouped = getattr(self, "is_grouped", False)

        # Flat collection
        if not is_grouped:
            # Compute per-recording matrices (aligned via all_states)
            per = {h: _matrix_from_summary(s, h) for h, s in self.items()}
            _warn_if_misaligned(per, "collection")
            if plot_individual:
                out = {}
                for h, df in per.items():
                    path = (
                        os.path.join(save_dir, f"{_sanitize(h)}_chord_{_sanitize(column)}.png")
                        if save_dir
                        else None
                    )
                    out[h] = _render(df, f"{h}: {column}", path)
                return out
            # aggregate sum (labels already aligned by all_states)
            if len(per) == 0:
                raise ValueError("No recordings found.")
            agg = sum(per.values())
            path = os.path.join(save_dir, f"chord_{_sanitize(column)}.png") if save_dir else None
            return _render(agg, f"Sum transitions: {column}", path)

        # Grouped collection
        out: dict = {}
        for g, sub_sc in self.items():
            # sub_sc is a SummaryCollection for the group
            per = {h: _matrix_from_summary(s, h) for h, s in sub_sc.items()}
            _warn_if_misaligned(per, f"group '{g}'")
            if plot_individual:
                inner = {}
                for h, df in per.items():
                    path = (
                        os.path.join(
                            save_dir,
                            f"{_sanitize(g)}_{_sanitize(h)}_chord_{_sanitize(column)}.png",
                        )
                        if save_dir
                        else None
                    )
                    inner[h] = _render(df, f"{g} · {h}: {column}", path)
                out[g] = inner
            else:
                if len(per) == 0:
                    continue
                agg = sum(per.values())
                path = (
                    os.path.join(save_dir, f"{_sanitize(g)}_chord_{_sanitize(column)}.png")
                    if save_dir
                    else None
                )
                out[g] = _render(agg, f"{g}: sum transitions · {column}", path)
        return out
