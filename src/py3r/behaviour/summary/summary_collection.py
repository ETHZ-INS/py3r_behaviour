from __future__ import annotations

import pandas as pd

from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.summary.summary import Summary
from py3r.behaviour.summary.summary_collection_batch_mixin import (
    SummaryCollectionBatchMixin,
)
from py3r.behaviour.summary.summary_result import SummaryResult
from py3r.behaviour.util.base_collection import BaseCollection


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
    def from_features_collection(cls, features_collection: FeaturesCollection, summary_cls=Summary):
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
        Collate scalar values (numeric, string, bool) from each Summary.data into
        a pandas DataFrame.

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
        binned = {k: v.make_bin(startframe, endframe) for k, v in self.summary_dict.items()}
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
    def bfa(
        self,
        column: str,
        all_states=None,
        numshuffles: int = 1000,
        pairs: list[tuple[str, str]] | None = None,
    ):
        """
        Behaviour Flow Analysis between groups for a grouped SummaryCollection.

        Requires the collection to be grouped (via groupby). Computes transition
        matrices per Summary within each group, then computes Manhattan distances
        between group means and surrogate distributions via shuffling.

        If `pairs` is provided, only those group pairs are analyzed; otherwise all
        unique pairs in `self.group_keys` are evaluated.

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
        >>> # compute all pairs
        >>> res = sc.bfa('state', all_states=['A','B'], numshuffles=2)
        >>> isinstance(res, dict) and 'observed' in next(iter(res.values()))
        True
        >>> # compute only specific pair(s)
        >>> res2 = sc.bfa('state', all_states=['A','B'], numshuffles=2, pairs=[('G1','G2')])
        >>> list(res2.keys()) == ['G1_vs_G2']
        True

        ```
        """
        if not getattr(self, "is_grouped", False):
            raise ValueError("bfa requires a grouped SummaryCollection (call groupby first)")

        from itertools import combinations

        # batch calculate transition matrix for each summary object
        transition_matrices_result = self.transition_matrix(column, all_states)
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
            _["observed"] = self._manhattan_distance_twogroups(list1, list2)
            _["surrogates"] = [
                self._shuffle_lists(*self._shuffle_lists(list1, list2))
                and self._manhattan_distance_twogroups(*self._shuffle_lists(list1, list2))
                for i in range(numshuffles)
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
        groups: list[str] | list[list[str]] | None = None,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 0,
        figsize: tuple[float, float] = (4.5, 4),
        show: bool = True,
        save_dir: str | None = None,
    ):
        """
        Plot a simple UMAP embedding of per-subject transition matrices for selected groups.

        Parameters
        ----------
        column:
            Name of the categorical column used to compute transition matrices.
        all_states:
            Optional explicit state ordering for transition matrices.
        groups:
            - Optional list of group keys (strings) to include; defaults to all.
            - Or a list of lists for sequential groups, e.g.
              ``[['control_pre','control_45min','control_90min'],
              ['treatment_pre','treatment_45min','treatment_90min']]``.
              Each sequence is plotted with a monochrome gradient.
        n_neighbors, min_dist, random_state:
            UMAP hyperparameters.
        figsize, show:
            Matplotlib options.

        Returns
        -------
        (fig, ax): Matplotlib figure and axis.

        Examples
        --------
        ```pycon
        >>> # xdoctest: +REQUIRES(module: umap)
        >>> import tempfile, shutil, os, pandas as pd
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
        >>> for i, (h, f) in enumerate(fc.items()):
        ...     pat = ['A','A','B','B','A'] * (len(f.tracking.data)//5 + 1)
        ...     states = pd.Series(pat[:len(f.tracking.data)], index=f.tracking.data.index)
        ...     f.store(states, 'state', meta={})
        ...     f.tracking.add_tag('group', f'G{i+1}')
        >>> sc = SummaryCollection.from_features_collection(fc.groupby('group'))
        >>> with tempfile.TemporaryDirectory() as outdir:
        ...     fig, ax = sc.plot_transition_umap(
        ...         column='state', all_states=['A','B'], groups=['G1','G2'],
        ...         show=False, save_dir=outdir)
        ...     os.path.exists(os.path.join(outdir, 'transition_umap.png'))
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
        matrices_result = self.transition_matrix(column, all_states)
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
            if any(isinstance(g, (list, tuple)) for g in groups):
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
        try:
            embedding = reducer.fit_transform(X_scaled)
        except TypeError:
            # fallback to random init if spectral layout fails for very small graphs
            reducer = umap.UMAP(
                n_neighbors=effective_neighbors,
                min_dist=min_dist,
                random_state=random_state,
                init="random",
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
    def _shuffle_lists(group1: list, group2: list) -> tuple[list, list]:
        import random

        n1 = len(group1)
        combined = group1 + group2
        random.shuffle(combined)
        new_group1 = combined[:n1]
        new_group2 = combined[n1:]
        return new_group1, new_group2

    # ---- Chord diagram (state transitions) ----
    def plot_chord(
        self,
        column: str,
        all_states: list[str | int],
        *,
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
            Explicit state ordering for transition matrices (required).
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

        if all_states is None:
            raise ValueError("all_states must be provided to ensure aligned matrices.")

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
            per = {
                h: s.transition_matrix(column, all_states=all_states).value for h, s in self.items()
            }
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
            per = {
                h: s.transition_matrix(column, all_states=all_states).value
                for h, s in sub_sc.items()
            }
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

    # -------------------------------------------------------------------------
    # Seaborn plotting wrappers
    # -------------------------------------------------------------------------

    @staticmethod
    def _gkey_to_label(gkey):
        """Format a group key tuple as a clean string label."""
        if isinstance(gkey, tuple):
            if len(gkey) == 1:
                return str(gkey[0])
            return ", ".join(str(x) for x in gkey)
        return str(gkey)

    @staticmethod
    def _rank_tuple(parts, tags, group_order):
        """Build a sort-key tuple from *parts* using *group_order* for *tags*.

        For each position, if the tag appears in *group_order*, the rank is the
        index within that list (unlisted values sort last).  Otherwise falls
        back to numeric-aware ordering.
        """
        ranks = []
        for i, tag in enumerate(tags):
            val = parts[i] if i < len(parts) else ""
            if tag in group_order:
                try:
                    ranks.append(group_order[tag].index(val))
                except ValueError:
                    ranks.append(len(group_order[tag]))
            else:
                try:
                    ranks.append((0, float(val)))
                except (ValueError, TypeError):
                    ranks.append((1, str(val)))
        return ranks

    @staticmethod
    def _sort_group_labels(label_to_tuple, group_order=None, groupby_tags=None):
        """
        Return group labels sorted according to *group_order*.

        Parameters
        ----------
        label_to_tuple : dict[str, tuple]
            Mapping of display label → raw group-key tuple.
        group_order : dict[str, list] | None
            ``{tag_name: [val, ...]}`` giving the desired order per tag.
            Tags not present fall back to :meth:`_smart_sort_labels`.
        groupby_tags : list[str] | None
            Tag names corresponding to tuple positions (from ``groupby(tags=...)``).

        Returns
        -------
        list[str]
            Group labels in the desired order.
        """
        if not group_order or not groupby_tags:
            return SummaryCollection._smart_sort_labels(label_to_tuple.keys())

        def _key(label):
            return SummaryCollection._rank_tuple(label_to_tuple[label], groupby_tags, group_order)

        return sorted(label_to_tuple.keys(), key=_key)

    @staticmethod
    def _build_group_palette(label_to_tuple, group_order=None, groupby_tags=None):
        """
        Build a colour palette from group labels mapped to their raw tuples.

        For multi-tag groups the first n-1 tuple elements choose a base colour
        (from ``tab10``) and the last element creates a light-to-dark gradient
        within that base.  Single-tag groups return ``None`` (seaborn default).

        When *group_order* and *groupby_tags* are provided, the ordering of
        base-colour groups (first n-1 tags) and shade values (last tag) respects
        the user-specified order.
        """
        import colorsys

        import matplotlib.pyplot as plt

        max_depth = max(len(t) for t in label_to_tuple.values())
        if max_depth <= 1:
            return None

        # Resolve base-group ordering (first n-1 tags)
        raw_bases = list(set(t[:-1] for t in label_to_tuple.values()))
        if group_order and groupby_tags and len(groupby_tags) > 1:
            base_tags = groupby_tags[:-1]
            base_groups = sorted(
                raw_bases,
                key=lambda bg: SummaryCollection._rank_tuple(bg, base_tags, group_order),
            )
        else:
            base_groups = sorted(raw_bases)

        base_colors = plt.cm.tab10.colors
        base_map = {bg: base_colors[i % len(base_colors)] for i, bg in enumerate(base_groups)}

        # Resolve shade ordering (last tag)
        raw_shades = set(t[-1] for t in label_to_tuple.values())
        last_tag = groupby_tags[-1] if groupby_tags else None
        if group_order and last_tag and last_tag in group_order:
            specified = group_order[last_tag]
            shade_values = [v for v in specified if v in raw_shades]
            # Append any values not in the specified order
            shade_values += SummaryCollection._smart_sort_labels(raw_shades - set(shade_values))
        else:
            shade_values = SummaryCollection._smart_sort_labels(raw_shades)
        n_shades = len(shade_values)

        palette = {}
        for label, parts in label_to_tuple.items():
            base_rgb = base_map[parts[:-1]]
            shade_idx = shade_values.index(parts[-1])

            # Blend from light (towards white) to full colour
            if n_shades > 1:
                t = 0.35 + (shade_idx / (n_shades - 1)) * 0.65
            else:
                t = 0.8

            # Adjust saturation via HLS so hue stays stable
            h, _l, s = colorsys.rgb_to_hls(*base_rgb[:3])
            lightness = 0.85 - t * 0.45  # range ~0.85 (light) → ~0.40 (dark)
            r, g, b = colorsys.hls_to_rgb(h, lightness, s)
            palette[label] = (r, g, b)

        return palette

    def _metric_to_tidy(
        self, metric, group_order=None
    ) -> tuple[pd.DataFrame, str, dict | None, list | None]:
        """
        Convert a metric (string key or BatchResult) to a tidy (long-form) DataFrame.

        Parameters
        ----------
        metric : str or BatchResult
            Metric to convert.
        group_order : dict[str, list] | None
            ``{tag_name: [value, ...]}`` controlling group display order.

        Returns
        -------
        tuple[pd.DataFrame, str, dict | None, list | None]
            - DataFrame with columns: _handle, _group (if grouped), component, value
            - metric_name string for labeling
            - palette dict (label → RGB) or None
            - sorted group labels list or None
        """
        from py3r.behaviour.util.collection_utils import BatchResult

        flat_self = self.flatten()
        is_grouped = getattr(self, "is_grouped", False)

        # Extract data based on metric type
        metric_name = None
        if isinstance(metric, str):
            metric_name = metric
            if is_grouped:
                data_map = {}
                for gkey, subcoll in self.items():
                    data_map[gkey] = {}
                    for handle, summary in subcoll.items():
                        if metric not in summary.data:
                            raise KeyError(f"Metric '{metric}' not found in Summary '{handle}'")
                        data_map[gkey][handle] = summary.data[metric]
            else:
                data_map = {}
                for handle, summary in flat_self.items():
                    if metric not in summary.data:
                        raise KeyError(f"Metric '{metric}' not found in Summary '{handle}'")
                    data_map[handle] = summary.data[metric]

        elif isinstance(metric, (dict, BatchResult)):
            raw = dict(metric) if isinstance(metric, BatchResult) else metric
            first_val = next(iter(raw.values()))
            if isinstance(first_val, dict):
                # Grouped structure: {group: {handle: SummaryResult}}
                is_grouped = True
                data_map = {}
                for gkey, subdict in raw.items():
                    data_map[gkey] = {}
                    for handle, sr in subdict.items():
                        val = sr.value if hasattr(sr, "value") else sr
                        data_map[gkey][handle] = val
                        if metric_name is None and hasattr(sr, "name"):
                            metric_name = sr.name
            else:
                # Flat structure: {handle: SummaryResult}
                data_map = {}
                for handle, sr in raw.items():
                    val = sr.value if hasattr(sr, "value") else sr
                    data_map[handle] = val
                    if metric_name is None and hasattr(sr, "name"):
                        metric_name = sr.name
        else:
            raise TypeError(f"metric must be str or BatchResult/dict, got {type(metric).__name__}")

        if metric_name is None:
            metric_name = "value"

        # Build tidy DataFrame: one row per (handle, component) pair
        # For grouped data, also track raw tuples for palette construction.
        rows = []
        label_to_tuple = {}  # {label_str: raw_tuple}
        if is_grouped:
            for gkey, subdict in data_map.items():
                gkey_tuple = gkey if isinstance(gkey, tuple) else (gkey,)
                gkey_str = self._gkey_to_label(gkey)
                label_to_tuple[gkey_str] = gkey_tuple
                for handle, val in subdict.items():
                    if isinstance(val, pd.Series):
                        for comp, v in val.items():
                            rows.append(
                                {
                                    "_handle": handle,
                                    "_group": gkey_str,
                                    "component": str(comp),
                                    "value": v,
                                }
                            )
                    else:
                        rows.append(
                            {
                                "_handle": handle,
                                "_group": gkey_str,
                                "component": metric_name,
                                "value": float(val),
                            }
                        )
        else:
            for handle, val in data_map.items():
                if isinstance(val, pd.Series):
                    for comp, v in val.items():
                        rows.append({"_handle": handle, "component": str(comp), "value": v})
                else:
                    rows.append(
                        {
                            "_handle": handle,
                            "component": metric_name,
                            "value": float(val),
                        }
                    )

        df = pd.DataFrame(rows)
        if len(df) == 0:
            raise ValueError("No data to plot.")

        # Build palette and sorted group labels from group structure
        groupby_tags = getattr(self, "_groupby_tags", None)
        if label_to_tuple:
            palette = self._build_group_palette(label_to_tuple, group_order, groupby_tags)
            sorted_groups = self._sort_group_labels(label_to_tuple, group_order, groupby_tags)
        else:
            palette = None
            sorted_groups = None

        return df, metric_name, palette, sorted_groups

    # Default sizing constants for seaborn wrappers
    _SNS_HEIGHT = 4.0  # fixed vertical size (inches)
    _SNS_WIDTH_PER_TICK = 0.65  # horizontal inches per x-axis tick position
    _SNS_MIN_WIDTH = 3.0  # minimum figure width (inches)

    @staticmethod
    def _auto_figsize(n_components: int, n_groups: int = 1, figsize=None) -> tuple[float, float]:
        """Compute default figure size.

        For single-component grouped data, the ticks are group names so
        width scales with ``n_groups``.  For multi-component data the ticks
        are component names and groups are dodged within each, so width
        scales with ``n_components``.
        """
        if figsize is not None:
            return figsize
        n_ticks = n_groups if n_components == 1 else n_components
        width = max(
            SummaryCollection._SNS_MIN_WIDTH,
            n_ticks * SummaryCollection._SNS_WIDTH_PER_TICK,
        )
        return (width, SummaryCollection._SNS_HEIGHT)

    @staticmethod
    def _smart_sort_labels(labels):
        """Sort labels numerically when possible, otherwise alphabetically."""

        def _sort_key(label):
            try:
                return (0, float(label))
            except (ValueError, TypeError):
                return (1, str(label))

        return sorted(labels, key=_sort_key)

    @staticmethod
    def _build_sns_kwargs(df, ax, palette=None, sorted_groups=None):
        """
        Build core seaborn plot kwargs based on data shape.

        - **Ungrouped**: ``x=component``, no hue.
        - **Grouped, single component** (scalar metric): ``x=_group``,
          ``hue=_group`` for colouring.  Legend is hidden (info in tick labels).
        - **Grouped, multi-component**: ``x=component``, ``hue=_group``,
          ``dodge=True``.  Seaborn packs groups tightly within each component
          position with natural gaps between components.

        Returns
        -------
        tuple[dict, bool]
            (plot_kwargs, hide_legend)
        """
        is_grouped = "_group" in df.columns
        n_components = df["component"].nunique()

        if not is_grouped:
            components = SummaryCollection._smart_sort_labels(df["component"].unique())
            return {
                "data": df,
                "x": "component",
                "y": "value",
                "order": components,
                "ax": ax,
            }, True

        groups = sorted_groups if sorted_groups is not None else sorted(df["_group"].unique())

        if n_components == 1:
            # Scalar grouped: groups as tick labels, hue for colouring only
            kwargs = {
                "data": df,
                "x": "_group",
                "y": "value",
                "hue": "_group",
                "dodge": False,
                "order": groups,
                "hue_order": groups,
                "ax": ax,
            }
            if palette:
                kwargs["palette"] = palette
            return kwargs, True  # legend redundant (info in tick labels)

        # Multi-component grouped: seaborn dodge handles spacing natively
        components = SummaryCollection._smart_sort_labels(df["component"].unique())
        kwargs = {
            "data": df,
            "x": "component",
            "y": "value",
            "hue": "_group",
            "dodge": True,
            "order": components,
            "hue_order": groups,
            "ax": ax,
        }
        if palette:
            kwargs["palette"] = palette
        return kwargs, False

    # ------------------------------------------------------------------
    # Shared pre/post helpers for seaborn wrappers
    # ------------------------------------------------------------------

    def _sns_pre_plot(self, metric, *, group_order=None, ax=None, figsize=None):
        """Prepare data, figure, and seaborn kwargs for a categorical plot.

        Returns
        -------
        fig, ax, df, metric_name, plot_kwargs, hide_legend, n_components, n_groups
        """
        import matplotlib.pyplot as plt

        df, metric_name, palette, sorted_groups = self._metric_to_tidy(metric, group_order)
        is_grouped = "_group" in df.columns
        n_components = df["component"].nunique()
        n_groups = df["_group"].nunique() if is_grouped else 1

        if ax is None:
            figsize = self._auto_figsize(n_components, n_groups, figsize)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        plot_kwargs, hide_legend = self._build_sns_kwargs(df, ax, palette, sorted_groups)
        return fig, ax, df, metric_name, plot_kwargs, hide_legend, n_components, n_groups

    @staticmethod
    def _sns_post_plot(
        fig,
        ax,
        *,
        metric_name,
        title=None,
        ylabel="Value",
        n_components=1,
        n_groups=1,
        hide_legend=True,
        legend_n_entries=None,
        savedir=None,
        filename=None,
        default_suffix="plot",
        show=True,
    ):
        """Apply styling, manage legend, save/show figure.

        Parameters
        ----------
        legend_n_entries : int | None
            When not None, deduplicate a multi-layer legend by keeping only
            the first *legend_n_entries* handles (used by superplot).
        """
        import os

        import matplotlib.pyplot as plt

        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(title or metric_name)

        # Rotate x-tick labels if many ticks
        n_ticks = n_groups if n_components == 1 else n_components
        if n_ticks > 3:
            ax.tick_params(axis="x", rotation=45)
            for label in ax.get_xticklabels():
                label.set_ha("right")

        # Legend: hide when redundant, otherwise place outside plot
        legend = ax.get_legend()
        if legend is not None:
            if hide_legend:
                legend.remove()
            elif legend_n_entries is not None:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles[:legend_n_entries],
                    labels[:legend_n_entries],
                    title="Group",
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                )
            else:
                legend.set_bbox_to_anchor((1.02, 1))
                legend.set_loc("upper left")

        plt.tight_layout()

        if savedir:
            os.makedirs(savedir, exist_ok=True)
            fname = filename or f"{metric_name}_{default_suffix}.png"
            fig.savefig(os.path.join(savedir, fname), dpi=150, bbox_inches="tight")

        if show:
            plt.show()

    # ------------------------------------------------------------------
    # Main single-function wrapper
    # ------------------------------------------------------------------

    def _sns_plot_common(
        self,
        plot_func,
        metric,
        *,
        group_order: dict | None = None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        ylabel: str = "Value",
        **kwargs,
    ):
        """
        Common wrapper logic for seaborn categorical plots.

        Parameters
        ----------
        plot_func : callable
            The seaborn plotting function (e.g., sns.stripplot).
        metric : str or BatchResult
            Metric to plot.
        group_order : dict[str, list] | None
            Control group display order. Keys are tag names (matching those
            passed to ``groupby(tags=...)``), values are lists of tag values
            in the desired order.  Example::

                group_order={"treatment": ["control", "FST"],
                             "timepoint": ["45m", "1d"]}
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show : bool
            Display the plot.
        savedir : str | None
            Directory to save figure.
        filename : str | None
            Custom filename.
        title : str | None
            Plot title.
        ylabel : str
            Y-axis label.
        **kwargs
            Passed to the seaborn plot function.

        Returns
        -------
        tuple[Figure, Axes, DataFrame]
        """
        # TODO: auto-populate ylabel with units once feature/summary methods
        # carry unit metadata (e.g. "Time (s)", "Distance (m)", "Speed (m/s)").
        # This would require storing a 'units' field in SummaryResult and
        # propagating it through _metric_to_tidy. See also Tracking.meta
        # 'distance_units' for existing precedent.

        fig, ax, df, metric_name, plot_kwargs, hide_legend, n_comp, n_groups = self._sns_pre_plot(
            metric,
            group_order=group_order,
            ax=ax,
            figsize=kwargs.pop("figsize", None),
        )
        plot_kwargs.update(kwargs)
        plot_func(**plot_kwargs)

        self._sns_post_plot(
            fig,
            ax,
            metric_name=metric_name,
            title=title,
            ylabel=ylabel,
            n_components=n_comp,
            n_groups=n_groups,
            hide_legend=hide_legend,
            savedir=savedir,
            filename=filename,
            default_suffix=plot_func.__name__,
            show=show,
        )
        return fig, ax, df

    def snsstrip(
        self,
        metric,
        *,
        group_order: dict | None = None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        **kwargs,
    ):
        """
        Strip plot (jittered scatter) using seaborn.

        Parameters
        ----------
        metric : str or BatchResult
            Either a key from Summary.data, or a BatchResult from a batch method.
        group_order : dict[str, list] | None
            Control group display order.  See :meth:`_sns_plot_common`.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show : bool
            Display the plot. Default True.
        savedir : str | None
            Directory to save figure.
        filename : str | None
            Custom filename.
        title : str | None
            Plot title.
        **kwargs
            Passed to seaborn.stripplot (e.g., jitter, alpha, size, palette).

        Returns
        -------
        tuple[Figure, Axes, DataFrame]

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
        >>> for f in fc.values():
        ...     f.store(pd.Series([True, False] * 15, index=f.tracking.data.index[:30]),
        ...             'active', meta={})
        >>> sc = SummaryCollection.from_features_collection(fc)
        >>> fig, ax, df = sc.snsstrip(sc.time_in_state('active'), show=False)
        >>> isinstance(df, pd.DataFrame)
        True

        ```
        """
        import seaborn as sns

        defaults = {"alpha": 0.7, "jitter": True, "size": 5}
        for k, v in defaults.items():
            kwargs.setdefault(k, v)

        return self._sns_plot_common(
            sns.stripplot,
            metric,
            group_order=group_order,
            ax=ax,
            show=show,
            savedir=savedir,
            filename=filename,
            title=title,
            **kwargs,
        )

    def snsswarm(
        self,
        metric,
        *,
        group_order: dict | None = None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        **kwargs,
    ):
        """
        Swarm plot (non-overlapping scatter) using seaborn.

        Parameters
        ----------
        metric : str or BatchResult
            Either a key from Summary.data, or a BatchResult from a batch method.
        group_order : dict[str, list] | None
            Control group display order.  See :meth:`_sns_plot_common`.
        ax, show, savedir, filename, title
            Save/display options.
        **kwargs
            Passed to seaborn.swarmplot (e.g., size, palette).

        Returns
        -------
        tuple[Figure, Axes, DataFrame]
        """
        import seaborn as sns

        defaults = {"size": 5}
        for k, v in defaults.items():
            kwargs.setdefault(k, v)

        return self._sns_plot_common(
            sns.swarmplot,
            metric,
            group_order=group_order,
            ax=ax,
            show=show,
            savedir=savedir,
            filename=filename,
            title=title,
            **kwargs,
        )

    def snsbar(
        self,
        metric,
        *,
        group_order: dict | None = None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        **kwargs,
    ):
        """
        Bar plot with error bars using seaborn.

        Parameters
        ----------
        metric : str or BatchResult
            Either a key from Summary.data, or a BatchResult from a batch method.
        group_order : dict[str, list] | None
            Control group display order.  See :meth:`_sns_plot_common`.
        ax, show, savedir, filename, title
            Save/display options.
        **kwargs
            Passed to seaborn.barplot (e.g., errorbar, palette, saturation).

        Returns
        -------
        tuple[Figure, Axes, DataFrame]
        """
        import seaborn as sns

        defaults = {"errorbar": "se", "capsize": 0.1}
        for k, v in defaults.items():
            kwargs.setdefault(k, v)

        return self._sns_plot_common(
            sns.barplot,
            metric,
            group_order=group_order,
            ax=ax,
            show=show,
            savedir=savedir,
            filename=filename,
            title=title,
            **kwargs,
        )

    def snsbox(
        self,
        metric,
        *,
        group_order: dict | None = None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        **kwargs,
    ):
        """
        Box plot using seaborn.

        Parameters
        ----------
        metric : str or BatchResult
            Either a key from Summary.data, or a BatchResult from a batch method.
        group_order : dict[str, list] | None
            Control group display order.  See :meth:`_sns_plot_common`.
        ax, show, savedir, filename, title
            Save/display options.
        **kwargs
            Passed to seaborn.boxplot (e.g., width, palette, fliersize).

        Returns
        -------
        tuple[Figure, Axes, DataFrame]
        """
        import seaborn as sns

        return self._sns_plot_common(
            sns.boxplot,
            metric,
            group_order=group_order,
            ax=ax,
            show=show,
            savedir=savedir,
            filename=filename,
            title=title,
            **kwargs,
        )

    def snsviolin(
        self,
        metric,
        *,
        group_order: dict | None = None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        **kwargs,
    ):
        """
        Violin plot using seaborn.

        Parameters
        ----------
        metric : str or BatchResult
            Either a key from Summary.data, or a BatchResult from a batch method.
        group_order : dict[str, list] | None
            Control group display order.  See :meth:`_sns_plot_common`.
        ax, show, savedir, filename, title
            Save/display options.
        **kwargs
            Passed to seaborn.violinplot (e.g., inner, split, palette).

        Returns
        -------
        tuple[Figure, Axes, DataFrame]
        """
        import seaborn as sns

        defaults = {"inner": "box"}
        for k, v in defaults.items():
            kwargs.setdefault(k, v)

        return self._sns_plot_common(
            sns.violinplot,
            metric,
            group_order=group_order,
            ax=ax,
            show=show,
            savedir=savedir,
            filename=filename,
            title=title,
            **kwargs,
        )

    def snspoint(
        self,
        metric,
        *,
        group_order: dict | None = None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        **kwargs,
    ):
        """
        Point plot (mean + CI) using seaborn.

        Parameters
        ----------
        metric : str or BatchResult
            Either a key from Summary.data, or a BatchResult from a batch method.
        group_order : dict[str, list] | None
            Control group display order.  See :meth:`_sns_plot_common`.
        ax, show, savedir, filename, title
            Save/display options.
        **kwargs
            Passed to seaborn.pointplot (e.g., errorbar, markers, linestyles).

        Returns
        -------
        tuple[Figure, Axes, DataFrame]
        """
        import seaborn as sns

        defaults = {"errorbar": "se", "capsize": 0.1, "join": False}
        for k, v in defaults.items():
            kwargs.setdefault(k, v)

        return self._sns_plot_common(
            sns.pointplot,
            metric,
            group_order=group_order,
            ax=ax,
            show=show,
            savedir=savedir,
            filename=filename,
            title=title,
            **kwargs,
        )

    def snssuperplot(
        self,
        metric,
        *,
        group_order: dict | None = None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        ylabel: str = "Value",
        bar_kwargs: dict | None = None,
        strip_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Superplot: bar plot (mean) with strip plot (individual dots) overlay.

        This is the "publication-ready" visualization showing mean bars with
        individual data points scattered on top, commonly used in scientific
        papers.  The dots are constrained within the bar width by default.

        Parameters
        ----------
        metric : str or BatchResult
            Either a key from Summary.data, or a BatchResult from a batch method.
        group_order : dict[str, list] | None
            Control group display order.  See :meth:`_sns_plot_common`.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show : bool
            Display the plot. Default True.
        savedir : str | None
            Directory to save figure.
        filename : str | None
            Custom filename.
        title : str | None
            Plot title.
        ylabel : str
            Y-axis label.
        bar_kwargs : dict | None
            Extra kwargs for barplot (e.g., errorbar, capsize, saturation).
        strip_kwargs : dict | None
            Extra kwargs for stripplot (e.g., alpha, size, jitter).
        **kwargs
            Common kwargs passed to both plots (e.g., palette, dodge).

        Returns
        -------
        tuple[Figure, Axes, DataFrame]

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
        >>> for f in fc.values():
        ...     states = pd.Series(['A', 'B', 'A'] * (len(f.tracking.data)//3 + 1),
        ...                        index=f.tracking.data.index)[:len(f.tracking.data)]
        ...     f.store(states, 'zone', meta={})
        >>> sc = SummaryCollection.from_features_collection(fc)
        >>> fig, ax, df = sc.snssuperplot(sc.time_in_state('zone'), show=False)
        >>> isinstance(df, pd.DataFrame)
        True

        ```
        """
        import seaborn as sns

        fig, ax, df, metric_name, common, hide_legend, n_comp, n_groups = self._sns_pre_plot(
            metric,
            group_order=group_order,
            ax=ax,
            figsize=kwargs.pop("figsize", None),
        )

        # Bar plot (mean + error bars) — base layer
        bar_defaults = {"errorbar": None, "capsize": 0.1, "alpha": 0.7, "zorder": 1}
        bar_kw = {**common, **bar_defaults, **(bar_kwargs or {}), **kwargs}
        sns.barplot(**bar_kw, legend=False)

        # Strip plot (individual dots) — overlay
        strip_defaults = {"alpha": 0.8, "jitter": True, "size": 4, "zorder": 2}
        strip_kw = {**common, **strip_defaults, **(strip_kwargs or {}), **kwargs}
        sns.stripplot(**strip_kw)

        self._sns_post_plot(
            fig,
            ax,
            metric_name=metric_name,
            title=title,
            ylabel=ylabel,
            n_components=n_comp,
            n_groups=n_groups,
            hide_legend=hide_legend,
            legend_n_entries=n_groups,
            savedir=savedir,
            filename=filename,
            default_suffix="superplot",
            show=show,
        )
        return fig, ax, df
