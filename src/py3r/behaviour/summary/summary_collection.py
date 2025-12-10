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
        ...     states = pd.Series(['A','A','B','B','A'] * (len(f.tracking.data)//5 + 1))[:len(f.tracking.data)]
        ...     states.index = f.tracking.data.index
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
                    raise ValueError(
                        f"Invalid group pair ({g1}, {g2}); valid groups: {sorted(map(_fmt_group, self.group_keys))}"
                    )
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
                and self._manhattan_distance_twogroups(
                    *self._shuffle_lists(list1, list2)
                )
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
        ...     states = pd.Series(['A','A','B','B','A'] * (len(f.tracking.data)//5 + 1))[:len(f.tracking.data)]
        ...     states.index = f.tracking.data.index
        ...     f.store(states, 'state', meta={})
        ...     f.tracking.add_tag('group', f'G{i+1}')
        >>> sc = SummaryCollection.from_features_collection(fc.groupby('group'))
        >>> bfa_out = sc.bfa('state', all_states=['A','B'], numshuffles=5)
        >>> # plot a single comparison and save it
        >>> with tempfile.TemporaryDirectory() as outdir:
        ...     fig, ax = SummaryCollection.plot_bfa_results(bfa_out, compare='G1_vs_G2', show=False, save_dir=outdir)
        ...     os.path.exists(os.path.join(outdir, 'G1_vs_G2.png'))
        True

        ```
        """
        import matplotlib.pyplot as plt
        import os

        def _sanitize(name: str) -> str:
            return "".join(
                ch if ch.isalnum() or ch in "-._" else "_" for ch in str(name)
            )

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
            - Optional list of group labels (strings) to include; defaults to all.
            - Or a list of lists for sequential groups, e.g.
              `[['control1','control2','control3'], ['treatment1','treatment2','treatment3']]`
              In this case, each sequence is plotted with a monochrome gradient of a distinct base color.
        n_neighbors, min_dist, random_state:
            UMAP hyperparameters.
        figsize, show:
            Matplotlib options.

        Returns
        -------
        (fig, ax): Matplotlib figure and axis.

        Examples
        --------
        xdoctest: +REQUIRES(module: umap-learn)
        ```pycon
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
        ...     states = pd.Series(['A','A','B','B','A'] * (len(f.tracking.data)//5 + 1))[:len(f.tracking.data)]
        ...     states.index = f.tracking.data.index
        ...     f.store(states, 'state', meta={})
        ...     f.tracking.add_tag('group', f'G{i+1}')
        >>> sc = SummaryCollection.from_features_collection(fc.groupby('group'))
        >>> with tempfile.TemporaryDirectory() as outdir:
        ...     fig, ax = sc.plot_transition_umap(column='state', all_states=['A','B'], groups=['G1','G2'], show=False, save_dir=outdir)
        ...     os.path.exists(os.path.join(outdir, 'transition_umap.png'))
        True

        ```
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler

        try:
            import umap  # type: ignore
        except Exception as e:
            raise ImportError(
                "UMAP is required for this plot. Please install 'umap-learn'."
            ) from e

        if not getattr(self, "is_grouped", False):
            raise ValueError(
                "UMAP plot requires a grouped SummaryCollection (call groupby first)."
            )

        # Compute transition matrices per subject per group
        matrices_result = self.transition_matrix(column, all_states)
        matrices = {
            group: {k: v.value for k, v in d.items()}
            for group, d in matrices_result.items()
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
            color_map = {
                g: base_colors[i % len(base_colors)]
                for i, g in enumerate(unique_groups)
            }
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
