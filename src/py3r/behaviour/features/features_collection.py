from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from py3r.behaviour.features.cluster_pipeline import (
    ClusteringConfig,
    ClusteringPipeline,
)
from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection_batch_mixin import (
    FeaturesCollectionBatchMixin,
)
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.util.base_collection import BaseCollection
from py3r.behaviour.util.collection_utils import BatchResult, _Indexer
from py3r.behaviour.util.dev_utils import dev_mode
from py3r.behaviour.util.series_utils import (
    apply_normalization_to_df,
    normalize_df,
)


class FeaturesCollection(BaseCollection, FeaturesCollectionBatchMixin):
    """
    Collection of Features objects, keyed by name.
    note: type-hints refer to Features, but factory methods allow for other classes
    these are intended ONLY for subclasses of Features, and this is enforced

    Examples
    --------
    ```pycon
    >>> import tempfile, shutil
    >>> from pathlib import Path
    >>> from py3r.behaviour.util.docdata import data_path
    >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
    >>> with tempfile.TemporaryDirectory() as d:
    ...     d = Path(d)
    ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
    ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
    ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
    >>> fc = FeaturesCollection.from_tracking_collection(tc)
    >>> list(sorted(fc.keys()))
    ['A', 'B']

    ```
    """

    _element_type = Features

    def __init__(self, features_dict: dict[str, Features]):
        super().__init__(features_dict)

    @property
    def features_dict(self):
        return self._obj_dict

    @classmethod
    def from_tracking_collection(
        cls, tracking_collection: TrackingCollection, feature_cls=Features
    ):
        """
        Create a FeaturesCollection from a TrackingCollection.

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        >>> fc = FeaturesCollection.from_tracking_collection(tc)
        >>> isinstance(fc['A'], Features) and isinstance(fc['B'], Features)
        True

        ```
        """
        if not issubclass(feature_cls, Features):
            raise TypeError(f"feature_cls must be Features or a subclass, got {feature_cls}")
        # If grouped, build a grouped FeaturesCollection preserving grouping
        if getattr(tracking_collection, "is_grouped", False):
            grouped_dict = {}
            for gkey, sub_tc in tracking_collection.items():
                # Validate mapping within subgroup
                for handle, t in sub_tc._obj_dict.items():
                    if handle != t.handle:
                        raise ValueError(
                            f"Key '{handle}' does not match object's handle '{t.handle}'"
                        )
                grouped_dict[gkey] = cls(
                    {handle: feature_cls(t) for handle, t in sub_tc._obj_dict.items()}
                )
            grouped_fc = cls(grouped_dict)
            grouped_fc._is_grouped = True
            grouped_fc._groupby_tags = getattr(tracking_collection, "groupby_tags", None)
            return grouped_fc
        # Flat case
        for handle, t in tracking_collection._obj_dict.items():
            if handle != t.handle:
                raise ValueError(f"Key '{handle}' does not match object's handle '{t.handle}'")
        return cls({handle: feature_cls(t) for handle, t in tracking_collection._obj_dict.items()})

    def within_boundary_static(
        self,
        point: str,
        boundary,
        boundary_name: str = None,
    ):
        """
        Collection-aware wrapper that supports:
          - a single static `boundary` (list[(x,y)]) applied to all items, or
          - a per-handle mapping of boundaries produced by batch `define_boundary`:
            - flat: {handle: list[(x,y)]}
            - grouped: {group_key: {handle: list[(x,y)]}}
            - BatchResult in either of the above shapes

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        >>> fc = FeaturesCollection.from_tracking_collection(tc)
        >>> boundaries = fc.define_boundary(['p1','p2','p3'], scaling=1.0)
        >>> res = fc.within_boundary_static('p1', boundaries)
        >>> isinstance(res, dict)
        True
        >>> any(isinstance(v, pd.Series) for v in res.values())
        True

        >>> # Grouped case: add tags on Tracking, group, then build grouped FeaturesCollection
        >>> # (boundaries BatchResult structure matches grouped layout)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        ...     tc['A'].add_tag('group', 'G1'); tc['B'].add_tag('group', 'G2')
        ...     gtc = tc.groupby('group')
        ...     gfc = FeaturesCollection.from_tracking_collection(gtc)
        ...     g_boundaries = gfc.define_boundary(['p1','p2','p3'], scaling=1.0)
        ...     g_res = gfc.within_boundary_static('p1', g_boundaries)
        >>> isinstance(g_res, dict)
        True
        >>> any(any(isinstance(s, pd.Series) for s in sub.values()) for sub in g_res.values())
        True

        ```
        """
        # Case 1: one boundary applied to all leaves -> use standard batch path
        if isinstance(boundary, list):
            return self._invoke_batch("within_boundary_static", point, boundary, boundary_name)

        # Case 2: mapping or BatchResult providing per-handle boundaries
        return self._invoke_batch_mapped(
            "within_boundary_static",
            args=(point,),
            kwargs={"boundary": boundary, "boundary_name": boundary_name},
        )

    def distance_to_boundary_static(
        self,
        point: str,
        boundary,
        boundary_name: str = None,
    ):
        """
        Collection-aware wrapper that supports:
          - a single static `boundary` (list[(x,y)]) applied to all items, or
          - a per-handle mapping of boundaries produced by batch `define_boundary`:
            - flat: {handle: list[(x,y)]}
            - grouped: {group_key: {handle: list[(x,y)]}}
            - BatchResult in either of the above shapes

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        >>> fc = FeaturesCollection.from_tracking_collection(tc)
        >>> boundaries = fc.define_boundary(['p1','p2','p3'], scaling=1.0)
        >>> res = fc.distance_to_boundary_static('p1', boundaries)
        >>> isinstance(res, dict)
        True
        >>> any(isinstance(v, pd.Series) for v in res.values())
        True

        >>> # Grouped case
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        ...     tc['A'].add_tag('group', 'G1'); tc['B'].add_tag('group', 'G2')
        ...     gtc = tc.groupby('group')
        ...     gfc = FeaturesCollection.from_tracking_collection(gtc)
        ...     g_boundaries = gfc.define_boundary(['p1','p2','p3'], scaling=1.0)
        ...     g_res = gfc.distance_to_boundary_static('p1', g_boundaries)
        >>> isinstance(g_res, dict)
        True
        >>> any(any(isinstance(s, pd.Series) for s in sub.values()) for sub in g_res.values())
        True

        ```
        """
        if isinstance(boundary, list):
            return self._invoke_batch("distance_to_boundary_static", point, boundary, boundary_name)

        return self._invoke_batch_mapped(
            "distance_to_boundary_static",
            args=(point,),
            kwargs={"boundary": boundary, "boundary_name": boundary_name},
        )

    @classmethod
    def from_list(cls, features_list: list[Features]):
        """
        Create a FeaturesCollection from a list of Features objects, keyed by handle

        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t1 = Tracking.from_dlc(str(p), handle='A', fps=30)
        ...     t2 = Tracking.from_dlc(str(p), handle='B', fps=30)
        >>> f1, f2 = Features(t1), Features(t2)
        >>> fc = FeaturesCollection.from_list([f1, f2])
        >>> list(sorted(fc.keys()))
        ['A', 'B']

        ```
        """
        handles = [obj.handle for obj in features_list]
        if len(handles) != len(set(handles)):
            raise Exception("handles must be unique")
        features_dict = {obj.handle: obj for obj in features_list}
        return cls(features_dict)

    def cluster_embedding(
        self,
        embedding_dict: dict[str, list[int]],
        n_clusters: int,
        random_state: int = 0,
        *,
        auto_normalize: bool = False,
        rescale_factors: dict | None = None,
        lowmem: bool = False,
        decimation_factor: int = 10,
        custom_scaling: dict[str, dict] | None = None,
        missing_policy: Literal["drop", "impute_weight"] = "drop",
    ):
        """
        Perform k-means clustering using the specified embedding.

        Unified behaviour for flat and grouped collections.
        Returns a BatchResult mapping:
          - grouped: {group_key: {feature_handle: FeaturesResult}}
          - flat:    {feature_handle: FeaturesResult}
        along with (centroids, normalization_factors or None).

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        >>> fc = FeaturesCollection.from_tracking_collection(tc)
        >>> # Create a trivial feature 'counter' in each Features to embed
        >>> for f in fc.values():
        ...     s = pd.Series(range(len(f.tracking.data)), index=f.tracking.data.index)
        ...     f.store(s, 'counter')
        >>> batch, centroids, norm = fc.cluster_embedding(
        ...     {'counter':[0]}, n_clusters=2, lowmem=True)
        >>> isinstance(centroids, pd.DataFrame)
        True
        >>> batch, centroids, norm = fc.cluster_embedding(
        ...     {'counter':[0]}, n_clusters=2, lowmem=True,
        ...     missing_policy='impute_weight')
        >>> isinstance(centroids, pd.DataFrame)
        True
        >>> batch, centroids, norm = fc.cluster_embedding(
        ...     {'counter':[0]}, n_clusters=2, lowmem=True,
        ...     missing_policy='drop')
        >>> isinstance(centroids, pd.DataFrame)
        True

        ```
        """

        # Delegate to the pluggable pipeline
        pipeline = ClusteringPipeline()
        cfg = ClusteringConfig(
            n_clusters=n_clusters,
            random_state=random_state,
            auto_normalize=auto_normalize,
            rescale_factors=rescale_factors,
            lowmem=lowmem,
            decimation_factor=decimation_factor,
            custom_scaling=custom_scaling,
            missing_policy=missing_policy,
        )
        result_dict, centroids, normalization_factors, _meta = pipeline.run(
            self, embedding_dict, cfg
        )
        return BatchResult(result_dict, self), centroids, normalization_factors

    def cluster_diagnostics(
        self,
        labels_result,
        n_clusters: int | None = None,
        *,
        low: float = 0.05,
        high: float = 0.90,
        verbose: bool = True,
    ):
        """
        Compute diagnostic stats for cluster label assignments.

        Parameters
        ----------
        labels_result:
            Mapping from handle (or group->handle) to FeaturesResult of integer labels (with NA).
            Accepts the return shape of `cluster_embedding(...)[0]` (BatchResult or dict).
        n_clusters:
            Optional number of clusters. If None, inferred from labels (max label + 1).
        low, high:
            Prevalence thresholds for low/high cluster labels per recording.
        verbose:
            If True, print a compact summary.

        Returns
        -------
        dict with:
            - 'global': {'cluster_prevalence': {label: frac, ...}, 'percent_nan': frac}
            - 'per_recording': DataFrame, rows per recording, cols
              ['percent_nan', 'num_missing', 'num_low', 'num_high']
            - 'summary': min/median/max for the per_recording columns
            - if grouped: 'per_group': {group_key: {'per_recording': df, 'summary': {...}}}
        """
        import pandas as pd

        # Unwrap BatchResult/dicts into a canonical mapping
        def is_grouped_mapping(obj) -> bool:
            if isinstance(obj, dict):
                if len(obj) == 0:
                    return False
                first_val = next(iter(obj.values()))
                return isinstance(first_val, dict)
            # BatchResult acts like mapping
            try:
                vals = list(obj.values())
                if not vals:
                    return False
                return isinstance(vals[0], dict)
            except Exception:
                return False

        def to_plain_dict(obj) -> dict:
            # converts BatchResult or dict to regular dict
            if isinstance(obj, dict):
                return obj
            return dict(obj)

        labels_map = to_plain_dict(labels_result)
        grouped = is_grouped_mapping(labels_map)

        # Helper to flatten to per-recording series
        def iter_series_flat(mapping):
            if grouped:
                for gkey, sub in mapping.items():
                    for handle, fr in to_plain_dict(sub).items():
                        yield (gkey, handle), pd.Series(fr)
            else:
                for handle, fr in mapping.items():
                    yield handle, pd.Series(fr)

        # Infer cluster set if needed
        if n_clusters is None:
            uniq = set()
            for _, s in iter_series_flat(labels_map):
                uniq |= set(s.dropna().astype(int).unique().tolist())
            n_clusters = (max(uniq) + 1) if uniq else 0
        cluster_labels = list(range(n_clusters))

        # Global counts
        total_frames = 0
        total_nan = 0
        cluster_counts = {c: 0 for c in cluster_labels}

        # Per-recording stats
        rows = []
        if grouped:
            per_group_rows = {}

        for key, s in iter_series_flat(labels_map):
            ser = s
            count_total = int(ser.shape[0])
            count_nan = int(ser.isna().sum())
            total_frames += count_total
            total_nan += count_nan
            counts = ser.value_counts(dropna=True).to_dict()
            for c in cluster_labels:
                cluster_counts[c] += int(counts.get(c, 0))
            # per recording prevalence using total frames (incl NaN)
            missing = sum(1 for c in cluster_labels if counts.get(c, 0) == 0)
            low_cnt = sum(
                1 for c in cluster_labels if (counts.get(c, 0) / max(1, count_total)) < low
            )
            high_cnt = sum(
                1 for c in cluster_labels if (counts.get(c, 0) / max(1, count_total)) > high
            )
            rec = {
                "id": key,
                "percent_nan": count_nan / max(1, count_total),
                "num_missing": missing,
                "num_low": low_cnt,
                "num_high": high_cnt,
            }
            if grouped:
                gkey, handle = key
                rec["group"] = gkey
                rec["handle"] = handle
                per_group_rows.setdefault(gkey, []).append(
                    {
                        "handle": handle,
                        **{
                            k: rec[k]
                            for k in [
                                "percent_nan",
                                "num_missing",
                                "num_low",
                                "num_high",
                            ]
                        },
                    }
                )
            rows.append(rec)

        # Build outputs
        global_out = {
            "cluster_prevalence": {
                c: cluster_counts[c] / max(1, total_frames) for c in cluster_labels
            },
            "percent_nan": total_nan / max(1, total_frames),
        }
        per_df = pd.DataFrame(rows).set_index("id")
        summary = (
            per_df[["percent_nan", "num_missing", "num_low", "num_high"]]
            .agg(["min", "median", "max"])
            .to_dict()
        )

        out = {
            "global": global_out,
            "per_recording": per_df,
            "summary": summary,
        }
        if grouped:
            per_group = {}
            for gkey, glist in per_group_rows.items():
                gdf = pd.DataFrame(glist).set_index("handle")
                gsummary = (
                    gdf[["percent_nan", "num_missing", "num_low", "num_high"]]
                    .agg(["min", "median", "max"])
                    .to_dict()
                )
                per_group[gkey] = {"per_recording": gdf, "summary": gsummary}
            out["per_group"] = per_group

        if verbose:
            # Compact printout
            import pprint

            pp = pprint.PrettyPrinter(indent=2, width=100)
            print("Cluster diagnostics:")
            print("- Global:")
            pp.pprint(global_out)
            print("- Summary (min/median/max across recordings):")
            pp.pprint(summary)

        return out

    # ---- Cross-prediction utilities migrated from MultipleFeaturesCollection ----
    @dev_mode
    @staticmethod
    def _train_and_predict_rms(
        *,
        train_handles,
        test_handles,
        get_source_array,
        get_target_array,
        get_source_index,
        get_target_index,
        get_source_columns,
        get_target_columns,
        predictor_cls,
        predictor_kwargs,
        normalize_source,
        normalize_pred,
    ):
        """
        Helper to train a predictor and compute RMS error for each test handle.
        Returns a list of RMS Series (one per test handle, in order).
        """
        import numpy as np
        import pandas as pd

        if predictor_kwargs is None:
            predictor_kwargs = {}

        # Prepare embeddings (arrays)
        train_X = [get_source_array(h) for h in train_handles]
        train_y = [get_target_array(h) for h in train_handles]
        test_X = [get_source_array(h) for h in test_handles]
        test_y = [get_target_array(h) for h in test_handles]

        # Normalize if needed
        if normalize_source:
            train_X_concat = pd.DataFrame(np.vstack(train_X))
            train_X_concat, rescale_factors = normalize_df(train_X_concat)
            lengths = [len(x) for x in train_X]
            starts = np.cumsum([0] + lengths[:-1])
            train_X = [
                train_X_concat.iloc[start : start + length].values
                for start, length in zip(starts, lengths, strict=True)
            ]
            test_X = [
                apply_normalization_to_df(pd.DataFrame(x), rescale_factors).values for x in test_X
            ]
        else:
            rescale_factors = None

        # Train predictor on concatenated arrays (convert to DataFrame)
        train_X_all = np.vstack(train_X)
        train_y_all = np.vstack(train_y)
        x_cols = get_source_columns(train_handles[0])
        y_cols = get_target_columns(train_handles[0])
        train_X_df = pd.DataFrame(train_X_all, columns=x_cols)
        train_y_df = pd.DataFrame(train_y_all, columns=y_cols)
        predictor = predictor_cls(**predictor_kwargs)
        predictor.fit(train_X_df, train_y_df)

        # Predict for each test handle and compute RMS
        rms_list = []
        for x, y, h in zip(test_X, test_y, test_handles, strict=True):
            x_df = pd.DataFrame(x, columns=get_source_columns(h))
            y_df = pd.DataFrame(y, columns=get_target_columns(h))
            preds = predictor.predict(x_df)
            preds = preds.reindex(index=y_df.index, columns=y_df.columns)
            rms = Features.rms_error_between_embeddings(y_df, preds, rescale=normalize_pred)
            rms_list.append(rms)
        return rms_list

    @dev_mode
    def cross_predict_rms(
        self,
        source_embedding: dict[str, list[int]],
        target_embedding: dict[str, list[int]],
        normalize_source: bool = False,
        normalize_pred: dict | str = None,
        set1: list | None = None,
        set2: list | None = None,
        predictor_cls=None,
        predictor_kwargs=None,
    ):
        """
        Dev mode only: not available in public release yet.
        """
        if predictor_cls is None:
            from py3r.behaviour.predictors import KNNPredictor

            predictor_cls = KNNPredictor
        if predictor_kwargs is None:
            predictor_kwargs = {}

        is_grouped = getattr(self, "is_grouped", False)
        if is_grouped:
            group_keys = list(self._obj_dict.keys())
            all_groups = group_keys
            if set1 is None:
                set1 = all_groups
            if set2 is None:
                set2 = all_groups
            within_groups = sorted(set(set1) | set(set2))
        else:
            within_groups = ["__flat__"]

        # Build embedding cache keyed by (group, handle) to avoid collisions
        embedding_cache = {}
        if is_grouped:
            groups_to_cache = sorted(set((set1 or []) + (set2 or [])))
            for g in groups_to_cache:
                for handle, feat in self._obj_dict[g].features_dict.items():
                    src_df = feat.embedding_df(source_embedding)
                    tgt_df = feat.embedding_df(target_embedding)
                    embedding_cache[(g, handle)] = {
                        "source_array": src_df.values,
                        "source_index": src_df.index,
                        "source_columns": src_df.columns,
                        "target_array": tgt_df.values,
                        "target_index": tgt_df.index,
                        "target_columns": tgt_df.columns,
                    }
        else:
            for handle, feat in self.features_dict.items():
                src_df = feat.embedding_df(source_embedding)
                tgt_df = feat.embedding_df(target_embedding)
                embedding_cache[("__flat__", handle)] = {
                    "source_array": src_df.values,
                    "source_index": src_df.index,
                    "source_columns": src_df.columns,
                    "target_array": tgt_df.values,
                    "target_index": tgt_df.index,
                    "target_columns": tgt_df.columns,
                }

        def get_source_array(key):
            return embedding_cache[key]["source_array"]

        def get_target_array(key):
            return embedding_cache[key]["target_array"]

        def get_source_index(key):
            return embedding_cache[key]["source_index"]

        def get_target_index(key):
            return embedding_cache[key]["target_index"]

        def get_source_columns(key):
            return embedding_cache[key]["source_columns"]

        def get_target_columns(key):
            return embedding_cache[key]["target_columns"]

        results = {"within": {}, "between": {}}

        # Within-group leave-one-out (or flat collection)
        for g in within_groups:
            if is_grouped:
                handles = list(self._obj_dict[g].features_dict.keys())
                group_key = g
            else:
                handles = list(self.features_dict.keys())
                group_key = "__flat__"
            rms_dict = {}
            for left_out in handles:
                train_handles = [(group_key, h) for h in handles if h != left_out]
                test_handles = [(group_key, left_out)]
                rms_list = self._train_and_predict_rms(
                    train_handles=train_handles,
                    test_handles=test_handles,
                    get_source_array=get_source_array,
                    get_target_array=get_target_array,
                    get_source_index=get_source_index,
                    get_target_index=get_target_index,
                    get_source_columns=get_source_columns,
                    get_target_columns=get_target_columns,
                    predictor_cls=predictor_cls,
                    predictor_kwargs=predictor_kwargs,
                    normalize_source=normalize_source,
                    normalize_pred=normalize_pred,
                )
                rms_dict[left_out] = rms_list[0]
            results["within"][group_key] = rms_dict

        # Between-group (only if truly grouped)
        if is_grouped:
            for g1 in set1:
                for g2 in set2:
                    if g1 == g2:
                        continue
                    source_handles = [(g1, h) for h in self._obj_dict[g1].features_dict.keys()]
                    target_handles = [(g2, h) for h in self._obj_dict[g2].features_dict.keys()]
                    rms_list = self._train_and_predict_rms(
                        train_handles=source_handles,
                        test_handles=target_handles,
                        get_source_array=get_source_array,
                        get_target_array=get_target_array,
                        get_source_index=get_source_index,
                        get_target_index=get_target_index,
                        get_source_columns=get_source_columns,
                        get_target_columns=get_target_columns,
                        predictor_cls=predictor_cls,
                        predictor_kwargs=predictor_kwargs,
                        normalize_source=normalize_source,
                        normalize_pred=normalize_pred,
                    )
                    rms_dict = {
                        name: rms
                        for name, rms in zip(
                            self._obj_dict[g2].features_dict.keys(),
                            rms_list,
                            strict=True,
                        )
                    }
                    key = f"from{g1}_to_{g2}"
                    results["between"][key] = rms_dict

        return results

    @dev_mode
    @staticmethod
    def plot_cross_predict_vs_within(results, from_group, to_group, show=True):
        """
        Dev mode only: not available in public release yet.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Keys
        between_key = f"from{from_group}_to_{to_group}"
        within_key = to_group
        between_dict = results["between"].get(between_key, {})
        within_dict = results["within"].get(within_key, {})
        handles = sorted(set(between_dict.keys()) & set(within_dict.keys()))
        if not handles:
            raise ValueError(f"No overlapping handles between {between_key} and {within_key}")
        between_means = [between_dict[h].mean(skipna=True) for h in handles]
        within_means = [within_dict[h].mean(skipna=True) for h in handles]
        diff_means = [b - w for b, w in zip(between_means, within_means, strict=True)]
        x = np.arange(len(handles))
        width = 0.3
        fig, ax = plt.subplots(figsize=(max(8, len(handles) * 0.7), 5))
        ax.bar(x + width, diff_means, width, label="between - within")
        ax.set_xticks(x)
        ax.set_xticklabels(handles, rotation=90)
        ax.set_ylabel("Mean RMS difference")
        ax.set_title(f"Cross-predict vs Within: {from_group} → {to_group}")
        from scipy.stats import ttest_rel

        t_stat, p_value = ttest_rel(between_means, within_means, nan_policy="omit")
        ax.text(
            0.99,
            0.99,
            f"Paired t-test: p = {p_value:.3g}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=12,
            color="red",
        )
        plt.tight_layout()
        if show:
            plt.show()
        return {
            "handles": handles,
            "between_means": between_means,
            "within_means": within_means,
            "diff_means": diff_means,
            "t_stat": t_stat,
            "p_value": p_value,
        }

    @dev_mode
    @staticmethod
    def plot_cross_predict_results(
        results,
        within_keys=None,
        between_keys=None,
        plot_type="bar",
        figsize=(10, 6),
        show=True,
    ):
        """
        Dev mode only: not available in public release yet.
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        records = []
        if within_keys is not None:
            for coll in within_keys:
                for feat, series in results["within"].get(coll, {}).items():
                    arr = series.dropna().values
                    for v in arr:
                        records.append({"Category": f"within_{coll}", "Feature": feat, "RMS": v})
        if between_keys is not None:
            for comp in between_keys:
                for feat, series in results["between"].get(comp, {}).items():
                    arr = series.dropna().values
                    for v in arr:
                        records.append({"Category": comp, "Feature": feat, "RMS": v})
        df = pd.DataFrame(records)
        plt.figure(figsize=figsize)
        if plot_type == "bar":
            means = df.groupby("Category").RMS.mean()
            means.plot(kind="bar", yerr=df.groupby("Category").RMS.std(), capsize=4)
            plt.ylabel("Mean RMS (mean of means)")
            plt.title("RMS prediction error by category")
        elif plot_type == "point":
            means = df.groupby(["Category", "Feature"]).RMS.mean().reset_index()
            pivot = means.pivot(index="Feature", columns="Category", values="RMS")
            within_col = [c for c in pivot.columns if c.startswith("within_")]
            between_col = [c for c in pivot.columns if not c.startswith("within_")]
            if len(within_col) == 1 and len(between_col) == 1:
                pivot["mean_diff"] = pivot[between_col[0]] - pivot[within_col[0]]
            else:
                pivot["mean_diff"] = np.nan
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(figsize[0], figsize[1] * 1.5),
                sharex=True,
                gridspec_kw={"height_ratios": [2, 1]},
            )
            sns.pointplot(data=means, x="Feature", y="RMS", hue="Category", dodge=True, ax=ax1)
            ax1.set_ylabel("mean RMS error")
            ax1.set_title("Cross-predict summary")
            ax1.tick_params(axis="x", rotation=90)
            ax2.bar(pivot.index, pivot["mean_diff"])
            ax2.axhline(0, color="gray", linestyle="--")
            ax2.set_ylabel("Mean (Between - Within)")
            ax2.set_title("Mean RMS Difference per Video")
            ax2.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            if show:
                plt.show()
            return df
        elif plot_type == "violin":
            sns.violinplot(data=df, x="Category", y="RMS", inner="point")
            plt.ylabel("RMS")
            plt.title("RMS prediction error by category")
        else:
            raise ValueError("plot_type must be 'bar', 'point', or 'violin'")
        plt.tight_layout()
        if show:
            plt.show()
        return df

    @dev_mode
    @staticmethod
    def dumbbell_plot_cross_predict(results, within_key, between_key, figsize=(3, 3), show=True):
        """
        Dev mode only: not available in public release yet.
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        features = sorted(
            set(
                list(results["within"].get(within_key, {}).keys())
                + list(results["between"].get(between_key, {}).keys())
            )
        )
        data = []
        for feat in features:
            mean_within = (
                results["within"].get(within_key, {}).get(feat, pd.Series(dtype=float)).mean()
            )
            mean_between = (
                results["between"].get(between_key, {}).get(feat, pd.Series(dtype=float)).mean()
            )
            data.append({"Feature": feat, "Within": mean_within, "Between": mean_between})
        df = pd.DataFrame(data)
        x = [0, 1]
        plt.figure(figsize=figsize)
        for _, row in df.iterrows():
            plt.plot(x, [row["Within"], row["Between"]], color="gray", lw=2, zorder=1)
            plt.scatter(x, [row["Within"], row["Between"]], s=60, color="black", zorder=2)
        plt.xticks(x, ["Within", "Between"])
        plt.ylabel("Mean RMS")
        plt.title(f"Dumbbell Plot: {within_key} vs {between_key}")
        plt.tight_layout()
        if show:
            plt.show()
        return df

    @dev_mode
    def train_knn_regressor(
        self,
        *,
        source_embedding: dict[str, list[int]],
        target_embedding: dict[str, list[int]],
        predictor_cls=None,
        predictor_kwargs=None,
        normalize_source: bool = False,
        **kwargs,
    ):
        """
        Dev mode only: not available in public release yet.
        """
        if predictor_cls is None:
            from py3r.behaviour.predictors import KNNPredictor

            predictor_cls = KNNPredictor
        if predictor_kwargs is None:
            predictor_kwargs = {}
        train_embed = self.embedding_df(source_embedding)
        target_embed = self.embedding_df(target_embedding)
        rescale_factors = None
        if normalize_source:
            train_embed, rescale_factors = normalize_df(train_embed)
        predictor = predictor_cls(**predictor_kwargs)
        predictor.fit(train_embed, target_embed)
        if normalize_source:
            return predictor, train_embed.columns, target_embed.columns, rescale_factors
        else:
            return predictor, train_embed.columns, target_embed.columns

    @dev_mode
    def predict_knn(
        self,
        model,
        source_embedding: dict[str, list[int]],
        target_embedding: dict[str, list[int]],
        rescale_factors: dict = None,
    ) -> pd.DataFrame:
        """
        Dev mode only: not available in public release yet.
        """
        test_embed = self.embedding_df(source_embedding)
        if rescale_factors is not None:
            test_embed = apply_normalization_to_df(test_embed, rescale_factors)
        target_embed = self.embedding_df(target_embedding)
        preds = model.predict(test_embed)
        # Ensure the output DataFrame has the same index and columns as target_embed
        preds = pd.DataFrame(preds, index=target_embed.index, columns=target_embed.columns)
        return preds

    def plot(self, arg=None, figsize=(8, 2), show: bool = True, title: str = None):
        """
        Plot features for all collections in the MultipleFeaturesCollection.
        - If arg is a BatchResult or dict: treat as batch result and plot for each collection.
        - Otherwise: treat as column name(s) or None and plot for each collection.
        - If title is provided, it will be used as the overall title for the figure.
        """
        import matplotlib.pyplot as plt

        # If grouped, delegate plotting per group and return a dict of (fig, axes)
        if getattr(self, "is_grouped", False):
            figs_axes = {}
            for gkey, sub in self.items():
                figs_axes[gkey] = sub.plot(arg, figsize=figsize, show=show, title=str(gkey))
            return figs_axes

        if arg is None:
            # Plot all columns for each Features object
            features_dict = {handle: obj.data for handle, obj in self.features_dict.items()}
            plot_type = "all"
        elif isinstance(arg, (str, list)):
            # Plot specified column(s) for each Features object
            if isinstance(arg, str):
                columns = [arg]
            else:
                columns = arg
            features_dict = {}
            for handle, obj in self.features_dict.items():
                # Only include columns that exist in this Features object
                cols = [col for col in columns if col in obj.data]
                if cols:
                    features_dict[handle] = obj.data[cols]
            plot_type = "columns"
        elif isinstance(arg, dict):
            # Batch result: plot each FeaturesResult
            features_dict = arg
            plot_type = "batch"
        else:
            raise TypeError(
                "Argument must be None, a string, a list of strings, or a batch result dict."
            )

        n = len(features_dict)
        if n == 0:
            raise ValueError("No features to plot.")
        fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, (handle, data) in zip(axes, features_dict.items(), strict=True):
            if plot_type == "batch":
                # FeaturesResult: plot as a single series
                ax.plot(data.index, data.values, label=getattr(data, "name", "value"))
            else:
                # DataFrame: plot all columns or selected columns
                if isinstance(data, pd.Series):
                    ax.plot(data.index, data.values, label=data.name)
                else:
                    data.plot(ax=ax)
            ax.set_title(str(handle))
            ax.set_xlabel("frame")
            ax.legend()
        if title is not None:
            fig.suptitle(title, fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
        else:
            plt.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def hist_time_in_state(
        self,
        column: str,
        *,
        level: str = "individual",
        max_unique: int = 50,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        figsize: tuple[float, float] = (8, 5),
        jitter: float = 0.1,
    ):
        """
        Plot histograms showing time spent in each state for a boolean or categorical column.

        For boolean columns: shows time where value is True.
        For categorical columns: shows time per state value.

        Parameters
        ----------
        column : str
            Column name from Features.data to plot.
        level : str
            Aggregation level:
            - "individual": one figure per Features object (recording)
            - "group": dots per recording within each group, mean line, groups side-by-side
            - "all": dots per recording, mean line across all
        max_unique : int
            Maximum number of unique values allowed for categorical columns. Default 50.
        show : bool
            If True, display the plot(s). Default True.
        savedir : str | None
            Directory to save figure(s). Figures are auto-named based on level:
            - individual: "{handle}_{column}_time_in_state.png"
            - group: "{column}_time_in_state_by_group.png" (or custom filename)
            - all: "{column}_time_in_state_all.png" (or custom filename)
        filename : str | None
            Custom filename for saved figure (used with level="group" or "all").
            For level="individual", filenames are always auto-generated per handle.
        figsize : tuple
            Figure size (width, height) in inches.
        jitter : float
            Amount of horizontal jitter for dots (0 = no jitter). Default 0.1.

        Returns
        -------
        For level="individual":
            dict[str, tuple[Figure, Axes, Series]] - mapping handle to (fig, ax, data)
        For level="group" or level="all":
            tuple[Figure, Axes, DataFrame] - single figure with per-recording data

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        >>> fc = FeaturesCollection.from_tracking_collection(tc)
        >>> # Create a boolean feature in each
        >>> for f in fc.values():
        ...     f.store(pd.Series([True, False] * 15, index=f.tracking.data.index[:30]),
        ...             'is_active', meta={})
        >>> results = fc.hist_time_in_state('is_active', level='individual', show=False)
        >>> len(results) == 2
        True
        >>> fig, ax, data = fc.hist_time_in_state('is_active', level='all', show=False)
        >>> isinstance(data, pd.DataFrame)
        True

        ```
        """
        import os

        import matplotlib.pyplot as plt

        from py3r.behaviour.features.features import _compute_time_in_state_single

        valid_levels = {"individual", "group", "all"}
        if level not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}, got '{level}'")

        flat_self = self.flatten()

        if level == "individual":
            # One figure per Features object
            results = {}
            for handle, feat in flat_self.items():
                fig, ax, data = feat.hist_time_in_state(
                    column,
                    max_unique=max_unique,
                    show=False,
                    savedir=savedir,
                    filename=f"{handle}_{column}_time_in_state.png",
                    figsize=figsize,
                    title=f"{handle}: {column}",
                )
                if show:
                    plt.show()
                results[handle] = (fig, ax, data)
            return results

        elif level == "group":
            if not self.is_grouped:
                raise ValueError(
                    "level='group' requires a grouped collection. "
                    "Use collection.groupby(tags) first."
                )

            # Collect per-recording data for each group
            # Structure: {state: {group: [values per recording]}}
            all_states = set()
            group_keys = []
            recordings_data = {}  # {group_str: {handle: Series}}

            for gkey, subcoll in self.items():
                gkey_str = str(gkey[0]) if len(gkey) == 1 else str(gkey)
                group_keys.append(gkey_str)
                recordings_data[gkey_str] = {}

                for handle, feat in subcoll.items():
                    if column not in feat.data.columns:
                        continue
                    fps = feat.tracking.meta.get("fps", 1.0)
                    feat_data = _compute_time_in_state_single(
                        feat.data[column], fps, max_unique=max_unique
                    )
                    recordings_data[gkey_str][handle] = feat_data
                    all_states.update(feat_data.index.tolist())

            all_states = sorted(all_states, key=str)
            n_states = len(all_states)
            n_groups = len(group_keys)

            if n_states == 0:
                raise ValueError(f"No data found for column '{column}'")

            # Build DataFrame: rows = recordings, columns = states, with group info
            rows = []
            for gkey_str in group_keys:
                for handle, feat_data in recordings_data[gkey_str].items():
                    row = {"_group": gkey_str, "_handle": handle}
                    for state in all_states:
                        row[state] = feat_data.get(state, 0.0)
                    rows.append(row)
            df = pd.DataFrame(rows)

            # Plot: dots + mean for each group, side-by-side
            fig, ax = plt.subplots(figsize=figsize)
            width = 0.8 / n_groups
            colors = plt.cm.tab10.colors

            for i, gkey_str in enumerate(group_keys):
                group_df = df[df["_group"] == gkey_str]
                color = colors[i % len(colors)]

                for j, state in enumerate(all_states):
                    x_base = j + (i - n_groups / 2 + 0.5) * width
                    values = group_df[state].values

                    # Dots with jitter
                    if jitter > 0:
                        x_jittered = x_base + np.random.uniform(
                            -jitter * width, jitter * width, len(values)
                        )
                    else:
                        x_jittered = np.full(len(values), x_base)

                    ax.scatter(x_jittered, values, color=color, alpha=0.7, s=30, zorder=3)

                    # Mean line
                    mean_val = np.mean(values) if len(values) > 0 else 0
                    ax.hlines(
                        mean_val,
                        x_base - width * 0.4,
                        x_base + width * 0.4,
                        color=color,
                        linewidth=2,
                        zorder=4,
                        label=gkey_str if j == 0 else None,
                    )

            ax.set_xticks(range(n_states))
            ax.set_xticklabels(all_states, rotation=45, ha="right")
            ax.set_ylabel("Time (seconds)")
            ax.set_title(f"{column}: Time in State by Group")
            ax.legend(title="Group")
            plt.tight_layout()

            if savedir:
                os.makedirs(savedir, exist_ok=True)
                fname = filename or f"{column}_time_in_state_by_group.png"
                fig.savefig(os.path.join(savedir, fname), dpi=150, bbox_inches="tight")
            if show:
                plt.show()

            return fig, ax, df

        else:  # level == "all"
            # Collect per-recording data
            all_states = set()
            recordings_data = {}  # {handle: Series}

            for handle, feat in flat_self.items():
                if column not in feat.data.columns:
                    continue
                fps = feat.tracking.meta.get("fps", 1.0)
                feat_data = _compute_time_in_state_single(
                    feat.data[column], fps, max_unique=max_unique
                )
                recordings_data[handle] = feat_data
                all_states.update(feat_data.index.tolist())

            all_states = sorted(all_states, key=str)
            n_states = len(all_states)

            if n_states == 0:
                raise ValueError(f"No data found for column '{column}'")

            # Build DataFrame: rows = recordings, columns = states
            rows = []
            for handle, feat_data in recordings_data.items():
                row = {"_handle": handle}
                for state in all_states:
                    row[state] = feat_data.get(state, 0.0)
                rows.append(row)
            df = pd.DataFrame(rows)

            # Plot: dots + mean
            fig, ax = plt.subplots(figsize=figsize)

            for j, state in enumerate(all_states):
                values = df[state].values

                # Dots with jitter
                if jitter > 0:
                    x_jittered = j + np.random.uniform(-jitter, jitter, len(values))
                else:
                    x_jittered = np.full(len(values), j)

                ax.scatter(x_jittered, values, color="steelblue", alpha=0.7, s=30, zorder=3)

                # Mean line
                mean_val = np.mean(values) if len(values) > 0 else 0
                ax.hlines(mean_val, j - 0.3, j + 0.3, color="black", linewidth=2, zorder=4)

            ax.set_xticks(range(n_states))
            ax.set_xticklabels(all_states, rotation=45, ha="right")
            ax.set_ylabel("Time (seconds)")
            ax.set_title(f"{column}: Time in State (All Recordings)")
            plt.tight_layout()

            if savedir:
                os.makedirs(savedir, exist_ok=True)
                fname = filename or f"{column}_time_in_state_all.png"
                fig.savefig(os.path.join(savedir, fname), dpi=150, bbox_inches="tight")
            if show:
                plt.show()

            return fig, ax, df

    def store(
        self,
        results_dict,
        name: str = None,
        meta: dict = None,
        overwrite: bool = False,
    ):
        """
        Store FeaturesResult objects returned by batch methods.

        - Flat collection: results_dict is {handle: FeaturesResult}
        - Grouped collection: results_dict is {group_key: {handle: FeaturesResult}}

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        >>> fc = FeaturesCollection.from_tracking_collection(tc)
        >>> # Build a simple FeaturesResult dict from distance_between
        >>> rd = {h: feat.distance_between('p1','p2') for h, feat in fc.items()}
        >>> fc.store(rd, name='d12')
        >>> all('d12' in feat.data.columns for feat in fc.values())
        True

        ```
        """
        if getattr(self, "is_grouped", False):
            for gkey, group_dict in results_dict.items():
                for handle, v in group_dict.items():
                    if hasattr(v, "store"):
                        v.store(name=name, meta=meta, overwrite=overwrite)
                    else:
                        # Accept raw Series-like leaf values as well
                        if isinstance(v, pd.Series):
                            self._obj_dict[gkey].features_dict[handle].store(
                                v, name, overwrite=overwrite, meta=meta or {}
                            )
                        else:
                            raise ValueError(f"{v} is not a FeaturesResult or Series")
            return
        # Flat case
        for handle, v in results_dict.items():
            if hasattr(v, "store"):
                v.store(name=name, meta=meta, overwrite=overwrite)
            else:
                if isinstance(v, pd.Series):
                    self.features_dict[handle].store(v, name, overwrite=overwrite, meta=meta or {})
                else:
                    raise ValueError(f"{v} is not a FeaturesResult or Series")

    @property
    def loc(self):
        return _Indexer(self, self._loc)

    @property
    def iloc(self):
        return _Indexer(self, self._iloc)

    def _loc(self, idx):
        return self.__class__({k: v.loc[idx] for k, v in self.features_dict.items()})

    def _iloc(self, idx):
        return self.__class__({k: v.iloc[idx] for k, v in self.features_dict.items()})
