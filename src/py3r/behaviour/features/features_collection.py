from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from py3r.behaviour.features.features import Features, FeaturesResult
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.util.base_collection import BaseCollection
from py3r.behaviour.util.collection_utils import _Indexer
from py3r.behaviour.util.dev_utils import dev_mode
from py3r.behaviour.util.series_utils import (
    normalize_df,
    apply_normalization_to_df,
    apply_custom_scaling,
)
from py3r.behaviour.util.collection_utils import BatchResult
from py3r.behaviour.features.features_collection_batch_mixin import (
    FeaturesCollectionBatchMixin,
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
            raise TypeError(
                f"feature_cls must be Features or a subclass, got {feature_cls}"
            )
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
            grouped_fc._groupby_tags = getattr(
                tracking_collection, "groupby_tags", None
            )
            return grouped_fc
        # Flat case
        for handle, t in tracking_collection._obj_dict.items():
            if handle != t.handle:
                raise ValueError(
                    f"Key '{handle}' does not match object's handle '{t.handle}'"
                )
        return cls(
            {
                handle: feature_cls(t)
                for handle, t in tracking_collection._obj_dict.items()
            }
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
        >>> batch, centroids, norm = fc.cluster_embedding({'counter':[0]}, n_clusters=2, lowmem=True)
        >>> isinstance(centroids, pd.DataFrame)
        True

        ```
        """

        # 1) Build embeddings map keyed by (group, feature)
        is_grouped = getattr(self, "is_grouped", False)
        flat_group_key = "__flat__"
        group_iter = self.items() if is_grouped else [(flat_group_key, self)]
        all_embeddings = {}
        for gkey, sub in group_iter:
            for feat_name, features in sub.features_dict.items():
                embed_df = features.embedding_df(embedding_dict).astype(np.float32)
                if lowmem:
                    embed_df = embed_df.iloc[::decimation_factor]
                all_embeddings[(gkey, feat_name)] = embed_df

        combined = pd.concat(
            all_embeddings.values(),
            keys=all_embeddings.keys(),
            names=["group", "feature", "frame"],
        )

        # 2) Optional scaling/normalization
        if custom_scaling is not None and (
            auto_normalize or rescale_factors is not None
        ):
            raise ValueError(
                "custom_scaling is mutually exclusive with auto_normalize or rescale_factors"
            )
        if auto_normalize:
            combined, normalization_factors = normalize_df(combined)
        elif rescale_factors is not None:
            combined = apply_normalization_to_df(combined, rescale_factors)
            normalization_factors = None
        elif custom_scaling is not None:
            combined = apply_custom_scaling(combined, custom_scaling)
            normalization_factors = None
        else:
            normalization_factors = None

        # 3) Cluster
        valid_mask = combined.notna().all(axis=1)
        valid_combined = combined[valid_mask]
        model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
            valid_combined
        )
        centroids = pd.DataFrame(model.cluster_centers_, columns=combined.columns)

        # 4) Assign labels
        combined_labels = pd.Series(np.nan, index=combined.index)
        combined_labels.loc[valid_mask] = model.labels_

        # 5) Reconstruct results (nested for grouped, flat dict for flat)
        meta = {
            "embedding_dict": embedding_dict,
            "n_clusters": n_clusters,
            "random_state": random_state,
            "auto_normalize": auto_normalize,
            "rescale_factors": rescale_factors,
            "lowmem": lowmem,
            "decimation_factor": decimation_factor,
        }

        if lowmem:
            # Assign by nearest centroid, item-by-item (wrap into FeaturesResult)
            if is_grouped:
                result_dict = {}
                for gkey, sub in self.items():
                    group_map = {}
                    for feat_name, feat in sub.features_dict.items():
                        labels = feat.assign_clusters_by_centroids(
                            embedding_dict, centroids
                        )
                        group_map[feat_name] = FeaturesResult(
                            labels, feat, f"kmeans_{n_clusters}", meta
                        )
                    result_dict[gkey] = group_map
            else:
                result_dict = {}
                for feat_name, feat in self.features_dict.items():
                    labels = feat.assign_clusters_by_centroids(
                        embedding_dict, centroids
                    )
                    result_dict[feat_name] = FeaturesResult(
                        labels, feat, f"kmeans_{n_clusters}", meta
                    )
        else:
            if is_grouped:
                result_dict = {}
                for gkey, sub in self.items():
                    group_map = {}
                    for feat_name, feat in sub.features_dict.items():
                        labels = combined_labels.xs(
                            (gkey, feat_name), level=["group", "feature"]
                        ).astype("Int64")
                        group_map[feat_name] = FeaturesResult(
                            labels, feat, f"kmeans_{n_clusters}", meta
                        )
                    result_dict[gkey] = group_map
            else:
                result_dict = {}
                for feat_name, feat in self.features_dict.items():
                    labels = combined_labels.xs(
                        (flat_group_key, feat_name), level=["group", "feature"]
                    ).astype("Int64")
                    result_dict[feat_name] = FeaturesResult(
                        labels, feat, f"kmeans_{n_clusters}", meta
                    )

        return BatchResult(result_dict, self), centroids, normalization_factors

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
                for start, length in zip(starts, lengths)
            ]
            test_X = [
                apply_normalization_to_df(pd.DataFrame(x), rescale_factors).values
                for x in test_X
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
        for x, y, h in zip(test_X, test_y, test_handles):
            x_df = pd.DataFrame(x, columns=get_source_columns(h))
            y_df = pd.DataFrame(y, columns=get_target_columns(h))
            preds = predictor.predict(x_df)
            preds = preds.reindex(index=y_df.index, columns=y_df.columns)
            rms = Features.rms_error_between_embeddings(
                y_df, preds, rescale=normalize_pred
            )
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
        Cross-prediction over grouped collections.

        - If grouped: computes within-group leave-one-out and between-group predictions.
        - If flat: computes within-collection leave-one-out only.

        Returns a dict with keys:
            'within': {group_key or '__flat__': {handle: rms_series}}
            'between': {fromA_to_B: {target_handle: rms_series}}  (grouped only)
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
                    source_handles = [
                        (g1, h) for h in self._obj_dict[g1].features_dict.keys()
                    ]
                    target_handles = [
                        (g2, h) for h in self._obj_dict[g2].features_dict.keys()
                    ]
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
                            self._obj_dict[g2].features_dict.keys(), rms_list
                        )
                    }
                    key = f"from{g1}_to_{g2}"
                    results["between"][key] = rms_dict

        return results

    @dev_mode
    @staticmethod
    def plot_cross_predict_vs_within(results, from_group, to_group, show=True):
        import numpy as np
        import matplotlib.pyplot as plt

        # Keys
        between_key = f"from{from_group}_to_{to_group}"
        within_key = to_group
        between_dict = results["between"].get(between_key, {})
        within_dict = results["within"].get(within_key, {})
        handles = sorted(set(between_dict.keys()) & set(within_dict.keys()))
        if not handles:
            raise ValueError(
                f"No overlapping handles between {between_key} and {within_key}"
            )
        between_means = [between_dict[h].mean(skipna=True) for h in handles]
        within_means = [within_dict[h].mean(skipna=True) for h in handles]
        diff_means = [b - w for b, w in zip(between_means, within_means)]
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
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        records = []
        if within_keys is not None:
            for coll in within_keys:
                for feat, series in results["within"].get(coll, {}).items():
                    arr = series.dropna().values
                    for v in arr:
                        records.append(
                            {"Category": f"within_{coll}", "Feature": feat, "RMS": v}
                        )
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
            sns.pointplot(
                data=means, x="Feature", y="RMS", hue="Category", dodge=True, ax=ax1
            )
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
    def dumbbell_plot_cross_predict(
        results, within_key, between_key, figsize=(3, 3), show=True
    ):
        import pandas as pd
        import matplotlib.pyplot as plt

        features = sorted(
            set(
                list(results["within"].get(within_key, {}).keys())
                + list(results["between"].get(between_key, {}).keys())
            )
        )
        data = []
        for feat in features:
            mean_within = (
                results["within"]
                .get(within_key, {})
                .get(feat, pd.Series(dtype=float))
                .mean()
            )
            mean_between = (
                results["between"]
                .get(between_key, {})
                .get(feat, pd.Series(dtype=float))
                .mean()
            )
            data.append(
                {"Feature": feat, "Within": mean_within, "Between": mean_between}
            )
        df = pd.DataFrame(data)
        x = [0, 1]
        plt.figure(figsize=figsize)
        for _, row in df.iterrows():
            plt.plot(x, [row["Within"], row["Between"]], color="gray", lw=2, zorder=1)
            plt.scatter(
                x, [row["Within"], row["Between"]], s=60, color="black", zorder=2
            )
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
        Train a regressor to predict a target embedding from a feature embedding on this Features object.
        Uses predictor_cls (default: KNNPredictor) and passes predictor_kwargs.
        If normalize_source is True, normalize the source embedding before training and return the rescale factors.
        Returns the trained model, input columns, target columns, and (optionally) the rescale factors.
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
        Predict using a trained regressor on this Features object.
        If rescale_factors is provided, normalize the source embedding before prediction.
        The prediction will match the shape and columns of self.embedding_df(target_embedding).
        """
        test_embed = self.embedding_df(source_embedding)
        if rescale_factors is not None:
            test_embed = apply_normalization_to_df(test_embed, rescale_factors)
        target_embed = self.embedding_df(target_embedding)
        preds = model.predict(test_embed)
        # Ensure the output DataFrame has the same index and columns as target_embed
        preds = pd.DataFrame(
            preds, index=target_embed.index, columns=target_embed.columns
        )
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
                figs_axes[gkey] = sub.plot(
                    arg, figsize=figsize, show=show, title=str(gkey)
                )
            return figs_axes

        if arg is None:
            # Plot all columns for each Features object
            features_dict = {
                handle: obj.data for handle, obj in self.features_dict.items()
            }
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
        fig, axes = plt.subplots(
            n, 1, figsize=(figsize[0], figsize[1] * n), sharex=True
        )
        if n == 1:
            axes = [axes]
        for ax, (handle, data) in zip(axes, features_dict.items()):
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
            for group_dict in results_dict.values():
                for v in group_dict.values():
                    if hasattr(v, "store"):
                        v.store(name=name, meta=meta, overwrite=overwrite)
                    else:
                        raise ValueError(f"{v} is not a FeaturesResult object")
            return
        # Flat case
        for v in results_dict.values():
            if hasattr(v, "store"):
                v.store(name=name, meta=meta, overwrite=overwrite)
            else:
                raise ValueError(f"{v} is not a FeaturesResult object")

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
