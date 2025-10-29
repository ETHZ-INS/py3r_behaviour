from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from py3r.behaviour.features.features_collection import (
    FeaturesCollection,
    FeaturesResult,
)
from py3r.behaviour.features.features import Features
from py3r.behaviour.tracking.multiple_tracking_collection import (
    MultipleTrackingCollection,
)
from py3r.behaviour.util.collection_utils import _Indexer, BatchResult
from py3r.behaviour.util.dev_utils import dev_mode, discontinued_method
from py3r.behaviour.util.series_utils import (
    normalize_df,
    apply_normalization_to_df,
    custom_scaling as apply_custom_scaling,
)
from py3r.behaviour.util.base_collection import BaseMultipleCollection


class MultipleFeaturesCollection(BaseMultipleCollection):
    """
    Collection of FeaturesCollection objects, keyed by name.
    """

    _element_type = FeaturesCollection
    _multiple_collection_type = "MultipleFeaturesCollection"

    def __init__(self, features_collections: dict[str, FeaturesCollection]):
        super().__init__(features_collections)

    @property
    def features_collections(self):
        return self._obj_dict

    @classmethod
    def from_multiple_tracking_collection(
        cls,
        multiple_tracking_collection: MultipleTrackingCollection,
        feature_cls=Features,
    ):
        """
        Factory method to create MultipleFeaturesCollection from a MultipleTrackingCollection object.
        """
        collections = {}
        for (
            coll_name,
            tracking_collection,
        ) in multiple_tracking_collection.tracking_collections.items():
            collections[coll_name] = FeaturesCollection.from_tracking_collection(
                tracking_collection, feature_cls=feature_cls
            )
        return cls(collections)

    def store(
        self,
        results_dict: dict[str, dict[str, FeaturesResult]],
        name: str = None,
        meta: dict = None,
        overwrite: bool = False,
    ):
        """
        Store all FeaturesResult objects in a two-layer dict (as returned by batch methods).
        Example:
            results = multiple_features_collection.speed('nose')
            multiple_features_collection.store(results)
        """
        for group_dict in results_dict.values():
            for v in group_dict.values():
                if hasattr(v, "store"):
                    v.store(name=name, meta=meta, overwrite=overwrite)
                else:
                    raise ValueError(f"{v} is not a FeaturesResult object")

    def plot(self, arg=None, figsize=(8, 2), show=True):
        """
        Plot features for all collections in the MultipleFeaturesCollection.
        - If arg is a BatchResult or dict: treat as batch result and plot for each collection.
        - Otherwise: treat as column name(s) or None and plot for each collection.
        """
        figs_axes = {}
        # If arg is a BatchResult or dict, treat as batch result
        if isinstance(arg, dict):
            for coll_name, group_dict in arg.items():
                if coll_name in self.features_collections:
                    figs_axes[coll_name] = self.features_collections[coll_name].plot(
                        group_dict, figsize=figsize, show=show, title=coll_name
                    )
        else:
            for coll_name, collection in self.features_collections.items():
                figs_axes[coll_name] = collection.plot(
                    arg, figsize=figsize, show=show, title=coll_name
                )
        return figs_axes

    def cluster_embedding(
        self,
        embedding_dict: dict[str, list[int]],
        n_clusters: int,
        random_state: int = 0,
        auto_normalize: bool = False,
        rescale_factors: dict | None = None,
        lowmem: bool = False,
        decimation_factor: int = 10,
        custom_scaling: dict[str, dict] | None = None,
    ):
        # Step 1: Build all embeddings
        all_embeddings = {}
        for coll_name, collection in self.features_collections.items():
            for feat_name, features in collection.features_dict.items():
                embed_df = features.embedding_df(embedding_dict).astype(np.float32)
                if lowmem:
                    embed_df_decimated = embed_df.iloc[::decimation_factor]
                    embed_df = embed_df_decimated
                all_embeddings[(coll_name, feat_name)] = embed_df

        # Step 2: Concatenate
        combined = pd.concat(
            all_embeddings.values(),
            keys=all_embeddings.keys(),
            names=["collection", "feature", "frame"],
        )

        # Step 2a (optional): Normalize or custom scaling (mutually exclusive)
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
        elif custom_scaling is not None:
            combined = apply_custom_scaling(combined, custom_scaling)

        # Step 3: Mask
        valid_mask = combined.notna().all(axis=1)
        valid_combined = combined[valid_mask]

        # Step 4: Cluster
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
            valid_combined
        )
        centroids = pd.DataFrame(model.cluster_centers_, columns=combined.columns)

        # Step 5: Assign labels
        clustermeta = {
            "embedding_dict": embedding_dict,
            "n_clusters": n_clusters,
            "random_state": random_state,
            "auto_normalize": auto_normalize,
            "rescale_factors": rescale_factors,
            "lowmem": lowmem,
            "decimation_factor": decimation_factor,
        }
        combined_labels = pd.Series(np.nan, index=combined.index)
        combined_labels.loc[valid_mask] = model.labels_
        # Step 6: Split
        if lowmem:
            nested_labels = self.assign_clusters_by_centroids(embedding_dict, centroids)
        else:
            nested_labels = {}
            for (coll_name, feat_name), _ in all_embeddings.items():
                idx = (coll_name, feat_name)
                # Get all rows for this (collection, feature)
                labels = combined_labels.xs(idx, level=["collection", "feature"])
                if coll_name not in nested_labels:
                    nested_labels[coll_name] = {}
                nested_labels[coll_name][feat_name] = FeaturesResult(
                    labels.astype("Int64"),
                    self[coll_name][feat_name],
                    f"kmeans_{n_clusters}",
                    clustermeta,
                )
            nested_labels = BatchResult(nested_labels, self)

        if auto_normalize:
            return nested_labels, centroids, normalization_factors
        else:
            return nested_labels, centroids, None

    @discontinued_method
    def knn_cross_predict_rms_matrix(
        self,
        source_embedding: dict[str, list[int]],
        target_embedding: dict[str, list[int]],
        n_neighbors: int = 5,
        normalize_source: bool = False,
        normalize_pred: dict | str = None,
        collections: list[str] = None,
    ):
        """
        For each Features object in each selected FeaturesCollection in the MultipleFeaturesCollection,
        train a kNN regressor and use it to predict the target embedding in every Features object
        in each selected FeaturesCollection, storing the full RMS error Series in a DataFrame for each collection pair.
        The same set of collections is used for both source and target.
        Returns a dict of DataFrames: { "from<source>_to_<target>": DataFrame }.
        """
        results = {}
        # Determine which collections to use
        all_keys = list(self.features_collections.keys())
        if collections is None:
            collections = all_keys

        for source_coll_name in collections:
            source_coll = self.features_collections[source_coll_name]
            for target_coll_name in collections:
                target_coll = self.features_collections[target_coll_name]
                df = pd.DataFrame(
                    index=source_coll.features_dict.keys(),
                    columns=target_coll.features_dict.keys(),
                    dtype=object,
                )
                for source_feat_name, source_feat in source_coll.features_dict.items():
                    # Train regressor
                    if normalize_source:
                        model, in_cols, out_cols, rescale_factors = (
                            source_feat.train_knn_regressor(
                                source_embedding=source_embedding,
                                target_embedding=target_embedding,
                                n_neighbors=n_neighbors,
                                normalize_source=True,
                            )
                        )
                    else:
                        model, in_cols, out_cols = source_feat.train_knn_regressor(
                            source_embedding=source_embedding,
                            target_embedding=target_embedding,
                            n_neighbors=n_neighbors,
                        )
                        rescale_factors = None

                    for (
                        target_feat_name,
                        target_feat,
                    ) in target_coll.features_dict.items():
                        preds = target_feat.predict_knn(
                            model,
                            source_embedding=source_embedding,
                            target_embedding=target_embedding,
                            rescale_factors=rescale_factors,
                        )
                        ground_truth = target_feat.embedding_df(target_embedding)
                        rms = Features.rms_error_between_embeddings(
                            ground_truth, preds, rescale=normalize_pred
                        )
                        df.at[source_feat_name, target_feat_name] = rms
                key = f"from{source_coll_name}_to_{target_coll_name}"
                results[key] = df
        return results

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
        Helper to train a predictor and compute RMS error for each test_feat.
        Returns a list of RMS Series (one per test_feat, in order).
        """
        import numpy as np
        import pandas as pd

        if predictor_kwargs is None:
            predictor_kwargs = {}

        # 1. Prepare embeddings (now as arrays)
        train_X = [get_source_array(h) for h in train_handles]
        train_y = [get_target_array(h) for h in train_handles]
        test_X = [get_source_array(h) for h in test_handles]
        test_y = [get_target_array(h) for h in test_handles]

        # 2. Normalize if needed (convert to DataFrame for normalization, then back to array)
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

        # 3. Train predictor (use arrays, convert to DataFrame if needed by predictor)
        train_X_all = np.vstack(train_X)
        train_y_all = np.vstack(train_y)
        # Use columns from the first train handle for DataFrame conversion if needed
        train_X_df = pd.DataFrame(
            train_X_all, columns=get_source_columns(train_handles[0])
        )
        train_y_df = pd.DataFrame(
            train_y_all, columns=get_target_columns(train_handles[0])
        )
        predictor = predictor_cls(**predictor_kwargs)
        predictor.fit(train_X_df, train_y_df)

        # 4. Predict and compute RMS for each test_feat
        rms_list = []
        for x, y, h in zip(test_X, test_y, test_handles):
            # Convert to DataFrame for predictor
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
        set1: list[str] = None,
        set2: list[str] = None,
        predictor_cls=None,
        predictor_kwargs=None,
    ):
        """
        Performs two types of cross-prediction:
        1. Within-collection leave-one-out: For each Features object in each collection in set1 or set2 (union), trains a predictor on all other Features objects in the same collection, predicts on the left-out object, and stores the RMS error Series.
        2. Between-collection: For each ordered pair of collections (A, B) with A in set1, B in set2, and A != B, trains a predictor on all Features objects in A, predicts on all Features objects in B, and stores the RMS error Series for each Features object in B.

        Args:
            source_embedding: dict mapping feature names to time shifts for input embedding.
            target_embedding: dict mapping feature names to time shifts for target embedding.
            normalize_source: Whether to normalize the source embedding during training.
            normalize_pred: Normalization for RMS calculation ('auto', dict, or None).
            set1: List of collection keys for the first set (default: all).
            set2: List of collection keys for the second set (default: all).
            predictor_cls: Predictor class to use (default: KNNPredictor).
            predictor_kwargs: dict of kwargs to pass to predictor_cls (e.g. n_neighbors, n_components, etc.)

        Returns:
            dict with keys:
                'within': {collection: {feature_name: rms_series}}
                'between': {fromA_to_B: {target_feature_name: rms_series}}
        """
        if predictor_cls is None:
            from py3r.behaviour.predictors import KNNPredictor

            predictor_cls = KNNPredictor
        if predictor_kwargs is None:
            predictor_kwargs = {}
        results = {"within": {}, "between": {}}
        all_keys = list(self.features_collections.keys())
        if set1 is None:
            set1 = all_keys
        if set2 is None:
            set2 = all_keys
        # Union for within
        within_collections = sorted(set(set1) | set(set2))

        # --- Precompute and cache all embeddings as numpy arrays (with index/columns) ---
        # Cache structure: {handle: {'array': ..., 'index': ..., 'columns': ...}}
        embedding_cache = {}
        for coll in set1 + set2:
            for handle, feat in self.features_collections[coll].features_dict.items():
                if handle not in embedding_cache:
                    src_df = feat.embedding_df(source_embedding)
                    tgt_df = feat.embedding_df(target_embedding)
                    embedding_cache[handle] = {
                        "source_array": src_df.values,
                        "source_index": src_df.index,
                        "source_columns": src_df.columns,
                        "target_array": tgt_df.values,
                        "target_index": tgt_df.index,
                        "target_columns": tgt_df.columns,
                    }

        # --- Helper to fetch embedding arrays by handle ---
        def get_source_array(handle):
            return embedding_cache[handle]["source_array"]

        def get_target_array(handle):
            return embedding_cache[handle]["target_array"]

        def get_source_index(handle):
            return embedding_cache[handle]["source_index"]

        def get_target_index(handle):
            return embedding_cache[handle]["target_index"]

        def get_source_columns(handle):
            return embedding_cache[handle]["source_columns"]

        def get_target_columns(handle):
            return embedding_cache[handle]["target_columns"]

        # --- Pass the cache and helpers to _train_and_predict_rms ---
        # Within-collection leave-one-out
        for coll_name in within_collections:
            coll = self.features_collections[coll_name]
            rms_dict = {}
            for left_out_name, left_out_feat in coll.features_dict.items():
                train_handles = [n for n in coll.features_dict if n != left_out_name]
                test_handles = [left_out_name]
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
                rms_dict[left_out_name] = rms_list[0]
            results["within"][coll_name] = rms_dict

        # Between-collection: all ordered pairs (A, B) with A in set1, B in set2, and A != B
        for coll1 in set1:
            for coll2 in set2:
                if coll1 == coll2:
                    continue
                source_coll = self.features_collections[coll1]
                target_coll = self.features_collections[coll2]
                train_handles = list(source_coll.features_dict.keys())
                test_handles = list(target_coll.features_dict.keys())
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
                rms_dict = {name: rms for name, rms in zip(test_handles, rms_list)}
                key = f"from{coll1}_to_{coll2}"
                results["between"][key] = rms_dict
        # Also do all ordered pairs (A, B) with A in set2, B in set1, and A != B
        for coll1 in set2:
            for coll2 in set1:
                if coll1 == coll2:
                    continue
                source_coll = self.features_collections[coll1]
                target_coll = self.features_collections[coll2]
                train_handles = list(source_coll.features_dict.keys())
                test_handles = list(target_coll.features_dict.keys())
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
                rms_dict = {name: rms for name, rms in zip(test_handles, rms_list)}
                key = f"from{coll1}_to_{coll2}"
                results["between"][key] = rms_dict
        return results

    @dev_mode
    @staticmethod
    def plot_cross_predict_vs_within(
        results, from_collection, to_collection, show=True
    ):
        """
        Plot mean RMS for between (fromX_to_Y), within (withinY), and their difference for each Features object in 'to_collection'.
        """
        # Keys
        between_key = f"from{from_collection}_to_{to_collection}"
        within_key = to_collection

        # Get dicts of {handle: pd.Series}
        between_dict = results["between"].get(between_key, {})
        within_dict = results["within"].get(within_key, {})

        # Handles present in both
        handles = sorted(set(between_dict.keys()) & set(within_dict.keys()))
        if not handles:
            raise ValueError(
                f"No overlapping handles between {between_key} and {within_key}"
            )

        # Compute means
        between_means = [between_dict[h].mean(skipna=True) for h in handles]
        within_means = [within_dict[h].mean(skipna=True) for h in handles]
        diff_means = [b - w for b, w in zip(between_means, within_means)]

        x = np.arange(len(handles))
        width = 0.3

        fig, ax = plt.subplots(figsize=(max(8, len(handles) * 0.7), 5))
        # ax.bar(x - width, between_means, width, label=f'from{from_collection}_to_{to_collection}')
        # ax.bar(x, within_means, width, label=f'within_{to_collection}')
        ax.bar(x + width, diff_means, width, label="between - within")

        ax.set_xticks(x)
        ax.set_xticklabels(handles, rotation=90)
        ax.set_ylabel("Mean RMS difference")
        ax.set_title(f"Cross-predict vs Within: {from_collection} → {to_collection}")
        # ax.legend()

        from scipy.stats import ttest_rel

        # Paired t-test
        t_stat, p_value = ttest_rel(between_means, within_means, nan_policy="omit")

        # Annotate on the plot
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
        plot_type="bar",  # 'bar', 'point', or 'violin'
        figsize=(10, 6),
        show=True,
    ):
        """
        Plot summary statistics from cross_predict_rms_leaveoneout_and_between results.

        Args:
            results: dict as returned by cross_predict_rms_leaveoneout_and_between
            within_keys: list of collection names to include from 'within'
            between_keys: list of between keys (e.g. 'fromA_to_B') to include from 'between'
            plot_type: 'bar', 'point', or 'violin'
            figsize: tuple for figure size
            show: whether to call plt.show()
        """
        # Gather data
        records = []
        # Within
        if within_keys is not None:
            for coll in within_keys:
                for feat, series in results["within"].get(coll, {}).items():
                    arr = series.dropna().values
                    for v in arr:
                        records.append(
                            {"Category": f"within_{coll}", "Feature": feat, "RMS": v}
                        )
        # Between
        if between_keys is not None:
            for comp in between_keys:
                for feat, series in results["between"].get(comp, {}).items():
                    arr = series.dropna().values
                    for v in arr:
                        records.append({"Category": comp, "Feature": feat, "RMS": v})

        df = pd.DataFrame(records)

        plt.figure(figsize=figsize)
        if plot_type == "bar":
            # Bar plot: mean of means per category
            means = df.groupby("Category").RMS.mean()
            means.plot(kind="bar", yerr=df.groupby("Category").RMS.std(), capsize=4)
            plt.ylabel("Mean RMS (mean of means)")
            plt.title("RMS prediction error by category")
        elif plot_type == "point":
            # Point plot: mean RMS per feature, grouped by category
            means = df.groupby(["Category", "Feature"]).RMS.mean().reset_index()
            # Pivot to get within and between as columns
            pivot = means.pivot(index="Feature", columns="Category", values="RMS")
            # Try to infer the within and between column names
            within_col = [c for c in pivot.columns if c.startswith("within_")]
            between_col = [c for c in pivot.columns if not c.startswith("within_")]
            if len(within_col) == 1 and len(between_col) == 1:
                pivot["mean_diff"] = pivot[between_col[0]] - pivot[within_col[0]]
            else:
                pivot["mean_diff"] = np.nan  # fallback if ambiguous

            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(figsize[0], figsize[1] * 1.5),
                sharex=True,
                gridspec_kw={"height_ratios": [2, 1]},
            )

            # Point plot
            sns.pointplot(
                data=means, x="Feature", y="RMS", hue="Category", dodge=True, ax=ax1
            )
            ax1.set_ylabel("mean RMS error")
            ax1.set_title(f"{within_keys[0]} vs {within_keys[1]}")
            ax1.tick_params(axis="x", rotation=90)

            # Bar plot of mean difference
            ax2.bar(pivot.index, pivot["mean_diff"])
            ax2.axhline(0, color="gray", linestyle="--")
            ax2.set_ylabel("Mean (Between - Within)")
            ax2.set_title("Mean RMS Difference per Video")
            ax2.tick_params(axis="x", rotation=90)

            plt.tight_layout()
            if show:
                plt.show()
            return df  # Return the DataFrame for further inspection if needed
        elif plot_type == "violin":
            # Violin plot: all raw RMS values
            sns.violinplot(data=df, x="Category", y="RMS", inner="point")
            plt.ylabel("RMS")
            plt.title("RMS prediction error by category")
        else:
            raise ValueError("plot_type must be 'bar', 'point', or 'violin'")

        plt.tight_layout()
        if show:
            plt.show()
        return df  # Return the DataFrame for further inspection if needed

    @dev_mode
    @staticmethod
    def dumbbell_plot_cross_predict(
        results, within_key, between_key, figsize=(3, 3), show=True
    ):
        """
        Plot a vertical dumbbell plot: x-axis is category (Within/Between), y-axis is RMS,
        each feature is a line connecting its within and between mean RMS.

        Args:
            results: dict as returned by cross_predict_rms_leaveoneout_and_between
            within_key: collection name for 'within' (e.g., 'POD14')
            between_key: key for 'between' (e.g., 'fromX_to_POD14')
            figsize: tuple for figure size
            show: whether to call plt.show()
        """
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

        # Prepare for plotting
        x = [0, 1]  # 0 = Within, 1 = Between
        plt.figure(figsize=figsize)
        for i, row in df.iterrows():
            plt.plot(x, [row["Within"], row["Between"]], color="gray", lw=2, zorder=1)
            plt.scatter(
                x, [row["Within"], row["Between"]], s=60, color="black", zorder=2
            )
            # plt.text(-0.05, row['Within'], row['Feature'], va='center', ha='right', fontsize=9)
        plt.xticks(x, ["Within", "Between"])
        plt.ylabel("Mean RMS")
        plt.title(f"Dumbbell Plot: {within_key} vs {between_key}")
        plt.tight_layout()
        if show:
            plt.show()
        return df  # Return the DataFrame for further inspection if needed

    @property
    def loc(self):
        return _Indexer(self, self._loc)

    @property
    def iloc(self):
        return _Indexer(self, self._iloc)

    def _loc(self, idx):
        return self.__class__(
            {k: v.loc[idx] for k, v in self.features_collections.items()}
        )

    def _iloc(self, idx):
        return self.__class__(
            {k: v.iloc[idx] for k, v in self.features_collections.items()}
        )
