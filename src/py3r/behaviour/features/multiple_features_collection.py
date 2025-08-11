import warnings
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
from py3r.behaviour.exceptions import BatchProcessError
from py3r.behaviour.util.collection_utils import _Indexer, BatchResult
from py3r.behaviour.util.dev_utils import dev_mode
from py3r.behaviour.util.bmicro_utils import (
    train_knn_from_embeddings,
    predict_knn_on_embedding,
)
from py3r.behaviour.util.series_utils import normalize_df, apply_normalization_to_df


class MultipleFeaturesCollection:
    """
    Collection of FeaturesCollection objects, keyed by name.
    """

    def __init__(self, features_collections: dict[str, FeaturesCollection]):
        self.features_collections = features_collections

    def __setitem__(self, key, value):
        """
        Set FeaturesCollection by handle (str).
        """
        if not isinstance(value, FeaturesCollection):
            raise TypeError(
                f"Value must be a FeaturesCollection, got {type(value).__name__}"
            )
        warnings.warn(
            "Direct assignment to MultipleFeaturesCollection is deprecated and may be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.features_collections[key] = value

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

    def __getattr__(self, name):
        def batch_method(*args, **kwargs):
            results = {}
            for coll_name, collection in self.features_collections.items():
                try:
                    results[coll_name] = getattr(collection, name)(*args, **kwargs)
                except Exception as e:
                    raise BatchProcessError(
                        collection_name=coll_name,
                        object_name=getattr(e, "object_name", None),
                        method=getattr(e, "method", None),
                        original_exception=getattr(e, "original_exception", e),
                    ) from e
            return BatchResult(results, self)

        return batch_method

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
    ):
        # Step 1: Build all embeddings
        all_embeddings = {}
        for coll_name, collection in self.features_collections.items():
            for feat_name, features in collection.features_dict.items():
                embed_df = features.embedding_df(embedding_dict)
                all_embeddings[(coll_name, feat_name)] = embed_df

        # Step 2: Concatenate
        combined = pd.concat(
            all_embeddings.values(),
            keys=all_embeddings.keys(),
            names=["collection", "feature", "frame"],
        )

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
        combined_labels = pd.Series(np.nan, index=combined.index, name="cluster")
        combined_labels.loc[valid_mask] = model.labels_

        # Step 6: Split
        nested_labels = {}
        for (coll_name, feat_name), _ in all_embeddings.items():
            idx = (coll_name, feat_name)
            # Get all rows for this (collection, feature)
            labels = combined_labels.xs(idx, level=["collection", "feature"])
            if coll_name not in nested_labels:
                nested_labels[coll_name] = {}
            nested_labels[coll_name][feat_name] = labels.astype("Int64")

        # Step 7: Return
        return nested_labels, centroids

    @dev_mode
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
    def cross_predict_rms(
        self,
        source_embedding: dict[str, list[int]],
        target_embedding: dict[str, list[int]],
        n_neighbors: int = 5,
        normalize_source: bool = False,
        normalize_pred: dict | str = None,
        set1: list[str] = None,
        set2: list[str] = None,
    ):
        """
        Performs two types of cross-prediction:
        1. Within-collection leave-one-out: For each Features object in each collection in set1 or set2 (union), trains a kNN regressor on all other Features objects in the same collection, predicts on the left-out object, and stores the RMS error Series.
        2. Between-collection: For each ordered pair of collections (A, B) with A in set1, B in set2, and A != B, trains a kNN regressor on all Features objects in A, predicts on all Features objects in B, and stores the RMS error Series for each Features object in B.

        Args:
            source_embedding: dict mapping feature names to time shifts for input embedding.
            target_embedding: dict mapping feature names to time shifts for target embedding.
            n_neighbors: Number of neighbors for kNN.
            normalize_source: Whether to normalize the source embedding during training.
            normalize_pred: Normalization for RMS calculation ('auto', dict, or None).
            set1: List of collection keys for the first set (default: all).
            set2: List of collection keys for the second set (default: all).

        Returns:
            dict with keys:
                'within': {collection: {feature_name: rms_series}}
                'between': {fromA_to_B: {target_feature_name: rms_series}}
        """
        results = {"within": {}, "between": {}}
        all_keys = list(self.features_collections.keys())
        if set1 is None:
            set1 = all_keys
        if set2 is None:
            set2 = all_keys
        # Union for within
        within_collections = sorted(set(set1) | set(set2))

        # Within-collection leave-one-out
        for coll_name in within_collections:
            coll = self.features_collections[coll_name]
            rms_dict = {}
            for left_out_name, left_out_feat in coll.features_dict.items():
                # Train on all others
                train_feats = [
                    f for n, f in coll.features_dict.items() if n != left_out_name
                ]
                train_embeds = [f.embedding_df(source_embedding) for f in train_feats]
                target_embeds = [f.embedding_df(target_embedding) for f in train_feats]
                if normalize_source:
                    # Use normalization from the training set
                    train_embeds_norm, rescale_factors = normalize_df(
                        pd.concat(train_embeds)
                    )
                    lengths = [len(e) for e in train_embeds]
                    starts = np.cumsum([0] + lengths[:-1])
                    train_embeds_norm_list = [
                        train_embeds_norm.iloc[start : start + length]
                        for start, length in zip(starts, lengths)
                    ]
                    model, in_cols, out_cols = train_knn_from_embeddings(
                        train_embeds_norm_list, target_embeds, n_neighbors
                    )
                else:
                    model, in_cols, out_cols = train_knn_from_embeddings(
                        train_embeds, target_embeds, n_neighbors
                    )
                    rescale_factors = None
                # Predict on left-out
                if normalize_source and rescale_factors is not None:
                    test_embed = left_out_feat.embedding_df(source_embedding)
                    test_embed = apply_normalization_to_df(test_embed, rescale_factors)
                else:
                    test_embed = left_out_feat.embedding_df(source_embedding)
                target_embed = left_out_feat.embedding_df(target_embedding)
                preds = predict_knn_on_embedding(
                    model, test_embed, target_embed.columns
                )
                preds = preds.reindex(
                    index=target_embed.index, columns=target_embed.columns
                )
                rms = Features.rms_error_between_embeddings(
                    target_embed, preds, rescale=normalize_pred
                )
                rms_dict[left_out_name] = rms
            results["within"][coll_name] = rms_dict

        # Between-collection: all ordered pairs (A, B) with A in set1, B in set2, and A != B
        for coll1 in set1:
            for coll2 in set2:
                if coll1 == coll2:
                    continue
                source_coll = self.features_collections[coll1]
                target_coll = self.features_collections[coll2]
                # Train on all in coll1
                train_embeds = [
                    f.embedding_df(source_embedding)
                    for f in source_coll.features_dict.values()
                ]
                target_embeds = [
                    f.embedding_df(target_embedding)
                    for f in source_coll.features_dict.values()
                ]
                if normalize_source:
                    train_embeds_norm, rescale_factors = normalize_df(
                        pd.concat(train_embeds)
                    )
                    lengths = [len(e) for e in train_embeds]
                    starts = np.cumsum([0] + lengths[:-1])
                    train_embeds_norm_list = [
                        train_embeds_norm.iloc[start : start + length]
                        for start, length in zip(starts, lengths)
                    ]
                    model, in_cols, out_cols = train_knn_from_embeddings(
                        train_embeds_norm_list, target_embeds, n_neighbors
                    )
                else:
                    model, in_cols, out_cols = train_knn_from_embeddings(
                        train_embeds, target_embeds, n_neighbors
                    )
                    rescale_factors = None
                # Predict on all in coll2
                rms_dict = {}
                for target_feat_name, target_feat in target_coll.features_dict.items():
                    if normalize_source and rescale_factors is not None:
                        test_embed = target_feat.embedding_df(source_embedding)
                        test_embed = apply_normalization_to_df(
                            test_embed, rescale_factors
                        )
                    else:
                        test_embed = target_feat.embedding_df(source_embedding)
                    target_embed = target_feat.embedding_df(target_embedding)
                    preds = predict_knn_on_embedding(
                        model, test_embed, target_embed.columns
                    )
                    preds = preds.reindex(
                        index=target_embed.index, columns=target_embed.columns
                    )
                    rms = Features.rms_error_between_embeddings(
                        target_embed, preds, rescale=normalize_pred
                    )
                    rms_dict[target_feat_name] = rms
                key = f"from{coll1}_to_{coll2}"
                results["between"][key] = rms_dict
        # Also do all ordered pairs (A, B) with A in set2, B in set1, and A != B
        for coll1 in set2:
            for coll2 in set1:
                if coll1 == coll2:
                    continue
                source_coll = self.features_collections[coll1]
                target_coll = self.features_collections[coll2]
                # Train on all in coll1
                train_embeds = [
                    f.embedding_df(source_embedding)
                    for f in source_coll.features_dict.values()
                ]
                target_embeds = [
                    f.embedding_df(target_embedding)
                    for f in source_coll.features_dict.values()
                ]
                if normalize_source:
                    train_embeds_norm, rescale_factors = normalize_df(
                        pd.concat(train_embeds)
                    )
                    lengths = [len(e) for e in train_embeds]
                    starts = np.cumsum([0] + lengths[:-1])
                    train_embeds_norm_list = [
                        train_embeds_norm.iloc[start : start + length]
                        for start, length in zip(starts, lengths)
                    ]
                    model, in_cols, out_cols = train_knn_from_embeddings(
                        train_embeds_norm_list, target_embeds, n_neighbors
                    )
                else:
                    model, in_cols, out_cols = train_knn_from_embeddings(
                        train_embeds, target_embeds, n_neighbors
                    )
                    rescale_factors = None
                # Predict on all in coll2
                rms_dict = {}
                for target_feat_name, target_feat in target_coll.features_dict.items():
                    if normalize_source and rescale_factors is not None:
                        test_embed = target_feat.embedding_df(source_embedding)
                        test_embed = apply_normalization_to_df(
                            test_embed, rescale_factors
                        )
                    else:
                        test_embed = target_feat.embedding_df(source_embedding)
                    target_embed = target_feat.embedding_df(target_embedding)
                    preds = predict_knn_on_embedding(
                        model, test_embed, target_embed.columns
                    )
                    preds = preds.reindex(
                        index=target_embed.index, columns=target_embed.columns
                    )
                    rms = Features.rms_error_between_embeddings(
                        target_embed, preds, rescale=normalize_pred
                    )
                    rms_dict[target_feat_name] = rms
                key = f"from{coll1}_to_{coll2}"
                results["between"][key] = rms_dict
        return results

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
            sns.pointplot(data=means, x="Feature", y="RMS", hue="Category", dodge=True)
            plt.ylabel("mean RMS error")
            plt.title(f"{within_keys[0]} vs {within_keys[1]}")
            plt.xticks(rotation=90)
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

    def __getitem__(self, key):
        """
        Get FeaturesCollection by handle (str), by integer index, or by slice.
        """
        if isinstance(key, int):
            handle = list(self.features_collections)[key]
            return self.features_collections[handle]
        elif isinstance(key, slice):
            handles = list(self.features_collections)[key]
            return self.__class__({h: self.features_collections[h] for h in handles})
        else:
            return self.features_collections[key]

    def keys(self):
        """Return the keys of the features_collections."""
        return self.features_collections.keys()

    def values(self):
        """Return the values of the features_collections."""
        return self.features_collections.values()

    def items(self):
        """Return the items of the features_collections."""
        return self.features_collections.items()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.features_collections)} FeaturesCollection objects>"
