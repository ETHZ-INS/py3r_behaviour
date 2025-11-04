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


class FeaturesCollection(BaseCollection):
    """
    Collection of Features objects, keyed by name.
    note: type-hints refer to Features, but factory methods allow for other classes
    these are intended ONLY for subclasses of Features, and this is enforced
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

        - Flat collection: clusters across all Features in the collection.
        - Grouped collection: clusters across all Features in all groups, and returns
          a nested BatchResult of FeaturesResult labels per group and feature.

        Returns:
            - If grouped: (nested_labels: BatchResult[{group: {feature: FeaturesResult}}], centroids: DataFrame, normalization_factors: Optional[dict])
            - If flat: (labels_dict: dict[handle -> Series[Int]], centroids: DataFrame)
        """

        # Grouped path mirrors former MultipleFeaturesCollection.cluster_embedding
        if getattr(self, "is_grouped", False):
            all_embeddings = {}
            for gkey, sub in self.items():
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

            valid_mask = combined.notna().all(axis=1)
            valid_combined = combined[valid_mask]

            model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
                valid_combined
            )
            centroids = pd.DataFrame(model.cluster_centers_, columns=combined.columns)

            combined_labels = pd.Series(np.nan, index=combined.index)
            combined_labels.loc[valid_mask] = model.labels_

            if lowmem:
                # Assign by nearest centroid per item
                nested_labels = {}
                for (gkey, feat_name), _ in all_embeddings.items():
                    feat = self[gkey][feat_name]
                    nested_labels.setdefault(gkey, {})[feat_name] = FeaturesResult(
                        feat.assign_clusters_by_centroids(embedding_dict, centroids),
                        feat,
                        f"kmeans_{n_clusters}",
                        {
                            "embedding_dict": embedding_dict,
                            "n_clusters": n_clusters,
                            "random_state": random_state,
                            "auto_normalize": auto_normalize,
                            "rescale_factors": rescale_factors,
                            "lowmem": lowmem,
                            "decimation_factor": decimation_factor,
                        },
                    )
            else:
                nested_labels = {}
                for (gkey, feat_name), _ in all_embeddings.items():
                    labels = combined_labels.xs(
                        (gkey, feat_name), level=["group", "feature"]
                    ).astype("Int64")
                    feat = self[gkey][feat_name]
                    nested_labels.setdefault(gkey, {})[feat_name] = FeaturesResult(
                        labels,
                        feat,
                        f"kmeans_{n_clusters}",
                        {
                            "embedding_dict": embedding_dict,
                            "n_clusters": n_clusters,
                            "random_state": random_state,
                            "auto_normalize": auto_normalize,
                            "rescale_factors": rescale_factors,
                            "lowmem": lowmem,
                            "decimation_factor": decimation_factor,
                        },
                    )

            return BatchResult(nested_labels, self), centroids, normalization_factors

        # Flat path: original behavior
        embedding_dfs = {
            name: f.embedding_df(embedding_dict)
            for name, f in self.features_dict.items()
        }
        columns = next(iter(embedding_dfs.values())).columns
        if not all(df.columns.equals(columns) for df in embedding_dfs.values()):
            raise ValueError("All embeddings must have the same columns")
        combined = pd.concat(embedding_dfs.values(), axis=0, keys=embedding_dfs.keys())
        valid_mask = combined.notna().all(axis=1)
        valid_combined = combined[valid_mask]
        model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
            valid_combined
        )
        centroids = pd.DataFrame(model.cluster_centers_, columns=combined.columns)
        combined_labels = pd.Series(np.nan, index=combined.index, name="cluster")
        combined_labels.loc[valid_mask] = model.labels_
        labels_dict = {
            name: combined_labels.xs(name, level=0).astype("Int64")
            for name in embedding_dfs.keys()
        }
        return labels_dict, centroids

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
