from __future__ import annotations
import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from py3r.behaviour.features.features import Features, FeaturesResult
from py3r.behaviour.features.multiple_features_collection import (
    MultipleFeaturesCollection,
)
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.exceptions import BatchProcessError
from py3r.behaviour.util.collection_utils import _Indexer, BatchResult
from py3r.behaviour.util.dev_utils import dev_mode
from py3r.behaviour.util.series_utils import normalize_df, apply_normalization_to_df


class FeaturesCollection:
    """
    Collection of Features objects, keyed by name.
    note: type-hints refer to Features, but factory methods allow for other classes
    these are intended ONLY for subclasses of Features, and this is enforced
    """

    features_dict: dict[str, Features]

    def __init__(self, features_dict: dict[str, Features]):
        self.features_dict = features_dict

    def __getattr__(self, name):
        def batch_method(*args, **kwargs):
            results = {}
            for key, obj in self.features_dict.items():
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
        # check that dict handles match tracking handles
        for handle, t in tracking_collection.tracking_dict.items():
            if handle != t.handle:
                raise ValueError(
                    f"Key '{handle}' does not match object's handle '{t.handle}'"
                )
        return cls(
            {
                handle: feature_cls(t)
                for handle, t in tracking_collection.tracking_dict.items()
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

    def groupby(self, tags):
        """
        Group the collection by one or more tags.
        Args:
            tags (str or list/tuple of str): Tag(s) to group by.
        Returns:
            MultipleCollection: A MultipleCollection object with groups named by tag values.
        Raises:
            ValueError: If any tag is missing for any element.
        """
        # Accept single tag as string
        if isinstance(tags, str):
            tags = [tags]
        tags = list(tags)

        groups = {}
        missing = []

        for obj in (
            self.tracking_dictvalues()
        ):  # assumes .values() yields the elements (e.g., Features, Tracking, etc.)
            # Check all tags are present
            try:
                key = tuple(str(obj.tags[tag]) for tag in tags)
            except KeyError as e:
                missing.append((getattr(obj, "handle", None), e.args[0]))
                continue
            groups.setdefault(key, []).append(obj)

        if missing:
            missing_str = "\n".join(f"{handle}: {tag}" for handle, tag in missing)
            raise ValueError(
                f"The following elements are missing required tags:\n{missing_str}"
            )

        # Create group names (e.g., 'male_treatment')
        def group_name(key_tuple):
            return "_".join(str(v) for v in key_tuple)

        # Build the group collections
        group_collections = {
            group_name(key): self.__class__.from_list(
                objs
            )  # assumes from_list constructor
            for key, objs in groups.items()
        }

        # Return a MultipleCollection of the appropriate type
        # You may need to adjust this line to use the correct MultipleCollection class
        return MultipleFeaturesCollection(
            group_collections
        )  # or MultipleTrackingCollection, etc.

    def cluster_embedding(
        self,
        embedding_dict: dict[str, list[int]],
        n_clusters: int,
        random_state: int = 0,
    ):
        """
        Perform k-means clustering across all Features objects using the specified embedding.
        Returns a dictionary of label Series (one per Features, keyed by name) and the centroids DataFrame.
        """

        # Build embeddings and keep names in sync
        embedding_dfs = {
            name: f.embedding_df(embedding_dict)
            for name, f in self.features_dict.items()
        }
        # Check all embeddings have the same columns
        columns = next(iter(embedding_dfs.values())).columns
        if not all(df.columns.equals(columns) for df in embedding_dfs.values()):
            raise ValueError("All embeddings must have the same columns")

        # Concatenate with keys to create a MultiIndex
        combined = pd.concat(embedding_dfs.values(), axis=0, keys=embedding_dfs.keys())
        valid_mask = combined.notna().all(axis=1)
        valid_combined = combined[valid_mask]

        # Fit kmeans only on valid rows
        model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
            valid_combined
        )
        centroids = pd.DataFrame(model.cluster_centers_, columns=combined.columns)

        # Assign cluster labels: nan for rows with any nan, cluster for valid rows
        combined_labels = pd.Series(np.nan, index=combined.index, name="cluster")
        combined_labels.loc[valid_mask] = model.labels_

        # Split back to per-object labels using the first level of the MultiIndex
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
        results_dict: dict[str, FeaturesResult],
        name: str = None,
        meta: dict = None,
        overwrite: bool = False,
    ):
        """
        Store all FeaturesResult objects in a one-layer dict (as returned by batch methods).
        Example:
            results = features_collection.speed('nose')
            features_collection.store(results)
        """
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

    def __getitem__(self, key):
        """
        Get Features by handle (str), by integer index, or by slice.
        """
        if isinstance(key, int):
            handle = list(self.features_dict)[key]
            return self.features_dict[handle]
        elif isinstance(key, slice):
            handles = list(self.features_dict)[key]
            return self.__class__({h: self.features_dict[h] for h in handles})
        else:
            return self.features_dict[key]

    def __setitem__(self, key, value):
        """
        Set Features by handle (str).
        """
        if not isinstance(value, Features):
            raise TypeError(f"Value must be a Features, got {type(value).__name__}")
        warnings.warn(
            "Direct assignment to FeaturesCollection is deprecated and may be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.features_dict[key] = value

    def keys(self):
        """Return the keys of the features_dict."""
        return self.features_dict.keys()

    def values(self):
        """Return the values of the features_dict."""
        return self.features_dict.values()

    def items(self):
        """Return the items of the features_dict."""
        return self.features_dict.items()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.features_dict)} Features objects>"
