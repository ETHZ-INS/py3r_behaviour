from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

from py3r.behaviour.features.features import Features, FeaturesResult
from py3r.behaviour.util.missing_tolerance import fit_frame_imputer, impute_frame
from py3r.behaviour.util.series_utils import (
    apply_custom_scaling,
    apply_normalization_to_df,
    normalize_df,
)


class BuildResult(NamedTuple):
    combined: pd.DataFrame
    is_grouped: bool
    flat_group_key: str
    key_to_feature: dict[tuple[str, str], Features]


class Preprocessor(Protocol):
    def build(
        self,
        fc,
        embedding_dict: dict[str, list[int]],
        *,
        lowmem: bool,
        decimation_factor: int,
    ) -> BuildResult: ...

    def scale(
        self,
        df: pd.DataFrame,
        *,
        auto_normalize: bool,
        rescale_factors: dict | None,
        custom_scaling: dict[str, dict] | None,
    ) -> tuple[pd.DataFrame, dict | None]: ...


class MissingPolicy(Protocol):
    def prepare(
        self, df: pd.DataFrame, policy: Literal["drop", "impute_weight"]
    ) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None, pd.Series]: ...


class Clusterer(Protocol):
    def fit(
        self,
        X: pd.DataFrame,
        *,
        sample_weight: pd.Series | None,
        n_clusters: int,
        random_state: int,
    ) -> tuple[object, pd.DataFrame]: ...


class Assigner(Protocol):
    def assign_lowmem(
        self,
        fc,
        embedding_dict: dict[str, list[int]],
        centroids: pd.DataFrame,
        *,
        scaling_factors: dict | None,
        custom_scaling: dict[str, dict] | None,
        impute_medians: pd.Series | None,
    ) -> dict: ...

    def combined_labels(
        self,
        combined: pd.DataFrame,
        model,
        *,
        valid_mask: pd.Series,
        missing_policy: Literal["drop", "impute_weight"],
    ) -> pd.Series: ...


@dataclass(frozen=True)
class ClusteringConfig:
    n_clusters: int
    random_state: int = 0
    auto_normalize: bool = False
    rescale_factors: dict | None = None
    lowmem: bool = False
    decimation_factor: int = 10
    custom_scaling: dict[str, dict] | None = None
    missing_policy: Literal["drop", "impute_weight"] = "drop"


class DefaultPreprocessor:
    def build(
        self,
        fc,
        embedding_dict: dict[str, list[int]],
        *,
        lowmem: bool,
        decimation_factor: int,
    ) -> BuildResult:
        is_grouped = getattr(fc, "is_grouped", False)
        flat_group_key = "__flat__"
        group_iter = fc.items() if is_grouped else [(flat_group_key, fc)]
        all_embeddings: dict[tuple[str, str], pd.DataFrame] = {}
        key_to_feature: dict[tuple[str, str], Features] = {}
        for gkey, sub in group_iter:
            for feat_name, features in sub.features_dict.items():
                embed_df = features.embedding_df(embedding_dict).astype(np.float32)
                if lowmem:
                    embed_df = embed_df.iloc[::decimation_factor]
                key = (gkey, feat_name)
                all_embeddings[key] = embed_df
                key_to_feature[key] = features
        combined = pd.concat(
            all_embeddings.values(),
            keys=all_embeddings.keys(),
            names=["group", "feature", "frame"],
        )
        return BuildResult(combined, is_grouped, flat_group_key, key_to_feature)

    def scale(
        self,
        df: pd.DataFrame,
        *,
        auto_normalize: bool,
        rescale_factors: dict | None,
        custom_scaling: dict[str, dict] | None,
    ) -> tuple[pd.DataFrame, dict | None]:
        if custom_scaling is not None and (auto_normalize or rescale_factors is not None):
            raise ValueError(
                "custom_scaling is mutually exclusive with auto_normalize or rescale_factors"
            )
        normalization_factors = None
        if auto_normalize:
            df, normalization_factors = normalize_df(df)
        elif rescale_factors is not None:
            df = apply_normalization_to_df(df, rescale_factors)
        elif custom_scaling is not None:
            df = apply_custom_scaling(df, custom_scaling)
        return df, normalization_factors


class DefaultMissingPolicy:
    def prepare(
        self, df: pd.DataFrame, policy: Literal["drop", "impute_weight"]
    ) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None, pd.Series]:
        if policy == "impute_weight":
            medians = fit_frame_imputer(df)
            X_imp, sample_w = impute_frame(df, medians)
            valid_mask = pd.Series(True, index=df.index)
            return X_imp, sample_w, medians, valid_mask
        else:
            valid_mask = df.notna().all(axis=1)
            X = df[valid_mask]
            return X, None, None, valid_mask


class KMeansClusterer:
    def fit(
        self,
        X: pd.DataFrame,
        *,
        sample_weight: pd.Series | None,
        n_clusters: int,
        random_state: int,
    ) -> tuple[object, pd.DataFrame]:
        model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
            X, sample_weight=None if sample_weight is None else sample_weight.values
        )
        centroids = pd.DataFrame(model.cluster_centers_, columns=X.columns)
        return model, centroids


class DefaultAssigner:
    def assign_lowmem(
        self,
        fc,
        embedding_dict: dict[str, list[int]],
        centroids: pd.DataFrame,
        *,
        scaling_factors: dict | None,
        custom_scaling: dict[str, dict] | None,
        impute_medians: pd.Series | None,
    ) -> dict:
        # Build dict mirrors existing lowmem assignment in FeaturesCollection
        is_grouped = getattr(fc, "is_grouped", False)
        if is_grouped:
            result_dict = {}
            for gkey, sub in fc.items():
                group_map = {}
                for feat_name, feat in sub.features_dict.items():
                    fr = feat.assign_clusters_by_centroids(
                        embedding_dict,
                        centroids,
                        rescale_factors=scaling_factors,
                        custom_scaling=custom_scaling,
                        impute_medians=impute_medians,
                    )
                    group_map[feat_name] = fr
                result_dict[gkey] = group_map
            return result_dict
        else:
            result_dict = {}
            for feat_name, feat in fc.features_dict.items():
                fr = feat.assign_clusters_by_centroids(
                    embedding_dict,
                    centroids,
                    rescale_factors=scaling_factors,
                    custom_scaling=custom_scaling,
                    impute_medians=impute_medians,
                )
                result_dict[feat_name] = fr
            return result_dict

    def combined_labels(
        self,
        combined: pd.DataFrame,
        model,
        *,
        valid_mask: pd.Series,
        missing_policy: Literal["drop", "impute_weight"],
    ) -> pd.Series:
        if missing_policy == "impute_weight":
            return pd.Series(model.labels_, index=combined.index)
        labels = pd.Series(np.nan, index=combined.index)
        labels.loc[valid_mask] = model.labels_
        return labels


class ClusteringPipeline:
    def __init__(
        self,
        pre: Preprocessor | None = None,
        missing: MissingPolicy | None = None,
        clusterer: Clusterer | None = None,
        assigner: Assigner | None = None,
    ):
        self.pre = pre or DefaultPreprocessor()
        self.missing = missing or DefaultMissingPolicy()
        self.clusterer = clusterer or KMeansClusterer()
        self.assigner = assigner or DefaultAssigner()

    def run(
        self,
        fc,
        embedding_dict: dict[str, list[int]],
        cfg: ClusteringConfig,
    ) -> tuple[dict, pd.DataFrame, dict | None, dict]:
        build = self.pre.build(
            fc,
            embedding_dict,
            lowmem=cfg.lowmem,
            decimation_factor=cfg.decimation_factor,
        )
        combined, norm = self.pre.scale(
            build.combined,
            auto_normalize=cfg.auto_normalize,
            rescale_factors=cfg.rescale_factors,
            custom_scaling=cfg.custom_scaling,
        )
        X, w, impute_medians, valid_mask = self.missing.prepare(combined, cfg.missing_policy)
        model, centroids = self.clusterer.fit(
            X, sample_weight=w, n_clusters=cfg.n_clusters, random_state=cfg.random_state
        )

        meta = {
            "embedding_dict": embedding_dict,
            "n_clusters": cfg.n_clusters,
            "random_state": cfg.random_state,
            "auto_normalize": cfg.auto_normalize,
            "rescale_factors": cfg.rescale_factors,
            "lowmem": cfg.lowmem,
            "decimation_factor": cfg.decimation_factor,
            "missing_policy": cfg.missing_policy,
            "impute_medians": None if impute_medians is None else impute_medians.to_dict(),
        }

        if cfg.lowmem:
            factors_for_assign = norm if cfg.auto_normalize else cfg.rescale_factors
            result_dict = self.assigner.assign_lowmem(
                fc,
                embedding_dict,
                centroids,
                scaling_factors=factors_for_assign,
                custom_scaling=cfg.custom_scaling,
                impute_medians=impute_medians,
            )
            return result_dict, centroids, norm, meta

        # Non-lowmem: reconstruct per-feature FeaturesResult from combined_labels
        combined_labels = self.assigner.combined_labels(
            combined, model, valid_mask=valid_mask, missing_policy=cfg.missing_policy
        )
        if build.is_grouped:
            result_dict = {}
            for (gkey, feat_name), feat in build.key_to_feature.items():
                labels = combined_labels.xs((gkey, feat_name), level=["group", "feature"]).astype(
                    "Int64"
                )
                result_dict.setdefault(gkey, {})[feat_name] = FeaturesResult(
                    labels, feat, f"kmeans_{cfg.n_clusters}", meta
                )
        else:
            result_dict = {}
            flat = build.flat_group_key
            for (_, feat_name), feat in build.key_to_feature.items():
                labels = combined_labels.xs((flat, feat_name), level=["group", "feature"]).astype(
                    "Int64"
                )
                result_dict[feat_name] = FeaturesResult(
                    labels, feat, f"kmeans_{cfg.n_clusters}", meta
                )
        return result_dict, centroids, norm, meta


# ---------------------------------------------------------------------------
# Streaming (MiniBatchKMeans + partial_fit) pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamingConfig:
    n_clusters: int
    random_state: int = 0
    auto_normalize: bool = False
    rescale_factors: dict | None = None
    custom_scaling: dict[str, dict] | None = None
    missing_policy: Literal["drop", "impute_weight"] = "drop"
    chunk_size: int = 10_000
    n_epochs: int = 3
    batch_size: int = 1024


def _iter_features(fc):
    """Yield (group_key, feat_name, Features) for flat or grouped collections."""
    is_grouped = getattr(fc, "is_grouped", False)
    if is_grouped:
        for gkey, sub in fc.items():
            for feat_name, feat in sub.features_dict.items():
                yield gkey, feat_name, feat
    else:
        for feat_name, feat in fc.features_dict.items():
            yield None, feat_name, feat


class StreamingClusteringPipeline:
    """
    Memory-friendly clustering via MiniBatchKMeans.partial_fit.

    Phase 1 (optional): streaming pass to compute normalisation factors.
    Phase 2: n_epochs passes of partial_fit over fixed-size chunks.
    Phase 3: per-Features assignment using the fitted centroids.

    Never builds a combined DataFrame — only one chunk is in memory at a time.
    """

    def run(
        self,
        fc,
        embedding_dict: dict[str, list[int]],
        cfg: StreamingConfig,
    ) -> tuple[dict, pd.DataFrame, dict | None, dict]:
        scaling_factors, impute_means = self._streaming_stats(fc, embedding_dict, cfg)

        model = MiniBatchKMeans(
            n_clusters=cfg.n_clusters,
            random_state=cfg.random_state,
            batch_size=cfg.batch_size,
        )

        columns = None
        for _epoch in range(cfg.n_epochs):
            for _gkey, _fname, feat in _iter_features(fc):
                embed_df = feat.embedding_df(embedding_dict).astype(np.float32)
                embed_df = self._apply_scaling(embed_df, scaling_factors, cfg)
                columns = embed_df.columns
                for start in range(0, len(embed_df), cfg.chunk_size):
                    chunk = embed_df.iloc[start : start + cfg.chunk_size]
                    X, w = self._prepare_chunk(chunk, cfg.missing_policy, impute_means)
                    if len(X) == 0:
                        continue
                    model.partial_fit(X, sample_weight=w)

        centroids = pd.DataFrame(model.cluster_centers_, columns=columns)

        meta = {
            "function": "cluster_embedding_stream",
            "embedding_dict": embedding_dict,
            "n_clusters": cfg.n_clusters,
            "random_state": cfg.random_state,
            "auto_normalize": cfg.auto_normalize,
            "rescale_factors": cfg.rescale_factors,
            "custom_scaling": cfg.custom_scaling,
            "missing_policy": cfg.missing_policy,
            "chunk_size": cfg.chunk_size,
            "n_epochs": cfg.n_epochs,
            "batch_size": cfg.batch_size,
            "impute_means": (impute_means.to_dict() if impute_means is not None else None),
        }

        result_dict = self._assign_all(
            fc, embedding_dict, centroids, cfg, scaling_factors, impute_means, meta
        )
        return result_dict, centroids, scaling_factors, meta

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _apply_scaling(df, scaling_factors, cfg):
        if scaling_factors is not None:
            return apply_normalization_to_df(df, scaling_factors)
        if cfg.custom_scaling is not None:
            return apply_custom_scaling(df, cfg.custom_scaling)
        return df

    @staticmethod
    def _prepare_chunk(chunk, missing_policy, impute_means):
        if missing_policy == "impute_weight" and impute_means is not None:
            X, w = impute_frame(chunk, impute_means)
            return X.values, w.values
        valid = chunk.notna().all(axis=1)
        return chunk[valid].values, None

    def _streaming_stats(self, fc, embedding_dict, cfg):
        """
        Single streaming pass to compute global column stats.

        Returns (scaling_factors, impute_means) where either may be None
        depending on cfg.  When auto_normalize is True, scaling_factors is
        a dict of per-column stds.  When missing_policy is "impute_weight",
        impute_means is a Series of per-column means (computed after scaling).
        """
        if cfg.custom_scaling is not None and (
            cfg.auto_normalize or cfg.rescale_factors is not None
        ):
            raise ValueError(
                "custom_scaling is mutually exclusive with auto_normalize or rescale_factors"
            )

        need_norm = cfg.auto_normalize and cfg.rescale_factors is None
        need_impute = cfg.missing_policy == "impute_weight"

        if cfg.rescale_factors is not None:
            scaling_factors = cfg.rescale_factors
        elif not need_norm:
            scaling_factors = None
        else:
            scaling_factors = None  # computed below

        if not need_norm and not need_impute:
            return scaling_factors, None

        # One pass over all embeddings to accumulate running sums
        col_sum = None
        col_sq_sum = None
        n_valid = 0
        for _gkey, _fname, feat in _iter_features(fc):
            embed_df = feat.embedding_df(embedding_dict).astype(np.float32)
            if not need_norm:
                embed_df = self._apply_scaling(embed_df, scaling_factors, cfg)
            valid = embed_df.notna().all(axis=1)
            vals = embed_df[valid].values
            if col_sum is None:
                col_sum = np.zeros(vals.shape[1], dtype=np.float64)
                col_sq_sum = np.zeros(vals.shape[1], dtype=np.float64)
            col_sum += vals.sum(axis=0)
            col_sq_sum += (vals**2).sum(axis=0)
            n_valid += len(vals)

        if n_valid == 0:
            raise ValueError("No valid rows found in embedding data")

        ref_cols = feat.embedding_df(embedding_dict).columns
        means = col_sum / n_valid

        if need_norm:
            stds = np.sqrt(col_sq_sum / n_valid - means**2)
            stds[stds == 0] = 1.0
            scaling_factors = dict(zip(ref_cols, stds, strict=True))

        impute_means = None
        if need_impute:
            if need_norm:
                # Means were computed on raw data; re-derive after scaling
                impute_means = pd.Series(
                    means / np.array(list(scaling_factors.values())), index=ref_cols
                )
            else:
                impute_means = pd.Series(means, index=ref_cols)

        return scaling_factors, impute_means

    @staticmethod
    def _assign_all(fc, embedding_dict, centroids, cfg, scaling_factors, impute_means, meta):
        is_grouped = getattr(fc, "is_grouped", False)
        result_dict = {}
        rescale_for_assign = scaling_factors if cfg.custom_scaling is None else None
        for gkey, feat_name, feat in _iter_features(fc):
            fr = feat.assign_clusters_by_centroids(
                embedding_dict,
                centroids,
                rescale_factors=rescale_for_assign,
                custom_scaling=cfg.custom_scaling,
                impute_medians=impute_means,
            )
            if is_grouped:
                result_dict.setdefault(gkey, {})[feat_name] = fr
            else:
                result_dict[feat_name] = fr
        return result_dict
