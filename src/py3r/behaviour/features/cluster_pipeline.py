from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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
