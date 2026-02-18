from __future__ import annotations

import warnings
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
    build_column_weights,
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
    normalize: bool = False
    feature_weights: dict[str, float] | None = None
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
        is_grouped = getattr(fc, "is_grouped", False)
        result_dict = {}
        items = (
            (
                (gkey, feat_name, feat)
                for gkey, sub in fc.items()
                for feat_name, feat in sub.features_dict.items()
            )
            if is_grouped
            else ((None, fn, f) for fn, f in fc.features_dict.items())
        )
        for gkey, feat_name, feat in items:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                fr = feat.assign_clusters_by_centroids(
                    embedding_dict,
                    centroids,
                    rescale_factors=scaling_factors,
                    custom_scaling=custom_scaling,
                    impute_medians=impute_medians,
                )
            if is_grouped:
                result_dict.setdefault(gkey, {})[feat_name] = fr
            else:
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
        using_legacy = (
            cfg.auto_normalize or cfg.rescale_factors is not None or cfg.custom_scaling is not None
        )
        using_new = cfg.normalize or cfg.feature_weights is not None
        if using_legacy and using_new:
            raise ValueError(
                "Cannot mix new params (normalize, feature_weights) with "
                "deprecated params (auto_normalize, rescale_factors, custom_scaling)."
            )
        if cfg.auto_normalize:
            warnings.warn(
                "auto_normalize is deprecated; use normalize=True instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if cfg.rescale_factors is not None:
            warnings.warn(
                "rescale_factors is deprecated; use normalize and/or feature_weights instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if cfg.custom_scaling is not None:
            warnings.warn(
                "custom_scaling is deprecated; use feature_weights instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        build = self.pre.build(
            fc,
            embedding_dict,
            lowmem=cfg.lowmem,
            decimation_factor=cfg.decimation_factor,
        )
        resolved_weights = None
        if using_new:
            if cfg.feature_weights is not None:
                resolved_weights = _resolve_feature_weights(
                    build.combined.columns, cfg.feature_weights
                )
            norm = _compute_scaling_factors(
                fc,
                embedding_dict,
                normalize=cfg.normalize,
                resolved_weights=resolved_weights,
            )
            combined = build.combined
            if norm is not None:
                combined = combined * pd.Series(norm)
        else:
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
            "normalize": cfg.normalize,
            "feature_weights": cfg.feature_weights,
            "resolved_feature_weights": resolved_weights,
            "auto_normalize": cfg.auto_normalize,
            "rescale_factors": cfg.rescale_factors,
            "lowmem": cfg.lowmem,
            "decimation_factor": cfg.decimation_factor,
            "missing_policy": cfg.missing_policy,
            "impute_medians": None if impute_medians is None else impute_medians.to_dict(),
        }

        if cfg.lowmem:
            if using_new:
                is_grouped = getattr(fc, "is_grouped", False)
                result_dict = {}
                for gkey, feat_name, feat in _iter_features(fc):
                    fr = feat.assign_clusters_by_centroids(
                        embedding_dict,
                        centroids,
                        scaling_factors=norm,
                        impute_medians=impute_medians,
                    )
                    if is_grouped:
                        result_dict.setdefault(gkey, {})[feat_name] = fr
                    else:
                        result_dict[feat_name] = fr
            else:
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
    normalize: bool = False
    feature_weights: dict[str, float] | None = None
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


def _resolve_feature_weights(
    columns: list[str] | pd.Index,
    feature_weights: dict[str, float],
) -> dict[str, float]:
    """
    Resolve substring *feature_weights* into per-column weights and print the mapping.

    *columns* must be the real embedding column names obtained from
    ``feat.embedding_df(embedding_dict).columns``.

    Raises ValueError (via build_column_weights) if any rule matches no column
    or any column matches multiple rules.
    """
    weights = build_column_weights(columns, feature_weights)

    for rule, w in feature_weights.items():
        matched = [c for c in columns if rule in c]
        print(f"  feature_weights: {rule!r} → {matched} × {w:.4g}")
    unmatched = [c for c in columns if weights[c] == 1.0]
    if unmatched:
        print(f"  feature_weights: unmatched columns (weight 1.0): {unmatched}")

    return weights


def _base_feature_for_column(col: str, embedding_dict: dict[str, list[int]]) -> str:
    """Map an embedding column name back to its base feature name."""
    for base in embedding_dict:
        if col.startswith(base + "_t"):
            return base
    raise ValueError(f"Cannot determine base feature for embedding column '{col}'")


def _compute_scaling_factors(
    fc,
    embedding_dict: dict[str, list[int]],
    *,
    normalize: bool,
    resolved_weights: dict[str, float] | None,
) -> dict[str, float] | None:
    """Compute combined per-embedding-column scaling factors."""
    if not normalize and not resolved_weights:
        return None

    base_stds: dict[str, float] | None = None
    if normalize:
        col_sum: dict[str, float] = {}
        col_sq: dict[str, float] = {}
        col_n: dict[str, int] = {}
        for _gkey, _fname, feat in _iter_features(fc):
            for base_col in embedding_dict:
                vals = feat.data[base_col].to_numpy(dtype=np.float64)
                finite = vals[np.isfinite(vals)]
                if base_col not in col_sum:
                    col_sum[base_col] = 0.0
                    col_sq[base_col] = 0.0
                    col_n[base_col] = 0
                col_sum[base_col] += finite.sum()
                col_sq[base_col] += (finite**2).sum()
                col_n[base_col] += len(finite)
        base_stds = {}
        for base_col in embedding_dict:
            n = col_n[base_col]
            if n == 0:
                base_stds[base_col] = 1.0
                continue
            mean = col_sum[base_col] / n
            std = float(np.sqrt(col_sq[base_col] / n - mean**2))
            base_stds[base_col] = std if std > 0 else 1.0

    first_feat = next(_iter_features(fc))[2]
    embed_cols = first_feat.embedding_df(embedding_dict).columns

    factors: dict[str, float] = {}
    for col in embed_cols:
        base = _base_feature_for_column(col, embedding_dict)
        norm_factor = 1.0 / base_stds[base] if base_stds is not None else 1.0
        weight = resolved_weights.get(col, 1.0) if resolved_weights else 1.0
        factors[col] = norm_factor * weight

    has_non_unity = any(v != 1.0 for v in factors.values())
    return factors if has_non_unity else None


class StreamingClusteringPipeline:
    """
    Memory-friendly clustering via MiniBatchKMeans.partial_fit.

    Phase 1 (optional): streaming pass to compute per-base-feature stds.
    Phase 2: n_epochs passes of partial_fit over fixed-size chunks.
    Phase 3: per-Features assignment using the fitted centroids.

    Normalisation is computed on base feature columns (before embedding)
    so all time-shifts of the same feature share the same std.

    The returned scaling_factors dict contains one float per embedding
    column — the combined effect of normalisation and column weights.
    Multiply raw embedding columns by these values to reproduce the
    transform.
    """

    def run(
        self,
        fc,
        embedding_dict: dict[str, list[int]],
        cfg: StreamingConfig,
    ) -> tuple[dict, pd.DataFrame, dict[str, float] | None, dict]:
        resolved_weights = None
        if cfg.feature_weights is not None:
            first_feat = next(_iter_features(fc))[2]
            first_cols = first_feat.embedding_df(embedding_dict).columns
            resolved_weights = _resolve_feature_weights(first_cols, cfg.feature_weights)

        base_stds = self._compute_base_stds(fc, embedding_dict, cfg) if cfg.normalize else None
        scaling_factors, impute_means = self._build_scaling(
            fc, embedding_dict, cfg, resolved_weights, base_stds
        )

        model = MiniBatchKMeans(
            n_clusters=cfg.n_clusters,
            random_state=cfg.random_state,
            batch_size=cfg.batch_size,
        )

        columns = None
        for _epoch in range(cfg.n_epochs):
            for _gkey, _fname, feat in _iter_features(fc):
                embed_df = feat.embedding_df(embedding_dict).astype(np.float32)

                if cfg.feature_weights is not None and _epoch == 0:
                    check = build_column_weights(embed_df.columns, cfg.feature_weights)
                    if check != resolved_weights:
                        raise ValueError(
                            f"feature_weights resolved differently for '{_fname}': "
                            f"expected {resolved_weights}, got {check}"
                        )

                if scaling_factors is not None:
                    embed_df = embed_df * pd.Series(scaling_factors)
                columns = embed_df.columns
                for start in range(0, len(embed_df), cfg.chunk_size):
                    chunk = embed_df.iloc[start : start + cfg.chunk_size]
                    X, w = self._prepare_chunk(
                        chunk,
                        cfg.missing_policy,
                        impute_means,
                    )
                    if len(X) == 0:
                        continue
                    model.partial_fit(X, sample_weight=w)

        centroids = pd.DataFrame(model.cluster_centers_, columns=columns)

        meta = {
            "function": "cluster_embedding_stream",
            "embedding_dict": embedding_dict,
            "n_clusters": cfg.n_clusters,
            "random_state": cfg.random_state,
            "normalize": cfg.normalize,
            "feature_weights": cfg.feature_weights,
            "resolved_feature_weights": resolved_weights,
            "missing_policy": cfg.missing_policy,
            "chunk_size": cfg.chunk_size,
            "n_epochs": cfg.n_epochs,
            "batch_size": cfg.batch_size,
            "impute_means": (impute_means.to_dict() if impute_means is not None else None),
        }

        result_dict = self._assign_all(
            fc,
            embedding_dict,
            centroids,
            scaling_factors,
            impute_means,
            meta,
        )
        return result_dict, centroids, scaling_factors, meta

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _prepare_chunk(chunk, missing_policy, impute_means):
        if missing_policy == "impute_weight" and impute_means is not None:
            X, w = impute_frame(chunk, impute_means)
            return X.values, w.values
        valid = chunk.notna().all(axis=1)
        return chunk[valid].values, None

    @staticmethod
    def _compute_base_stds(fc, embedding_dict, cfg):
        """Streaming pass over base feature columns to get per-feature stds."""
        col_sum: dict[str, float] = {}
        col_sq_sum: dict[str, float] = {}
        col_n: dict[str, int] = {}
        for _gkey, _fname, feat in _iter_features(fc):
            for base_col in embedding_dict:
                vals = feat.data[base_col].to_numpy(dtype=np.float64)
                finite = vals[np.isfinite(vals)]
                if base_col not in col_sum:
                    col_sum[base_col] = 0.0
                    col_sq_sum[base_col] = 0.0
                    col_n[base_col] = 0
                col_sum[base_col] += finite.sum()
                col_sq_sum[base_col] += (finite**2).sum()
                col_n[base_col] += len(finite)

        base_stds: dict[str, float] = {}
        for base_col in embedding_dict:
            n = col_n[base_col]
            if n == 0:
                base_stds[base_col] = 1.0
                continue
            mean = col_sum[base_col] / n
            std = np.sqrt(col_sq_sum[base_col] / n - mean**2)
            base_stds[base_col] = std if std > 0 else 1.0
        return base_stds

    @staticmethod
    def _build_scaling(fc, embedding_dict, cfg, resolved_weights, base_stds):
        """
        Build the combined per-embedding-column scaling factors and impute means.

        Each factor is: (1 / base_std if normalize else 1) * (column_weight or 1).
        Multiply raw embedding columns by these to reproduce the transform.
        """
        first_feat = next(_iter_features(fc))[2]
        embed_cols = first_feat.embedding_df(embedding_dict).columns

        factors: dict[str, float] = {}
        for col in embed_cols:
            base = _base_feature_for_column(col, embedding_dict)
            norm_factor = 1.0 / base_stds[base] if base_stds is not None else 1.0
            weight = resolved_weights.get(col, 1.0) if resolved_weights else 1.0
            factors[col] = norm_factor * weight

        has_non_unity = any(v != 1.0 for v in factors.values())
        scaling_factors = factors if has_non_unity else None

        impute_means = None
        if cfg.missing_policy == "impute_weight":
            # Streaming means on the scaled embedding
            col_sum = np.zeros(len(embed_cols), dtype=np.float64)
            n_valid = 0
            scale_arr = np.array([factors[c] for c in embed_cols], dtype=np.float64)
            for _gkey, _fname, feat in _iter_features(fc):
                embed_df = feat.embedding_df(embedding_dict).astype(np.float32)
                vals = embed_df.values * scale_arr
                row_valid = np.isfinite(vals).all(axis=1)
                col_sum += vals[row_valid].sum(axis=0)
                n_valid += row_valid.sum()
            if n_valid > 0:
                impute_means = pd.Series(col_sum / n_valid, index=embed_cols)

        return scaling_factors, impute_means

    @staticmethod
    def _assign_all(fc, embedding_dict, centroids, scaling_factors, impute_means, meta):
        is_grouped = getattr(fc, "is_grouped", False)
        result_dict = {}
        for gkey, feat_name, feat in _iter_features(fc):
            fr = feat.assign_clusters_by_centroids(
                embedding_dict,
                centroids,
                scaling_factors=scaling_factors,
                impute_medians=impute_means,
            )
            if is_grouped:
                result_dict.setdefault(gkey, {})[feat_name] = fr
            else:
                result_dict[feat_name] = fr
        return result_dict
