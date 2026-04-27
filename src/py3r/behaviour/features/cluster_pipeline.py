from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

from py3r.behaviour.features.centroids_df import CentroidsDf
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
        centroids_obj: CentroidsDf,
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
    normalize_details: dict[str, Literal["individual", "global", "none"]] | None = None
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
        centroids_obj: CentroidsDf,
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
            # impute_medians is read automatically from centroids_obj.scaling_recipe
            fr = feat.assign_clusters_by_centroids(centroids_obj)
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
        if cfg.auto_normalize:
            raise NotImplementedError("auto_normalize was removed; use normalize=True instead.")
        if cfg.rescale_factors is not None:
            raise NotImplementedError(
                "rescale_factors was removed; use normalize and/or feature_weights instead."
            )
        if cfg.custom_scaling is not None:
            raise NotImplementedError("custom_scaling was removed; use feature_weights instead.")

        build = self.pre.build(
            fc,
            embedding_dict,
            lowmem=cfg.lowmem,
            decimation_factor=cfg.decimation_factor,
        )

        using_new = (
            cfg.normalize or cfg.feature_weights is not None or cfg.normalize_details is not None
        )

        resolved_weights: dict[str, float] | None = None
        resolved_norm: dict[str, Literal["individual", "global", "none"]] = {}
        global_base_stds: dict[str, float] = {}

        if using_new:
            first_feat = next(_iter_features(fc))[2]
            embed_cols = list(first_feat.embedding_df(embedding_dict).columns)

            if cfg.feature_weights is not None:
                resolved_weights = _resolve_feature_weights(embed_cols, cfg.feature_weights)

            default_mode: Literal["global", "none"] = "global" if cfg.normalize else "none"
            resolved_norm = _resolve_column_modes(
                embed_cols, cfg.normalize_details, default=default_mode
            )

            bases_needed = {
                _base_feature_for_column(c, embedding_dict)
                for c, m in resolved_norm.items()
                if m == "global"
            }
            global_base_stds = _compute_global_base_stds(
                fc, embedding_dict, bases_needed=bases_needed
            )

            constant_factors = _constant_scaling_factors(
                embed_cols,
                embedding_dict,
                resolved_norm=resolved_norm,
                resolved_weights=resolved_weights,
                global_base_stds=global_base_stds,
            )
            norm = constant_factors

            has_individual = any(m == "individual" for m in resolved_norm.values())
            if has_individual:
                combined = _apply_per_features_scaling(
                    build.combined,
                    fc,
                    embedding_dict,
                    resolved_norm=resolved_norm,
                    resolved_weights=resolved_weights,
                    global_base_stds=global_base_stds,
                    flat_key=build.flat_group_key,
                )
            else:
                combined = build.combined
                if constant_factors is not None:
                    combined = combined * pd.Series(constant_factors)

        else:
            combined, norm = self.pre.scale(
                build.combined,
                auto_normalize=cfg.auto_normalize,
                rescale_factors=cfg.rescale_factors,
                custom_scaling=cfg.custom_scaling,
            )
            first_feat = next(_iter_features(fc))[2]
            embed_cols = list(first_feat.embedding_df(embedding_dict).columns)
            resolved_norm = {c: "none" for c in embed_cols}

        X, w, impute_medians, valid_mask = self.missing.prepare(combined, cfg.missing_policy)
        model, centroids_df = self.clusterer.fit(
            X, sample_weight=w, n_clusters=cfg.n_clusters, random_state=cfg.random_state
        )

        # Build recipe after impute_medians is known (it depends on missing_policy).
        if using_new:
            scaling_recipe = _build_scaling_recipe(
                embedding_dict, embed_cols, resolved_norm, constant_factors, impute_medians
            )
        else:
            scaling_recipe = _build_scaling_recipe(
                embedding_dict, embed_cols, resolved_norm, norm, impute_medians
            )
        centroids = CentroidsDf(df=centroids_df, scaling_recipe=scaling_recipe)

        meta = {
            "embedding_dict": embedding_dict,
            "n_clusters": cfg.n_clusters,
            "random_state": cfg.random_state,
            "normalize": cfg.normalize,
            "normalize_details": cfg.normalize_details,
            "feature_weights": cfg.feature_weights,
            "resolved_feature_weights": resolved_weights,
            "auto_normalize": cfg.auto_normalize,
            "rescale_factors": cfg.rescale_factors,
            "lowmem": cfg.lowmem,
            "decimation_factor": cfg.decimation_factor,
            "missing_policy": cfg.missing_policy,
            "impute_medians": None if impute_medians is None else impute_medians.to_dict(),
            "scaling_recipe": scaling_recipe,
        }

        if cfg.lowmem:
            # impute_medians is embedded in centroids.scaling_recipe
            result_dict = self.assigner.assign_lowmem(fc, centroids)
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
    normalize_details: dict[str, Literal["individual", "global", "none"]] | None = None
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


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _resolve_column_modes(
    columns: list[str] | pd.Index,
    normalize_details: dict[str, Literal["individual", "global", "none"]] | None,
    *,
    default: Literal["global", "none"],
) -> dict[str, Literal["individual", "global", "none"]]:
    """
    Resolve substring→mode rules into a per-column mode dict.

    Rules are matched by substring against full embedding column names.
    Raises ValueError if any rule matches no columns or any column matches
    multiple rules (overlap/ambiguity).
    """
    cols = list(columns)
    if not normalize_details:
        return {c: default for c in cols}

    matched_counts = {rule: 0 for rule in normalize_details}
    out: dict[str, Literal["individual", "global", "none"]] = {}
    for c in cols:
        matches = [rule for rule in normalize_details if rule in c]
        if len(matches) > 1:
            raise ValueError(f"normalize_details rules overlap for column '{c}': {matches}")
        if len(matches) == 0:
            out[c] = default
        else:
            rule = matches[0]
            matched_counts[rule] += 1
            out[c] = normalize_details[rule]

    unused = [r for r, n in matched_counts.items() if n == 0]
    if unused:
        raise ValueError(f"normalize_details rules matched no columns: {unused}")
    return out


def _safe_std(x: np.ndarray) -> float:
    """Std of finite values in *x*; returns 1.0 if empty or zero."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 1.0
    std = float(np.std(finite))
    return std if std > 0 else 1.0


def _compute_global_base_stds(
    fc,
    embedding_dict: dict[str, list[int]],
    *,
    bases_needed: set[str],
) -> dict[str, float]:
    """
    Compute global std for each base feature in *bases_needed*, pooled across
    the entire collection.  Returns an empty dict when *bases_needed* is empty.
    """
    if not bases_needed:
        return {}
    col_sum: dict[str, float] = {b: 0.0 for b in bases_needed}
    col_sq: dict[str, float] = {b: 0.0 for b in bases_needed}
    col_n: dict[str, int] = {b: 0 for b in bases_needed}
    for _gk, _fn, feat in _iter_features(fc):
        for base in bases_needed:
            vals = feat.data[base].to_numpy(dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            col_sum[base] += float(finite.sum())
            col_sq[base] += float((finite**2).sum())
            col_n[base] += int(finite.size)
    result: dict[str, float] = {}
    for base in bases_needed:
        n = col_n[base]
        if n == 0:
            result[base] = 1.0
        else:
            mean = col_sum[base] / n
            std = float(np.sqrt(max(0.0, col_sq[base] / n - mean**2)))
            result[base] = std if std > 0 else 1.0
    return result


def _scaling_for_features(
    feat: Features,
    embedding_dict: dict[str, list[int]],
    *,
    resolved_norm: dict[str, Literal["individual", "global", "none"]],
    resolved_weights: dict[str, float] | None,
    global_base_stds: dict[str, float],
) -> dict[str, float] | None:
    """
    Full per-embedding-column scaling for a single Features object.

    For "individual" columns, the std is computed from *feat* itself.
    For "global" columns, it uses pre-computed *global_base_stds*.
    For "none" columns, no normalisation factor is applied.
    Weights are always applied on top.

    Returns ``None`` when all factors are 1.0.
    """
    embed_cols = list(feat.embedding_df(embedding_dict).columns)
    factors: dict[str, float] = {}
    for col in embed_cols:
        base = _base_feature_for_column(col, embedding_dict)
        mode = resolved_norm.get(col, "none")
        if mode == "none":
            norm_f = 1.0
        elif mode == "global":
            denom = global_base_stds.get(base, 1.0)
            norm_f = 1.0 / denom if denom > 0 else 1.0
        elif mode == "individual":
            norm_f = 1.0 / _safe_std(feat.data[base].to_numpy(dtype=np.float64))
        else:
            raise ValueError(f"Unknown normalisation mode: {mode!r}")
        weight = resolved_weights.get(col, 1.0) if resolved_weights else 1.0
        factors[col] = norm_f * weight
    has_non_unity = any(v != 1.0 for v in factors.values())
    return factors if has_non_unity else None


def _constant_scaling_factors(
    embed_cols: list[str] | pd.Index,
    embedding_dict: dict[str, list[int]],
    *,
    resolved_norm: dict[str, Literal["individual", "global", "none"]],
    resolved_weights: dict[str, float] | None,
    global_base_stds: dict[str, float],
) -> dict[str, float] | None:
    """
    Constant per-embedding-column multipliers: weights plus any global norm factors.

    Individual-mode columns contribute only their weight (not their per-Features
    std, which varies).  This is the ``scaling_factors`` third return value.

    Returns ``None`` when all factors are 1.0.
    """
    factors: dict[str, float] = {}
    for col in list(embed_cols):
        base = _base_feature_for_column(col, embedding_dict)
        mode = resolved_norm.get(col, "none")
        if mode == "global":
            denom = global_base_stds.get(base, 1.0)
            norm_f = 1.0 / denom if denom > 0 else 1.0
        else:
            norm_f = 1.0
        weight = resolved_weights.get(col, 1.0) if resolved_weights else 1.0
        factors[col] = norm_f * weight
    has_non_unity = any(v != 1.0 for v in factors.values())
    return factors if has_non_unity else None


def _build_scaling_recipe(
    embedding_dict: dict[str, list[int]],
    columns: list[str],
    resolved_norm: dict[str, Literal["individual", "global", "none"]],
    constant_factors: dict[str, float] | None,
    impute_medians: pd.Series | None = None,
) -> dict:
    """
    Build the minimal scaling recipe stored inside a ``CentroidsDf``.

    Parameters
    ----------
    embedding_dict :
        The embedding used during clustering (for reproducing the embedding).
    columns :
        Actual embedding column names (sanity-check on load).
    resolved_norm :
        Per-column normalisation mode, already resolved from user rules.
    constant_factors :
        The ``scaling_factors`` constant multipliers (weights + global norms).
    impute_medians :
        Per-column fill values used during training when
        ``missing_policy="impute_weight"``.  Stored so future calls to
        ``assign_clusters_by_centroids`` can reproduce the same imputation.
        ``None`` when ``missing_policy="drop"``.
    """
    normalize_individual_base: dict[str, bool] = {}
    for base in embedding_dict:
        base_cols = [c for c in columns if c.startswith(base + "_t")]
        normalize_individual_base[base] = any(
            resolved_norm.get(c) == "individual" for c in base_cols
        )
    return {
        "version": 1,
        "embedding_dict": embedding_dict,
        "columns": columns,
        "normalize_individual_base": normalize_individual_base,
        "constant_factors": {} if constant_factors is None else dict(constant_factors),
        "impute_medians": None if impute_medians is None else impute_medians.to_dict(),
    }


def _apply_per_features_scaling(
    combined: pd.DataFrame,
    fc,
    embedding_dict: dict[str, list[int]],
    *,
    resolved_norm: dict[str, Literal["individual", "global", "none"]],
    resolved_weights: dict[str, float] | None,
    global_base_stds: dict[str, float],
    flat_key: str,
) -> pd.DataFrame:
    """
    Apply per-Features (including individual) scaling to the combined DataFrame.

    The combined DF is indexed by (group, feature, frame).  Each (group, feature)
    slice gets its own scaling dict computed by ``_scaling_for_features``.
    """
    result = combined.copy()
    for gkey, feat_name, feat in _iter_features(fc):
        per_sf = _scaling_for_features(
            feat,
            embedding_dict,
            resolved_norm=resolved_norm,
            resolved_weights=resolved_weights,
            global_base_stds=global_base_stds,
        )
        if per_sf is None:
            continue
        key0 = gkey if gkey is not None else flat_key
        idx = pd.IndexSlice[key0, feat_name, :]
        result.loc[idx, :] = result.loc[idx, :].mul(pd.Series(per_sf), axis=1).values
    return result


class StreamingClusteringPipeline:
    """
    Memory-friendly clustering via MiniBatchKMeans.partial_fit.

    Phase 1: Resolve normalisation modes and compute global stds where needed.
    Phase 2: n_epochs passes of partial_fit over fixed-size chunks, one Features
             at a time.  Per-Features scaling (including individual std) is applied
             chunk-by-chunk.
    Phase 3: Per-Features assignment using the fitted centroids.

    The returned ``scaling_factors`` (third element) is a dict of constant
    per-embedding-column multipliers — weights plus any global normalisation.
    For columns using "individual" normalisation the per-Features std is *not*
    included (it varies per recording); the ``CentroidsDf.scaling_recipe`` captures
    this so it can be reproduced on future datasets.
    """

    def run(
        self,
        fc,
        embedding_dict: dict[str, list[int]],
        cfg: StreamingConfig,
    ) -> tuple[dict, pd.DataFrame, dict[str, float] | None, dict]:
        first_feat = next(_iter_features(fc))[2]
        embed_cols = list(first_feat.embedding_df(embedding_dict).columns)

        resolved_weights: dict[str, float] | None = None
        if cfg.feature_weights is not None:
            resolved_weights = _resolve_feature_weights(embed_cols, cfg.feature_weights)

        default_mode: Literal["global", "none"] = "global" if cfg.normalize else "none"
        resolved_norm = _resolve_column_modes(
            embed_cols, cfg.normalize_details, default=default_mode
        )

        bases_needed = {
            _base_feature_for_column(c, embedding_dict)
            for c, m in resolved_norm.items()
            if m == "global"
        }
        global_base_stds = _compute_global_base_stds(fc, embedding_dict, bases_needed=bases_needed)

        constant_factors, impute_means = self._build_scaling(
            fc,
            embedding_dict,
            cfg,
            embed_cols=embed_cols,
            resolved_weights=resolved_weights,
            resolved_norm=resolved_norm,
            global_base_stds=global_base_stds,
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

                per_sf = _scaling_for_features(
                    feat,
                    embedding_dict,
                    resolved_norm=resolved_norm,
                    resolved_weights=resolved_weights,
                    global_base_stds=global_base_stds,
                )
                if per_sf is not None:
                    embed_df = embed_df * pd.Series(per_sf)
                columns = embed_df.columns
                for start in range(0, len(embed_df), cfg.chunk_size):
                    chunk = embed_df.iloc[start : start + cfg.chunk_size]
                    X, w = self._prepare_chunk(chunk, cfg.missing_policy, impute_means)
                    if len(X) == 0:
                        continue
                    model.partial_fit(X, sample_weight=w)

        centroids_df = pd.DataFrame(model.cluster_centers_, columns=columns)
        scaling_recipe = _build_scaling_recipe(
            embedding_dict, list(embed_cols), resolved_norm, constant_factors, impute_means
        )
        centroids = CentroidsDf(df=centroids_df, scaling_recipe=scaling_recipe)

        meta = {
            "function": "cluster_embedding_stream",
            "embedding_dict": embedding_dict,
            "n_clusters": cfg.n_clusters,
            "random_state": cfg.random_state,
            "normalize": cfg.normalize,
            "normalize_details": cfg.normalize_details,
            "feature_weights": cfg.feature_weights,
            "resolved_feature_weights": resolved_weights,
            "missing_policy": cfg.missing_policy,
            "chunk_size": cfg.chunk_size,
            "n_epochs": cfg.n_epochs,
            "batch_size": cfg.batch_size,
            "impute_means": (impute_means.to_dict() if impute_means is not None else None),
            "scaling_recipe": scaling_recipe,
        }

        result_dict = self._assign_all(fc, centroids, meta=meta)
        return result_dict, centroids, constant_factors, meta

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _prepare_chunk(chunk, missing_policy, impute_means):
        if missing_policy == "impute_weight" and impute_means is not None:
            X, w = impute_frame(chunk, impute_means)
            return X.values, w.values
        valid = chunk.notna().all(axis=1)
        return chunk[valid].values, None

    @staticmethod
    def _build_scaling(
        fc,
        embedding_dict: dict[str, list[int]],
        cfg: StreamingConfig,
        *,
        embed_cols: list[str],
        resolved_weights: dict[str, float] | None,
        resolved_norm: dict[str, Literal["individual", "global", "none"]],
        global_base_stds: dict[str, float],
    ) -> tuple[dict[str, float] | None, pd.Series | None]:
        """
        Compute constant scaling factors and (optionally) imputation means.

        The constant factors are weights plus any global normalisation — they
        do not include per-Features individual std (that is applied at fit/assign
        time via ``_scaling_for_features``).
        """
        constant_factors = _constant_scaling_factors(
            embed_cols,
            embedding_dict,
            resolved_norm=resolved_norm,
            resolved_weights=resolved_weights,
            global_base_stds=global_base_stds,
        )

        impute_means: pd.Series | None = None
        if cfg.missing_policy == "impute_weight":
            col_sum = np.zeros(len(embed_cols), dtype=np.float64)
            n_valid = 0
            for _gkey, _fname, feat in _iter_features(fc):
                embed_df = feat.embedding_df(embedding_dict).astype(np.float32)
                per_sf = _scaling_for_features(
                    feat,
                    embedding_dict,
                    resolved_norm=resolved_norm,
                    resolved_weights=resolved_weights,
                    global_base_stds=global_base_stds,
                )
                if per_sf is not None:
                    scale_arr = np.array([per_sf[c] for c in embed_cols], dtype=np.float64)
                    vals = embed_df.values * scale_arr
                else:
                    vals = embed_df.values.astype(np.float64)
                row_valid = np.isfinite(vals).all(axis=1)
                col_sum += vals[row_valid].sum(axis=0)
                n_valid += int(row_valid.sum())
            if n_valid > 0:
                impute_means = pd.Series(col_sum / n_valid, index=embed_cols)

        return constant_factors, impute_means

    @staticmethod
    def _assign_all(
        fc,
        centroids: CentroidsDf,
        *,
        meta: dict,
    ) -> dict:
        is_grouped = getattr(fc, "is_grouped", False)
        result_dict: dict = {}
        for gkey, feat_name, feat in _iter_features(fc):
            # impute_medians is read automatically from centroids.scaling_recipe
            fr = feat.assign_clusters_by_centroids(centroids)
            if is_grouped:
                result_dict.setdefault(gkey, {})[feat_name] = fr
            else:
                result_dict[feat_name] = fr
        return result_dict
