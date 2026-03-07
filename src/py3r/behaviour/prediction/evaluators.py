from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from py3r.behaviour.util.series_utils import apply_normalization_to_df, normalize_df

WithinPolicy = Literal["leave_one_out"]
BetweenPolicy = Literal["pooled", "source_leave_one_out_ensemble"]


@dataclass(frozen=True)
class PredictionJob:
    """Single training/prediction job for cross-prediction orchestration."""

    train_group: str
    test_group: str
    train_handles: tuple[str, ...]
    test_handles: tuple[str, ...]
    tag: str
    aggregation_key: str


@dataclass(frozen=True)
class ComparisonPlan:
    """
    Declarative plan for within-vs-between prediction comparisons.

    The asymmetry concern (within uses many LOO predictors, between often one
    pooled predictor) is encoded by ``between_policy``.
    """

    within_policy: WithinPolicy
    between_policy: BetweenPolicy
    within_jobs: tuple[PredictionJob, ...]
    between_jobs: tuple[PredictionJob, ...]


def build_group_comparison_plan(
    *,
    group_to_handles: dict[str, list[str]],
    within_groups: list[str],
    between_pairs: list[tuple[str, str]],
    within_policy: WithinPolicy = "leave_one_out",
    between_policy: BetweenPolicy = "pooled",
) -> ComparisonPlan:
    """Build a deterministic job plan for grouped cross-prediction analysis."""
    if within_policy != "leave_one_out":
        raise NotImplementedError("Only leave_one_out within policy is currently supported.")

    within_jobs: list[PredictionJob] = []
    between_jobs: list[PredictionJob] = []

    for g in within_groups:
        handles = tuple(group_to_handles[g])
        if len(handles) < 2:
            continue
        for left_out in handles:
            train_handles = tuple(h for h in handles if h != left_out)
            within_jobs.append(
                PredictionJob(
                    train_group=g,
                    test_group=g,
                    train_handles=train_handles,
                    test_handles=(left_out,),
                    tag=f"within_{g}_loo_{left_out}",
                    aggregation_key=f"within::{g}::{left_out}",
                )
            )

    for from_group, to_group in between_pairs:
        source = tuple(group_to_handles[from_group])
        target = tuple(group_to_handles[to_group])
        if len(source) == 0 or len(target) == 0:
            continue

        if between_policy == "pooled":
            between_jobs.append(
                PredictionJob(
                    train_group=from_group,
                    test_group=to_group,
                    train_handles=source,
                    test_handles=target,
                    tag=f"between_{from_group}_to_{to_group}_pooled",
                    aggregation_key=f"between::{from_group}->{to_group}",
                )
            )
            continue

        if between_policy == "source_leave_one_out_ensemble":
            for left_out in source:
                train_handles = tuple(h for h in source if h != left_out)
                if len(train_handles) == 0:
                    continue
                between_jobs.append(
                    PredictionJob(
                        train_group=from_group,
                        test_group=to_group,
                        train_handles=train_handles,
                        test_handles=target,
                        tag=f"between_{from_group}_to_{to_group}_source_loo_{left_out}",
                        aggregation_key=f"between::{from_group}->{to_group}::{left_out}",
                    )
                )
            continue

        raise ValueError(f"Unknown between_policy: {between_policy}")

    return ComparisonPlan(
        within_policy=within_policy,
        between_policy=between_policy,
        within_jobs=tuple(within_jobs),
        between_jobs=tuple(between_jobs),
    )


class CrossGroupEvaluator:
    """
    Use-case runner for offline within-vs-between evaluation.

    This is a scaffold class: execution is intentionally deferred until we
    finalize predictor backend wiring and tidy-table schema.
    """

    def __init__(self, *, plan: ComparisonPlan):
        self.plan = plan

    @staticmethod
    def _rms_error(
        ground_truth: pd.DataFrame, prediction: pd.DataFrame, rescale: dict | str = None
    ) -> pd.Series:
        if not ground_truth.columns.equals(prediction.columns) or not ground_truth.index.equals(
            prediction.index
        ):
            raise ValueError("Input DataFrames must have the same columns and index")
        gt = ground_truth
        pred = prediction
        if rescale is not None:
            if rescale == "auto":
                gt, rescale_factors = normalize_df(gt)
                pred = apply_normalization_to_df(pred, rescale_factors)
            elif isinstance(rescale, dict):
                gt = apply_normalization_to_df(gt, rescale)
                pred = apply_normalization_to_df(pred, rescale)
            else:
                raise ValueError("rescale must be None, a dict, or 'auto'")
        diff = gt - pred
        rms = np.sqrt((diff**2).mean(axis=1))
        mask = gt.notna().all(axis=1) & pred.notna().all(axis=1)
        rms[~mask] = np.nan
        return rms

    @staticmethod
    def _fit_predict_rms(
        *,
        train_X: list[pd.DataFrame],
        train_y: list[pd.DataFrame],
        test_X: list[pd.DataFrame],
        test_y: list[pd.DataFrame],
        predictor_cls,
        predictor_kwargs: dict | None,
        normalize_source: bool,
        normalize_pred: dict | str | None,
    ) -> list[pd.Series]:
        if predictor_kwargs is None:
            predictor_kwargs = {}

        if normalize_source:
            train_X_concat = pd.concat(train_X, axis=0)
            train_X_norm, rescale_factors = normalize_df(train_X_concat)
            lengths = [len(x) for x in train_X]
            starts = np.cumsum([0] + lengths[:-1])
            train_X = [
                train_X_norm.iloc[start : start + length].copy()
                for start, length in zip(starts, lengths, strict=True)
            ]
            test_X = [apply_normalization_to_df(x.copy(), rescale_factors) for x in test_X]

        train_X_df = pd.concat(train_X, axis=0)
        train_y_df = pd.concat(train_y, axis=0)
        predictor = predictor_cls(**predictor_kwargs)
        predictor.fit(train_X_df, train_y_df)

        out: list[pd.Series] = []
        for x_df, y_df in zip(test_X, test_y, strict=True):
            pred_df = predictor.predict(x_df)
            pred_df = pred_df.reindex(index=y_df.index, columns=y_df.columns)
            out.append(CrossGroupEvaluator._rms_error(y_df, pred_df, rescale=normalize_pred))
        return out

    @staticmethod
    def summarize_handle_paired_errors(
        *,
        within_by_handle: dict[str, pd.Series],
        between_by_handle: dict[str, pd.Series],
        eps: float = 1e-6,
        group: str | None = None,
        comparison_key: str | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        handles = sorted(set(within_by_handle.keys()) & set(between_by_handle.keys()))
        rows = []
        for h in handles:
            w = within_by_handle[h].astype(float)
            b = between_by_handle[h].astype(float)
            aligned = pd.concat({"within": w, "between": b}, axis=1).dropna()
            if len(aligned) == 0:
                rows.append(
                    {
                        "handle": h,
                        "group": group,
                        "comparison_key": comparison_key,
                        "n_frames": 0,
                        "within_mean": np.nan,
                        "between_mean": np.nan,
                        "delta_mean": np.nan,
                        "ratio_mean": np.nan,
                        "log_ratio_mean": np.nan,
                    }
                )
                continue
            ratio = (aligned["between"] + eps) / (aligned["within"] + eps)
            log_ratio = np.log(ratio)
            rows.append(
                {
                    "handle": h,
                    "group": group,
                    "comparison_key": comparison_key,
                    "n_frames": int(len(aligned)),
                    "within_mean": float(aligned["within"].mean()),
                    "between_mean": float(aligned["between"].mean()),
                    "delta_mean": float((aligned["between"] - aligned["within"]).mean()),
                    "ratio_mean": float(ratio.mean()),
                    "log_ratio_mean": float(log_ratio.mean()),
                }
            )

        summary_df = pd.DataFrame(rows)
        if len(summary_df) == 0:
            return summary_df, {
                "n_handles": 0,
                "paired_t_stat": np.nan,
                "paired_t_p": np.nan,
                "wilcoxon_stat": np.nan,
                "wilcoxon_p": np.nan,
            }

        valid = summary_df[["within_mean", "between_mean"]].dropna()
        if len(valid) >= 2:
            t_res = ttest_rel(valid["between_mean"], valid["within_mean"], nan_policy="omit")
            try:
                w_res = wilcoxon(valid["between_mean"], valid["within_mean"], zero_method="wilcox")
                w_stat = float(w_res.statistic)
                w_p = float(w_res.pvalue)
            except Exception:
                w_stat = np.nan
                w_p = np.nan
            stats = {
                "n_handles": int(len(valid)),
                "paired_t_stat": float(t_res.statistic),
                "paired_t_p": float(t_res.pvalue),
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
            }
        else:
            stats = {
                "n_handles": int(len(valid)),
                "paired_t_stat": np.nan,
                "paired_t_p": np.nan,
                "wilcoxon_stat": np.nan,
                "wilcoxon_p": np.nan,
            }
        return summary_df, stats
