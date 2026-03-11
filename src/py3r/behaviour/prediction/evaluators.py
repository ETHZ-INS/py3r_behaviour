from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from py3r.behaviour.util.series_utils import apply_normalization_to_df, normalize_df


class CrossGroupEvaluator:
    """Helper methods for cross-group prediction evaluation."""

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
