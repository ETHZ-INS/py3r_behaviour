from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap_mean_delta(
    delta: np.ndarray | pd.Series,
    *,
    block_size: int,
    n_boot: int = 1000,
    random_state: int = 0,
) -> dict[str, float]:
    """
    Block bootstrap CI for mean(delta) under temporal autocorrelation.

    Parameters
    ----------
    delta
        1D frame-level paired contrast, e.g. ``err_between - err_within``.
    block_size
        Number of contiguous frames per resampled block.
    n_boot
        Number of bootstrap resamples.
    """
    x = np.asarray(delta, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    rng = np.random.default_rng(random_state)
    n = x.size
    starts = np.arange(0, max(1, n - block_size + 1))
    n_blocks = int(np.ceil(n / block_size))
    boot_means = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        picks = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([x[s : s + block_size] for s in picks])[:n]
        boot_means[i] = np.nanmean(sample)

    return {
        "mean": float(np.nanmean(x)),
        "ci_low": float(np.nanpercentile(boot_means, 2.5)),
        "ci_high": float(np.nanpercentile(boot_means, 97.5)),
    }


def permutation_test_recording_level(
    table: pd.DataFrame,
    *,
    value_col: str,
    recording_col: str,
    group_col: str,
    n_perm: int = 5000,
    random_state: int = 0,
) -> dict[str, float]:
    """
    Two-group permutation test on recording-level means.

    The test intentionally shuffles recording-level group labels (not frames)
    to respect temporal dependence within a recording.
    """
    req = {value_col, recording_col, group_col}
    missing = req - set(table.columns)
    if missing:
        raise ValueError(f"Missing columns for permutation test: {sorted(missing)}")

    rec = (
        table.groupby([recording_col, group_col], dropna=False)[value_col]
        .mean()
        .reset_index()
        .dropna(subset=[value_col, group_col])
    )
    groups = rec[group_col].unique().tolist()
    if len(groups) != 2:
        raise ValueError("permutation_test_recording_level requires exactly 2 groups")

    g0, g1 = groups
    vals = rec[value_col].to_numpy(dtype=np.float64)
    labels = rec[group_col].to_numpy()
    obs = float(vals[labels == g1].mean() - vals[labels == g0].mean())

    rng = np.random.default_rng(random_state)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        stat = vals[perm == g1].mean() - vals[perm == g0].mean()
        if abs(stat) >= abs(obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"observed_delta": obs, "p_value": float(p), "n_perm": int(n_perm)}
