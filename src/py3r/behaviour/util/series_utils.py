from math import floor, ceil
import pandas as pd
import numpy as np


def mode(series: pd.Series):
    return series.value_counts().index[0]


def rolling_apply(
    frame: pd.Series, window: int, func, center: bool = True
) -> pd.Series:
    """a custom rolling_apply that accepts non-numeric input"""
    if center:
        index = frame.index[ceil(window / 2) - 1 : -floor(window / 2)]
        values = [
            func(frame.iloc[i : i + window]) for i in range(len(frame) - window + 1)
        ]
    else:
        index = frame.index[window - 1 :]
        values = [
            func(frame.iloc[i : i + window]) for i in range(len(frame) - window + 1)
        ]

    return pd.Series(data=values, index=index).reindex(frame.index)


def gen_encoder_decoder(s: pd.Series):
    """generates a numeric encoder/decoder pair for categorical non-numeric data
    - Robust to missing values (np.nan, pd.NA, None): they are excluded from labels
    - Preserves first-occurrence order (stable) rather than using set()
    """
    # Drop missing and preserve order of first appearance
    non_missing = s[~s.isna()]
    labels = list(pd.unique(non_missing))
    encoding = list(np.arange(len(labels)))
    encoder = dict(zip(labels, encoding))
    decoder = dict(zip(encoding, labels))

    return encoder, decoder


def smooth_block(s: pd.Series, window: int) -> pd.Series:
    """
    drop labels that occur in blocks of less than window
    replace them with value from previous block in the series
    unless there is no previous block, in which case it fills
    from next block
    """
    # Build encoder on non-missing values; map missing to NaN code
    encoder, decoder = gen_encoder_decoder(s)

    # Numeric codes for known labels, NaN for missing/unseen
    codes = pd.Series([encoder.get(v, np.nan) for v in s], index=s.index, dtype="float64")

    # Compute block ids in an NA-safe way by using a sentinel for NaN
    sentinel = -1.0  # encoder indices start at 0, so -1 is safe as separator
    codes_filled = codes.fillna(sentinel)
    block_ids = (codes_filled != codes_filled.shift()).cumsum()

    # Count non-missing items per block (ignore NaNs for length)
    block_lengths = s.groupby(block_ids).transform("count")

    # Replace labels in short blocks (<= window) with NaN, then fill from neighbors
    new_codes = codes.copy()
    new_codes[block_lengths <= window] = np.nan
    new_codes.ffill(inplace=True)
    new_codes.bfill(inplace=True)

    # Decode back to original labels; keep NaN as missing if it remains
    output_vals = [
        (decoder[int(v)] if pd.notna(v) else np.nan) for v in new_codes.to_numpy()
    ]
    return pd.Series(output_vals, index=s.index, name=s.name)


def get_block(s: pd.Series, window: int) -> pd.Series:
    """
    drop labels that occur in blocks of less than window
    replace them with value from previous block in the series
    unless there is no previous block, in which case it fills
    from next block
    """
    # Robust computation of block lengths with NA-safe comparison
    encoder, _ = gen_encoder_decoder(s)
    codes = pd.Series([encoder.get(v, np.nan) for v in s], index=s.index, dtype="float64")
    sentinel = -1.0
    codes_filled = codes.fillna(sentinel)
    block_ids = (codes_filled != codes_filled.shift()).cumsum()
    block_lengths = s.groupby(block_ids).transform("count")
    # Only non-missing elements can be part of a "kept" block
    return (block_lengths >= window) & (~s.isna())


def remove_block(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """
    drop labels that occur in blocks where the second
    series is equal to True and
    replace them with value from previous block
    """

    # Identify contiguous True regions in s2; treat missing as False
    s1_out = s1.copy()
    mask_bool = s2.fillna(False).astype(bool).to_numpy()
    mask_int = mask_bool.astype(np.int8)
    diffs = np.diff(np.concatenate(([0], mask_int, [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for start, end in zip(starts, ends):
        # Only act if the region actually has any True (it always should by construction)
        if mask_bool[start:end].any():
            if start > 0:
                replacement_value = s1_out.iloc[start - 1]
            else:
                try:
                    replacement_value = s1_out.iloc[end]
                except IndexError:
                    raise IndexError(f"Index {end} out of range for pandas series s1")
            s1_out.iloc[start:end] = replacement_value

    # Step 3: Assign back to DataFrame
    return s1_out


def normalize_df(df: pd.DataFrame, z_score: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Normalize the columns of a DataFrame.
    If z_score is True, subtract mean and divide by std (z-score normalization).
    Returns the normalized DataFrame and a dict of the rescaling factors.
    If z_score is True, rescale_factors is {col: {'mean': mean, 'std': std}}.
    If z_score is False, rescale_factors is {col: std}.
    """
    if z_score:
        means = df.mean(axis=0)
        stds = df.std(axis=0, ddof=0)
        normalized = (df - means) / stds
        rescale_factors = {
            col: {"mean": means[col], "std": stds[col]} for col in df.columns
        }
    else:
        stds = df.std(axis=0, ddof=0)
        normalized = df / stds
        rescale_factors = stds.to_dict()
    return normalized, rescale_factors


def apply_normalization_to_df(df: pd.DataFrame, rescale_factors: dict) -> pd.DataFrame:
    """
    Apply normalization to a DataFrame using the provided rescale factors.
    Supports both std-only and mean+std (z-score) normalization.
    """
    normalized = df.copy()
    for col in df.columns:
        factor = rescale_factors[col]
        if isinstance(factor, dict):
            # z-score normalization
            normalized[col] = (df[col] - factor["mean"]) / factor["std"]
        else:
            # std-only normalization
            normalized[col] = df[col] / factor
    return normalized


def apply_custom_scaling(df: pd.DataFrame, scaling: dict[str, dict]) -> pd.DataFrame:
    """
    Apply custom per-column scaling based on substring matches.

    Rules:
    - Each key in `scaling` is matched against column names by substring containment.
    - For a matched column, apply (optional) normalization dividing by its std, then multiplying by `scale`.
    - If a column matches more than one key, raise ValueError.

    Example: apply_custom_scaling(df, {"accel": {"normalize": False, "scale": 3.0}, "dist": {"normalize": True, "scale": 1.0}})

    The input is not mutated; a scaled copy is returned.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("custom_scaling expects a DataFrame")
    if not isinstance(scaling, dict):
        raise TypeError("custom_scaling expects a dict for scaling configuration")

    df_scaled = df.copy()

    # Build column -> matching keys mapping
    col_matches: dict[str, list[str]] = {col: [] for col in df_scaled.columns}
    for key in scaling.keys():
        if not isinstance(key, str):
            raise TypeError("custom_scaling keys must be strings")
        for col in df_scaled.columns:
            if key in col:
                col_matches[col].append(key)

    # Enforce uniqueness
    conflicts = [col for col, keys in col_matches.items() if len(keys) > 1]
    if conflicts:
        raise ValueError(f"custom_scaling: columns match multiple keys: {conflicts}")

    # Apply scaling
    for col, keys in col_matches.items():
        if not keys:
            continue
        key = keys[0]
        cfg = scaling.get(key, {})
        if not isinstance(cfg, dict):
            raise TypeError(f"custom_scaling for key '{key}' must be a dict")

        do_norm = bool(cfg.get("normalize", False))
        scale = float(cfg.get("scale", 1.0))

        s = df_scaled[col].astype(float)
        if do_norm:
            std = float(np.nanstd(s.values, ddof=0))
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            s = s / std
        if scale != 1.0:
            s = s * scale
        df_scaled[col] = s

    return df_scaled
