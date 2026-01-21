from __future__ import annotations

import numpy as np
import pandas as pd


def fit_frame_imputer(df: pd.DataFrame) -> pd.Series:
    """
    Compute robust per-column medians (numeric-only) for imputation.
    Returns a pandas Series indexed by column name.
    """
    return df.median(numeric_only=True)


def impute_frame(
    df: pd.DataFrame, medians: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Impute NaNs using precomputed per-column medians.

    Returns:
      - imputed: DataFrame (same shape/index/columns)
      - observed_fraction: per-row ratio of non-NaN entries in the original df
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("impute_frame expects a pandas DataFrame")
    if not isinstance(medians, pd.Series):
        raise TypeError("medians must be a pandas Series indexed by column name")

    num_present_per_row = df.notna().sum(axis=1)
    observed_fraction = (num_present_per_row / max(df.shape[1], 1)).astype(np.float32)

    # Validate coverage: medians must cover all numeric columns of df
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing = [c for c in numeric_cols if c not in medians.index]
    if missing:
        raise ValueError(f"Imputer medians are missing required columns: {missing}")

    # fillna aligns by column name; non-numeric columns will be left unchanged
    imputed = df.fillna(medians)
    return imputed, observed_fraction
