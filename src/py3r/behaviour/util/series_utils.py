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
    """generates a numeric encoder/decoder pair for categorical non-numeric data"""

    labels = list(set(s))
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

    encoder, decoder = gen_encoder_decoder(s)

    _ = pd.DataFrame()
    _["s"] = [encoder[i] for i in s]

    # count length of blocks of identical values
    x = (s != s.shift()).cumsum()
    y = s.groupby(x).count()
    _["blocklengths"] = [y.loc[i] for i in x]

    # replace blocks
    _["s"][_["blocklengths"] <= window] = np.nan
    _["s"].ffill(inplace=True)
    _["s"].bfill(inplace=True)
    output = pd.Series([decoder[i] for i in _["s"]])

    return output


def get_block(s: pd.Series, window: int) -> pd.Series:
    """
    drop labels that occur in blocks of less than window
    replace them with value from previous block in the series
    unless there is no previous block, in which case it fills
    from next block
    """

    encoder, decoder = gen_encoder_decoder(s)

    _ = pd.DataFrame()
    _["s"] = [encoder[i] for i in s]

    # count length of blocks of identical values
    x = (s != s.shift()).cumsum()
    y = s.groupby(x).count()
    _["blocklengths"] = [y.loc[i] for i in x]

    return _["blocklengths"] >= window


def remove_block(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """
    drop labels that occur in blocks where the second
    series is equal to True and
    replace them with value from previous block
    """

    mask = s1.to_numpy().astype(int)
    diffs = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for start, end in zip(starts, ends):
        if s2[start:end].to_numpy().any():
            if start > 0:
                replacement_value = s1[start - 1]
            else:
                replacement_value = s1[end + 1]
            s1[start:end] = replacement_value

    # Step 3: Assign back to DataFrame
    return s1


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
