import warnings
from math import ceil, floor
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype


def mode(series: pd.Series):
    return series.value_counts().index[0]


def rolling_apply(frame: pd.Series, window: int, func, center: bool = True) -> pd.Series:
    """a custom rolling_apply that accepts non-numeric input"""
    if center:
        index = frame.index[ceil(window / 2) - 1 : -floor(window / 2)]
        values = [func(frame.iloc[i : i + window]) for i in range(len(frame) - window + 1)]
    else:
        index = frame.index[window - 1 :]
        values = [func(frame.iloc[i : i + window]) for i in range(len(frame) - window + 1)]

    return pd.Series(data=values, index=index).reindex(frame.index)


def gen_encoder_decoder(s: pd.Series):
    """generates a numeric encoder/decoder pair for categorical non-numeric data"""

    labels = list(set(s))
    encoding = list(np.arange(len(labels)))
    encoder = dict(zip(labels, encoding, strict=True))
    decoder = dict(zip(encoding, labels, strict=True))

    return encoder, decoder


def smooth_block(s: pd.Series, window: int) -> pd.Series:
    """
    deprecated: use block_filter and block_fill instead

    drop labels that occur in blocks of less than window
    replace them with value from previous block in the series
    unless there is no previous block, in which case it fills
    from next block
    """

    warnings.warn(
        "smooth_block is deprecated, use block_filter and block_fill instead",
        DeprecationWarning,
        stacklevel=2,
    )

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


def block_filter(s: pd.Series, min_block: int = 2) -> pd.Series:
    """
    Mark short observed categorical blocks as np.nan.

    Existing missing values are preserved. Only contiguous non-missing runs
    of identical labels are considered "blocks".

    Parameters
    ----------
    s : pd.Series
        Input categorical series.
    min_block : int, default 2
        Minimum block length to keep. Blocks shorter than this are replaced
        with ``np.nan``.
    """
    if not isinstance(min_block, int) or min_block < 1:
        raise ValueError("min_block must be an integer >= 1")

    out = s.copy()
    observed = s.notna()
    # New run starts when we hit observed data after a gap, or when label changes.
    run_starts = observed & ((~observed.shift(fill_value=False)) | s.ne(s.shift()))
    run_ids = run_starts.cumsum().where(observed)
    run_lengths = run_ids.groupby(run_ids).transform("size")
    out.loc[observed & (run_lengths < min_block)] = np.nan

    return out


def block_fill(
    s: pd.Series,
    *,
    max_gap: int = 1,
    direction: Literal["forward", "backward", "both"] = "both",
    require_same_label: bool = True,
) -> pd.Series:
    """
    Fill short missing runs in categorical data using local neighbors only.

    Parameters
    ----------
    s : pd.Series
        Input categorical series.
    max_gap : int, default 1
        Maximum length of a missing run to fill. Longer runs are left missing.
    direction : {"forward", "backward", "both"}, default "both"
        Neighbor direction used for fill.
    require_same_label : bool, default True
        Only applies when ``direction="both"``. If True, a gap is filled only
        when both bracketing labels exist and are equal.
    """
    if not isinstance(max_gap, int) or max_gap < 0:
        raise ValueError("max_gap must be an integer >= 0")
    if direction not in {"forward", "backward", "both"}:
        raise ValueError("direction must be one of: 'forward', 'backward', 'both'")
    if max_gap == 0:
        return s.copy()

    out = s.copy()
    na_mask = s.isna().to_numpy()
    edges = np.diff(np.concatenate(([0], na_mask.astype(np.int8), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    for start, end in zip(starts, ends, strict=True):
        gap_len = end - start
        if gap_len > max_gap:
            continue

        left_exists = start > 0 and pd.notna(out.iloc[start - 1])
        right_exists = end < len(out) and pd.notna(out.iloc[end])

        fill_value = None
        if direction == "forward":
            if left_exists:
                fill_value = out.iloc[start - 1]
        elif direction == "backward":
            if right_exists:
                fill_value = out.iloc[end]
        else:
            if require_same_label:
                if left_exists and right_exists and out.iloc[start - 1] == out.iloc[end]:
                    fill_value = out.iloc[start - 1]
            else:
                if left_exists:
                    fill_value = out.iloc[start - 1]
                elif right_exists:
                    fill_value = out.iloc[end]

        if fill_value is not None:
            out.iloc[start:end] = fill_value

    return out


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

    mask = s1.astype("Int64").to_numpy()
    diffs = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for start, end in zip(starts, ends, strict=True):
        if s2[start:end].to_numpy().any():
            if start > 0:
                replacement_value = s1.iloc[start - 1]
            else:
                try:
                    replacement_value = s1.iloc[end]
                except IndexError as e:
                    raise IndexError(f"Index {end} out of range for pandas series s1") from e
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
        rescale_factors = {col: {"mean": means[col], "std": stds[col]} for col in df.columns}
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
    - For a matched column, apply (optional) normalization dividing by
      its std, then multiplying by `scale`.
    - If a column matches more than one key, raise ValueError.

    Example: apply_custom_scaling(df, {"accel": {"normalize": False,
      "scale": 3.0}, "dist": {"normalize": True, "scale": 1.0}})

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


def build_column_weights(
    columns: list[str] | pd.Index, rules: dict[str, float]
) -> dict[str, float]:
    """
    Build a per-column weight dict from substring-matching rules.

    Each key in *rules* is matched against *columns* by substring containment.
    A column that matches no rule gets weight 1.0.  A column that matches
    more than one rule raises ``ValueError``.  A rule that matches *no*
    column also raises ``ValueError`` (likely a typo in the rule key).

    Parameters
    ----------
    columns : list[str] | pd.Index
        The embedding column names (e.g. from ``embedding_df.columns``).
    rules : dict[str, float]
        Mapping of substring → weight, e.g. ``{"speed": 4.0, "accel": 2.0}``.

    Returns
    -------
    dict[str, float]
        ``{column_name: weight}`` for every column.

    Raises
    ------
    ValueError
        If a column matches more than one rule, or a rule matches no column.

    Examples
    --------
    >>> cols = ["speed_t0", "speed_t+1", "accel_t0", "dist_t0"]
    >>> build_column_weights(cols, {"speed": 4.0, "accel": 2.0})
    {'speed_t0': 4.0, 'speed_t+1': 4.0, 'accel_t0': 2.0, 'dist_t0': 1.0}

    """
    weights: dict[str, float] = {}
    used_rules: set[str] = set()
    for col in columns:
        matches = [key for key in rules if key in col]
        if len(matches) > 1:
            raise ValueError(f"Column '{col}' matches multiple rules: {matches}")
        if matches:
            weights[col] = rules[matches[0]]
            used_rules.add(matches[0])
        else:
            weights[col] = 1.0

    unused = set(rules) - used_rules
    if unused:
        raise ValueError(
            f"feature_weights keys matched no columns: {sorted(unused)}. "
            f"Check for typos. Available columns: {list(columns)}"
        )
    return weights


def latencies_from_bool(ser: pd.Series) -> list[int]:
    """
    Takes a boolean series and calculates any onsets where
    n = False -> n+1 = True.

    Args:
        ser (pd.Series): a series of boolean type

    Returns:
        list[int]: a list of onset positions (integer indices)

    Examples:
    ```pycon
     >>> import pandas as pd
     >>> import pytest
     >>> series = pd.Series([1,1,0,0,0,1,1,0,0,1,1], dtype = bool)
     >>> latencies = latencies_from_bool(series)
     >>> latencies
     [0, 5, 9]

     >>> series_NAs = pd.Series([1,1,0,0,pd.NA,1,1,0,0,1,1], dtype = 'boolean')
     >>> latencies = latencies_from_bool(series_NAs)
     >>> latencies
     [0, 5, 9]

     ```

    """
    if not pd.api.types.is_bool_dtype(ser):
        raise TypeError("Series must be of boolean dtype")

    # Treat NaNs as False to avoid spurious onsets
    s = ser.fillna(False)

    onsets = ~s.shift(1, fill_value=False) & s
    return onsets[onsets].index.to_list()


def smooth_bool_series(ser: pd.Series, window: int = 1) -> pd.Series:
    """
    Smooth a boolean series using majority voting over a rolling window.
    Removes single-sample spikes and dropouts.

    Args:
        ser (pd.Series): boolean time series

    Returns:
        pd.Series: smoothed boolean series

    Examples:
    ```pycon
     >>> import pandas as pd
     >>> import pytest
     >>> series = pd.Series([1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1], dtype = bool)
     >>> smoothed = smooth_bool_series(series, window = 3)
     >>> smoothed = [int(s) for s in smoothed]
     >>> smoothed
     [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]

     ```
    """
    if not pd.api.types.is_bool_dtype(ser):
        raise TypeError("Series must be of boolean dtype")

    s = ser.fillna(False)
    smoothed = s.rolling(window=window, center=True, min_periods=1).mean() >= 0.5
    return smoothed.astype(bool)


def latencies_from_series(
    series: pd.Series,
    target_value: str | float | int | None = None,
    threshold_op: Literal[">", ">=", "<=", "<", "==", "!="] = "==",
    integration_window: int = 1,
) -> list[int]:
    """
    Compute onset latencies from a pandas Series.

    The input series is converted to a boolean condition either directly
    (if already boolean) or by applying a comparison against `target_value`
    using `threshold_op`. Optionally, the boolean series is temporally
    smoothed via an integration window. Latencies are defined as indices
    where the signal transitions from False to True.

    Parameters
    ----------
    series : pd.Series
        Input time series.
    target_value : str | float | int | None, optional
        Value to compare against for non-boolean series. Required unless
        `series` is already boolean.
    threshold_op : {">", ">=","<=", "<", "==", "!="}, default "=="
        Comparison operator used to generate the boolean condition.
        Only "==" and "!=" are valid for string comparisons.
    integration_window : int, default 1
        Window size for boolean integration/smoothing. Values > 1 apply
        temporal smoothing before latency extraction.

    Returns
    -------
    list[int]
        Indices of False → True transitions in the resulting boolean series.

    Raises
    ------
    ValueError
        If `threshold_op` is invalid, `target_value` is missing for a
        non-boolean series, or an invalid operator is used for string data.

    Examples:
    ```pycon
     >>> import pandas as pd
     >>> import pytest
     >>> series_bool = pd.Series([1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1], dtype = bool)
     >>> latencies = latencies_from_series(series_bool)
     >>> latencies
     [0, 5, 8]
     >>> series_NA = pd.Series([1, 1, 1, 0, pd.NA, 1, 0, 0, 1, 1, 1], dtype = 'boolean')
     >>> latencies = latencies_from_series(series_NA)
     >>> latencies
     [0, 5, 8]
     >>> latencies = latencies_from_series(series_bool, integration_window = 3)
     >>> latencies
     [0, 8]
     >>> series_float = pd.Series([1., 2., 3., 4., 1., 1., 2., 3., 3., 4., 5.], dtype = float)
     >>> latencies = latencies_from_series(series_float, target_value = 2., threshold_op = ">" )
     >>> latencies
     [2, 7]
     >>> series_int = pd.Series([1, 2, 3, 4, 1, 1, 2, 3, 3, 4, 5], dtype = int)
     >>> latencies = latencies_from_series(series_int, target_value = 2, threshold_op = "<" )
     >>> latencies
     [0, 4]
     >>> series_str = pd.Series(['A', 'A', 'B', 'A', 'A', 'C', 'C', 'A', 'A'], dtype = str)
     >>> latencies = latencies_from_series(series_str, target_value = 'A')
     >>> latencies
     [0, 3, 7]
     >>> latencies = latencies_from_series(series_str, target_value = 'A', threshold_op = "!=")
     >>> latencies
     [2, 5]

     ```
    """

    ops = {
        ">": lambda s: s > target_value,
        "<": lambda s: s < target_value,
        ">=": lambda s: s >= target_value,
        "<=": lambda s: s <= target_value,
        "==": lambda s: s == target_value,
        "!=": lambda s: s != target_value,
    }
    valid_str_ops = {"==", "!="}
    series_dtype = series.dtype
    target_dtype = pd.Series([target_value]).dtype

    if threshold_op not in ops:
        raise ValueError(f"Invalid threshold_op: {threshold_op}")

    # If already boolean, ignore target_value and op
    if pd.api.types.is_bool_dtype(series):
        bool_series = series.copy()
    else:
        if target_dtype is None:
            raise ValueError("target_value must be provided for non-boolean series")

        if not np.issubdtype(series_dtype, target_dtype.type) and not np.issubdtype(
            target_dtype, series_dtype.type
        ):
            raise TypeError(
                f"Dtype mismatch: series dtype={series_dtype}, target_value dtype={target_dtype}"
            )

        if isinstance(target_value, str) and threshold_op not in valid_str_ops:
            raise ValueError(
                f"Operator '{threshold_op}' not valid for string comparisons {valid_str_ops}"
            )
        bool_series = ops[threshold_op](series)

    if integration_window > 1:
        bool_series = smooth_bool_series(bool_series, integration_window)

    return latencies_from_bool(bool_series)


def ensure_nullable_boolean(series: pd.Series, *, label: str) -> pd.Series:
    """
    Validate/coerce a series to pandas nullable boolean dtype.

    Accepts:
    - native bool dtype
    - pandas nullable boolean dtype
    - object-like series containing only True/False/NA values

    Examples
    --------
    ```pycon
    >>> import pandas as pd
    >>> s = pd.Series([True, False, None], dtype="object")
    >>> out = ensure_nullable_boolean(s, label="zone")
    >>> str(out.dtype)
    'boolean'
    >>> out.tolist()
    [True, False, <NA>]

    ```
    """
    if is_bool_dtype(series):
        return series.astype("boolean")

    non_na = series.dropna()
    # Strict bool-like check to avoid accepting integer 0/1 as booleans.
    if non_na.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        warnings.warn(
            f"Boolean source '{label}' had non-boolean dtype; coercing to nullable boolean.",
            stacklevel=3,
        )
        return series.astype("boolean")

    raise TypeError(
        f"Source '{label}' must be boolean/nullable-boolean (or contain only True/False/NA). "
        f"Got dtype '{series.dtype}'."
    )


def compose_state_from_boolean_sources(
    sources: dict[str, pd.Series],
    *,
    index: pd.Index,
    priority: list[str] | None = None,
    none_label: str = "none",
) -> pd.Series:
    """
    Compose a single categorical state series from labeled boolean sources.

    State assignment follows ``priority`` order. The first True label wins per frame.
    Frames with no True label are assigned ``none_label``.

    Examples
    --------
    ```pycon
    >>> import pandas as pd
    >>> idx = pd.RangeIndex(4, name="frame")
    >>> sources = {
    ...     "corner": pd.Series([True, False, True, False], index=idx),
    ...     "food": pd.Series([False, True, True, False], index=idx),
    ... }
    >>> state = compose_state_from_boolean_sources(
    ...     sources,
    ...     index=idx,
    ...     priority=["food", "corner"],
    ...     none_label="none",
    ... )
    >>> state.tolist()
    ['corner', 'food', 'food', 'none']

    ```
    """
    if not isinstance(sources, dict) or len(sources) == 0:
        raise ValueError("sources must be a non-empty mapping of label -> Series")

    labels = list(sources.keys())
    if any(not isinstance(lbl, str) or lbl == "" for lbl in labels):
        raise TypeError("All source labels must be non-empty strings")

    if priority is None:
        order = labels
    else:
        if len(priority) != len(set(priority)):
            raise ValueError("priority contains duplicate labels")
        unknown = [lbl for lbl in priority if lbl not in sources]
        if unknown:
            raise ValueError(f"priority contains unknown label(s): {unknown}")
        missing = [lbl for lbl in labels if lbl not in priority]
        order = list(priority) + missing

    bool_df = pd.DataFrame(index=index)
    for label, series in sources.items():
        if not isinstance(series, pd.Series):
            raise TypeError(f"Source '{label}' must be a pandas Series")
        aligned = series.reindex(index)
        bool_df[label] = ensure_nullable_boolean(aligned, label=label)

    state = pd.Series(none_label, index=index, dtype="object")
    for label in order:
        mask = bool_df[label].fillna(False)
        state.loc[(state == none_label) & mask] = label

    return state
