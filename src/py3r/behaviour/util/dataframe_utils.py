from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd


def filter_by_threshold(
    df: pd.DataFrame,
    reference_col: str,
    threshold: float,
    operation: Literal["gt", "ge", "lt", "le", "eq", "ne"] = "ge",
) -> pd.DataFrame:
    """
    Filter rows of a DataFrame based on a threshold applied to a reference column.

    Rows that do NOT satisfy the comparison are not removed. Instead, all columns
    EXCEPT the reference column are set to NaN for those rows. The reference
    column is always preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to be filtered.
    reference_col : str
        Name of the column against which the threshold comparison is applied.
    threshold : float
        Threshold value used in the comparison.
    operation : {"gt", "ge", "lt", "le", "eq", "ne"} defaults to "ge"
        Comparison operation to apply:
        - "gt": greater than (>)
        - "ge": greater than or equal (>=)
        - "lt": less than (<)
        - "le": less than or equal (<=)
        - "eq": equal to (==)
        - "ne": not equal (!=)

    Returns
    -------
    pd.DataFrame
        A new DataFrame where rows failing the comparison are set entirely to NaN.

    Example
    ```pycon
    >>> import pandas as pd
    >>> import pytest
    >>> df = pd.DataFrame({'a':[1.,2.,3.],'b':[2.,3.,4.]})
    >>> df_filtered = filter_by_threshold(df, 'a', 2.)
    >>> print(df_filtered['a'].values)
    [1. 2. 3.]
    >>> print(df_filtered['b'].values)
    [nan 3. 4.]
    >>> with pytest.raises(KeyError):
    >>>     filter_by_threshold(df, 'non_existing', 2.)
    >>> with pytest.raises(ValueError):
    >>>     filter_by_threshold(df, 'a', 2., operation = 'invalid')

    ```
    """
    if reference_col not in df.columns:
        raise KeyError(f"Column '{reference_col}' not found in DataFrame")

    ops = {
        "gt": lambda s: s > threshold,
        "ge": lambda s: s >= threshold,
        "lt": lambda s: s < threshold,
        "le": lambda s: s <= threshold,
        "eq": lambda s: s == threshold,
        "ne": lambda s: s != threshold,
    }

    if operation not in ops:
        raise ValueError(f"Unsupported operation: {operation}")

    mask = ops[operation](df[reference_col])

    result = df.copy()
    non_ref_cols = result.columns.difference([reference_col])
    result.loc[~mask, non_ref_cols] = np.nan

    return result


def euclidean_distance(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    method: Literal["median", "element_wise"] = "element_wise",
    dims: Iterable[str] | None = None,
) -> float | pd.Series:
    """
     Compute Euclidean distance between two DataFrames in N-dimensional space.

    Methods
    -------
     - 'median'      : distance between column-wise median vectors (scalar)
     - 'element_wise': row-wise Euclidean distance (pd.Series)

    Parameters
    ----------
     df1 : pd.DataFrame
         First DataFrame.
     df2 : pd.DataFrame
         Second DataFrame.
     method : {'median', 'element_wise'}, default 'element_wise'
         Distance computation strategy.
     dims : Tuple of str, optional
         Columns to use. If None, uses the intersection of numeric columns.

    Returns
    -------
     float or pd.Series
         Scalar distance for 'median';
         Series of row-wise distances for 'element_wise'.

    Raises
    ------
     ValueError
         If no common numeric columns are found or indices mismatch for element-wise.
     KeyError
         If specified dims are missing from either DataFrame.

    Example
    ```pycon
     >>> import pandas as pd
     >>> import pytest
     >>> df1 = pd.DataFrame({'a':[1.,2.,3.],'b':[2.,3.,4.],'c':[4.,5.,6.],'d':['x','y','z']})
     >>> df2 = pd.DataFrame({'a':[2.,3.,4.],'b':[2.,3.,4.],'c':[4.,5.,6.],'d':['x','y','z']})
     >>> df3 = pd.DataFrame({'e':[1.,2.,3.],'f':[2.,3.,4.],'g':[4.,5.,6.],'h':['x','y','z']})
     >>> euclidean_distance(df1, df2, "median")
     1.0
     >>> euclidean_distance(df1, df2).values
     array([1., 1., 1.])
     >>> euclidean_distance(df1, df2, dims = ('b','c')).values
     array([0., 0., 0.])
     >>> with pytest.raises(KeyError):
     >>>     euclidean_distance(df1, df3, dims = ('x','d'))
     >>> with pytest.raises(KeyError):
     >>>     euclidean_distance(df1, df2, dims = ('a','non_existing'))
     >>> with pytest.raises(ValueError):
     >>>     euclidean_distance(df1, df3)
     >>> with pytest.raises(ValueError):
     >>>     euclidean_distance(df1, df2, method = 'non_existing')

     ```
    """
    if dims is None:
        dims = sorted(
            set(df1.select_dtypes(include="number").columns)
            & set(df2.select_dtypes(include="number").columns)
        )
        if not dims:
            raise ValueError("No common numeric columns found.")
    else:
        dims = tuple(dims)
        missing_1 = set(dims) - set(df1.columns)
        missing_2 = set(dims) - set(df2.columns)
        if missing_1:
            raise KeyError(f"Missing columns in df1: {missing_1}")
        if missing_2:
            raise KeyError(f"Missing columns in df2: {missing_2}")

    if method == "median":
        v1 = df1.loc[:, dims].median().to_numpy(dtype=float)
        v2 = df2.loc[:, dims].median().to_numpy(dtype=float)
        return float(np.linalg.norm(v1 - v2))

    if method == "element_wise":
        if not df1.index.equals(df2.index):
            raise ValueError("DataFrames must have identical indices for element-wise distance.")
        diff = df1.loc[:, dims].to_numpy(dtype=float) - df2.loc[:, dims].to_numpy(dtype=float)
        return pd.Series(np.linalg.norm(diff, axis=1), index=df1.index, name="euclidean_distance")

    raise ValueError(f"Unknown method: {method}")


def point_to_axis_distance(
    P: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    *,
    signed: bool = False,
) -> np.ndarray:
    """Compute framewise perpendicular distance from a point to an infinite axis.

    The axis passes through A and B and extends infinitely in both directions.
    All arrays must share the same shape ``(n_frames, n_dims)``.

    Parameters
    ----------
    P : np.ndarray
        Query point coordinates, shape ``(n_frames, n_dims)``.
    A : np.ndarray
        First axis reference point, shape ``(n_frames, n_dims)``.
    B : np.ndarray
        Second axis reference point, shape ``(n_frames, n_dims)``.
    signed : bool, default False
        If True, return a signed distance (2-D axes only).  Positive means P
        is to the *right* when facing from A to B; negative means P is to the
        *left*.  Raises ``ValueError`` for ``n_dims != 2`` when True.

    Returns
    -------
    np.ndarray
        Framewise (signed) perpendicular distances, shape ``(n_frames,)``.

    Notes
    -----
    Degenerate frames where A == B (zero-length direction) return ``nan``
    for both the signed and unsigned cases, as the axis direction is undefined.

    Examples
    --------
    ```pycon
    >>> import numpy as np
    >>> P = np.array([[0.0, 1.0], [3.0, 0.0], [0.0, -1.0]])
    >>> A = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    >>> B = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    >>> point_to_axis_distance(P, A, B)
    array([1., 0., 1.])
    >>> point_to_axis_distance(P, A, B, signed=True)
    array([-1.,  0.,  1.])

    ```
    """
    AP = P - A  # (n, d)
    AB = B - A  # (n, d)

    ab_sq = np.sum(AB * AB, axis=1)  # (n,)

    if signed:
        n_dims = P.shape[1]
        if n_dims != 2:
            raise ValueError(f"signed=True requires a 2-D axis; got n_dims={n_dims}.")
        # Signed distance = projection of AP onto the right-hand perpendicular
        # of the unit direction d = AB / |AB|.
        # perp_right = (d[1], -d[0])  →  AP · perp_right / |AB| * |AB| simplifies to:
        #   (AP[0] * AB[1] - AP[1] * AB[0]) / |AB|
        # Degenerate frames (A == B) → axis undefined, return NaN.
        ab_norm = np.sqrt(ab_sq)
        with np.errstate(invalid="ignore", divide="ignore"):
            cross = AP[:, 0] * AB[:, 1] - AP[:, 1] * AB[:, 0]
            return np.where(ab_norm > 0, cross / ab_norm, np.nan)

    # Scalar projection of P onto the infinite axis through A and B.
    # Degenerate frames (A == B) → axis undefined, return NaN.
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(ab_sq > 0, np.sum(AP * AB, axis=1) / ab_sq, np.nan)

    closest = A + t[:, np.newaxis] * AB  # (n, d)
    return np.linalg.norm(P - closest, axis=1)  # (n,)


def scale_columns(df: pd.DataFrame, factor: float, cols: Iterable[str]) -> pd.DataFrame:
    """
     Multiply selected DataFrame columns by a scalar factor.

    Parameters
    ----------
     df : pd.DataFrame
         Input DataFrame.
     factor : float
         Scalar multiplier.
     cols : Tuple[str]
         Columns to scale.

    Returns
    -------
     pd.DataFrame
         DataFrame with scaled columns.

    Example
    ```pycon
     >>> import pandas as pd
     >>> import pytest
     >>> df = pd.DataFrame({'a':[1.,2.,3.],'b':[2.,3.,4.],'c':[4.,5.,6.],'d':['x','y','z']})
     >>> df_scaled = scale_columns(df, 2.0, ('a','b'))
     >>> df_scaled['a'].values
     array([2., 4., 6.])
     >>> df_scaled['b'].values
     array([4., 6., 8.])
     >>> df_scaled['c'].values
     array([4., 5., 6.])
     >>> with pytest.raises(TypeError):
     >>>     df_scaled = scale_columns(df, 2.0, ('a','d'))
     >>> with pytest.raises(KeyError):
     >>>     df_scaled = scale_columns(df, 2.0, ('a','x'))

     ```
    """
    out = df.copy()

    for c in cols:
        if c not in out.columns:
            raise KeyError(f"Column '{c}' not found in DataFrame")
        if not pd.api.types.is_numeric_dtype(out[c]):
            raise TypeError(f"Column '{c}' must be numeric, got dtype {out[c].dtype}")
        out[c] *= factor

    return out


def normalize_transition_matrix(tm: pd.DataFrame) -> pd.DataFrame:
    """
    Row-normalise a transition-count matrix to transition probabilities.

    Each row is divided by its row sum so that non-zero rows become probability
    distributions that sum to 1.  Rows whose sum is zero (a state that was
    observed but never left in this recording) are filled with ``0.0`` rather
    than ``NaN``, treating an unseen transition as having probability 0.

    Parameters
    ----------
    tm : pd.DataFrame
        Square (or rectangular) transition-count matrix with matching index
        and columns.

    Returns
    -------
    pd.DataFrame
        Row-normalised DataFrame with the same shape, index, and columns as
        the input.

    Example
    ```pycon
    >>> import pandas as pd
    >>> tm = pd.DataFrame({'A': [3, 0], 'B': [1, 0]}, index=['A', 'B'])
    >>> result = normalize_transition_matrix(tm)
    >>> float(result.loc['A', 'A'])
    0.75
    >>> float(result.loc['A', 'B'])
    0.25
    >>> float(result.loc['B', 'A'])
    0.0
    >>> float(result.loc['B', 'B'])
    0.0

    ```
    """
    row_sums = tm.sum(axis=1)
    return tm.div(row_sums.replace(0, float("nan")), axis=0).fillna(0.0)


def coarse_grain_dataframe(
    data: pd.DataFrame,
    *,
    window: int,
    method: Literal["mean", "median", "min", "max"] = "mean",
    non_numeric: Literal["drop", "nan", "first", "mode", "error"] = "drop",
) -> pd.DataFrame:
    """
    Coarse-grain a DataFrame over fixed, non-overlapping row windows.

    Numeric columns are aggregated with ``method``. Boolean and other
    non-numeric columns are handled according to ``non_numeric``.
    """
    valid_methods = {"mean", "median", "min", "max"}
    if not isinstance(window, int) or window < 1:
        raise ValueError(f"window must be a positive integer, got {window!r}")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {sorted(valid_methods)}, got {method!r}")

    bins = np.arange(len(data)) // window
    grouped = data.groupby(bins, sort=True)

    non_numeric_cols = [
        col
        for col in data.columns
        if pd.api.types.is_bool_dtype(data[col]) or not pd.api.types.is_numeric_dtype(data[col])
    ]
    numeric_cols = [col for col in data.columns if col not in non_numeric_cols]

    if non_numeric == "error" and non_numeric_cols:
        raise TypeError(
            "coarse_grain encountered non-numeric columns "
            f"(including boolean dtypes): {non_numeric_cols}"
        )

    coarse_data = (
        grouped[numeric_cols].aggregate(method)
        if numeric_cols
        else pd.DataFrame(index=grouped.size().index)
    )

    if non_numeric_cols and non_numeric != "drop":
        if non_numeric == "nan":
            extra = pd.DataFrame(
                {
                    col: pd.Series(pd.NA, index=coarse_data.index, dtype="object")
                    for col in non_numeric_cols
                }
            )
        elif non_numeric == "first":
            extra = grouped[non_numeric_cols].first()
        elif non_numeric == "mode":

            def _mode_or_na(s: pd.Series):
                modes = s.mode(dropna=True)
                return modes.iloc[0] if not modes.empty else pd.NA

            extra = grouped[non_numeric_cols].agg(_mode_or_na)
        else:
            extra = pd.DataFrame(index=coarse_data.index)

        coarse_data = pd.concat([coarse_data, extra], axis=1)[data.columns]
    else:
        coarse_data = coarse_data[numeric_cols]

    coarse_data.index = pd.RangeIndex(
        start=0,
        stop=len(coarse_data),
        step=1,
        name=data.index.name or "frame",
    )
    return coarse_data
