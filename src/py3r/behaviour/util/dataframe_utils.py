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
