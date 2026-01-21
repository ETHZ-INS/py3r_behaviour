import pandas as pd
import numpy as np
from typing import Literal, Tuple

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


def median_euclidean_distance(df: pd.DataFrame,
                                df2: pd.DataFrame,
                                dims: Tuple[str]) -> float:
    """
    Compute the Euclidean distance between the column-wise medians of two DataFrames.

    For each column name in `dims`, the median is computed in both `df` and `df2`,
    and the Euclidean distance between the resulting median vectors is returned.

    Parameters
    ----------
    df : pd.DataFrame
        First DataFrame.
    df2 : pd.DataFrame
        Second DataFrame.
    dims : Tuple[str]
        Column names over which to compute medians.

    Returns
    -------
    float
        Euclidean distance between the median vectors.
        
    Example
    
    ```pycon
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({'a':[1.,2.,3.],'b':[2.,3.,4.]})
    >>> df2 = pd.DataFrame({'a':[2.,3.,4.],'b':[3.,4.,5.]})
    >>> dist = median_euclidean_distance(df1, df2,('a','b')) 
    >>> dist == np.sqrt(2)
    np.True_
    
    ```
    """
    distance = np.sqrt(
        sum(
            [
                (
                    df[dim].median()
                    - df2[dim].median()
                )
                ** 2
                for dim in dims
            ]
        )
    )
    return distance

def scale_columns(df: pd.DataFrame,
                        factor: float,
                        cols: Tuple[str]) -> pd.DataFrame:
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
            raise TypeError(
                f"Column '{c}' must be numeric, got dtype {out[c].dtype}"
            )
        out[c] *= factor

    return out
    
    