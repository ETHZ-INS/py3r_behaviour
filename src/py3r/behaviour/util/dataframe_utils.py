import pandas as pd
import numpy as np
from typing import Literal

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