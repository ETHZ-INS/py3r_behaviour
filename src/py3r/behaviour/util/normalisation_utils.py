import pandas as pd

def normalise_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Normalise the columns of a DataFrame by dividing each column by its standard deviation.
    Returns the normalised DataFrame and a dict of the rescaling factors (std for each column).
    """
    rescale_factors = df.std(axis=0, ddof=0).to_dict()
    normalised = df.copy()
    for col, factor in rescale_factors.items():
        normalised[col] = df[col] / factor
    return normalised, rescale_factors

def apply_normalisation_to_df(df: pd.DataFrame, rescale_factors: dict) -> pd.DataFrame:
    """
    Apply normalisation to a DataFrame using the provided rescale factors (dict of column: factor).
    Checks that the columns match exactly. Returns the normalised DataFrame.
    Raises ValueError if columns do not match.
    """
    if set(df.columns) != set(rescale_factors.keys()):
        raise ValueError("Columns of DataFrame and rescale_factors do not match.")
    normalised = df.copy()
    for col in df.columns:
        normalised[col] = df[col] / rescale_factors[col]
    return normalised 