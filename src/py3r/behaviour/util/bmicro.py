import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from typing import List, Tuple


def train_knn_from_embeddings(
    train_list: List[pd.DataFrame],
    target_list: List[pd.DataFrame],
    n_neighbors: int = 5,
    **kwargs
) -> Tuple[KNeighborsRegressor, pd.Index, pd.Index]:
    """
    Trains a KNN regressor from lists of input and target embedding DataFrames.
    Concatenates all train and target, drops rows with any NaNs, and fits the model.
    Returns the trained model, input columns, and target columns.
    """
    # Check that all train dataframes have the same columns
    for train in train_list:
        if not train.columns.equals(train_list[0].columns):
            raise ValueError("All train_list dataframes must have the same columns")
    # Check that all target dataframes have the same columns
    for target in target_list:
        if not target.columns.equals(target_list[0].columns):
            raise ValueError("All target_list dataframes must have the same columns")
    train = pd.concat(train_list, axis=0)
    target = pd.concat(target_list, axis=0)
    valid_mask = train.notna().all(axis=1) & target.notna().all(axis=1)
    train_valid = train[valid_mask]
    target_valid = target[valid_mask]
    if len(train_valid) == 0:
        raise ValueError("No valid rows to train on after dropping NaNs.")
    model = KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs)
    model.fit(train_valid, target_valid)
    return model, train_valid.columns, target_valid.columns


def predict_knn_on_embedding(
    model: KNeighborsRegressor,
    test: pd.DataFrame,
    target_columns: pd.Index
) -> pd.DataFrame:
    """
    Predicts using a trained KNN regressor on a new embedding DataFrame.
    Returns a DataFrame of predictions indexed like test, with columns matching target_columns.
    """
    valid_mask = test.notna().all(axis=1)
    preds = pd.DataFrame(np.nan, index=test.index, columns=target_columns)
    if valid_mask.any():
        y_pred = model.predict(test[valid_mask])
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        preds.loc[valid_mask, :] = y_pred
    return preds


def rms_error_between_embeddings(df1: pd.DataFrame, df2: pd.DataFrame, rescale_factors: dict = None) -> pd.Series:
    """
    Compute the root mean squared error (RMS) for each row between two embedding DataFrames.
    If rescale_factors is provided, normalize both DataFrames using this dict before computing the error.
    Returns a Series indexed like the input DataFrames, with NaN for rows where either input has NaNs.
    """
    if not df1.columns.equals(df2.columns) or not df1.index.equals(df2.index):
        raise ValueError("Input DataFrames must have the same columns and index")
    if rescale_factors is not None:
        if set(df1.columns) != set(rescale_factors.keys()):
            raise ValueError("Columns of DataFrames and rescale_factors do not match.")
        df1 = df1.copy()
        df2 = df2.copy()
        for col in df1.columns:
            df1[col] = df1[col] / rescale_factors[col]
            df2[col] = df2[col] / rescale_factors[col]
    diff = df1 - df2
    # Compute RMS error for each row, ignoring rows with any NaNs
    rms = np.sqrt((diff ** 2).mean(axis=1))
    # Set to NaN if either input row has any NaNs
    mask = df1.notna().all(axis=1) & df2.notna().all(axis=1)
    rms[~mask] = np.nan
    return rms