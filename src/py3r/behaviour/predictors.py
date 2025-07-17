import pandas as pd
from typing import Any
from sklearn.neighbors import KNeighborsRegressor

class BasePredictor:
    """
    Abstract base class for predictors.
    """
    def fit(self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs) -> "BasePredictor":
        raise NotImplementedError

    def predict(self, test_X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def fit_predict(self, train_X: pd.DataFrame, train_y: pd.DataFrame, test_X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.fit(train_X, train_y, **kwargs)
        return self.predict(test_X, **kwargs)

class KNNPredictor(BasePredictor):
    """
    Ordinary k-Nearest Neighbors regressor predictor.
    """
    def __init__(self, n_neighbors: int = 5, **kwargs):
        self.n_neighbors = n_neighbors
        self.model = None
        self.model_kwargs = kwargs

    def fit(self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs) -> "KNNPredictor":
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, **self.model_kwargs)
        self.model.fit(train_X, train_y)
        return self

    def predict(self, test_X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        preds = self.model.predict(test_X)
        return pd.DataFrame(preds, index=test_X.index, columns=self.model._y.columns if hasattr(self.model, '_y') else None)

    def fit_predict(self, train_X: pd.DataFrame, train_y: pd.DataFrame, test_X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.fit(train_X, train_y, **kwargs)
        return self.predict(test_X, **kwargs) 