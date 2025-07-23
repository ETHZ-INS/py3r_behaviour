import pandas as pd
from typing import Any
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA

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
    
class KNNPredictorPCA(BasePredictor):
    """
    K-Nearest Neighbors regressor predictor with PCA preprocessing.
    """
    def __init__(self, n_neighbors: int = 5, n_components: int = 10, **kwargs):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.knn_model = None
        self.pca_model = None
        self.knn_kwargs = kwargs['knn_kwargs']
        self.pca_kwargs = kwargs['pca_kwargs']

    def fit(self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs) -> "KNNPredictorPCA":
        self.pca_model = PCA(n_components=self.n_components, **self.pca_kwargs)
        train_X_pca = self.pca_model.fit_transform(train_X)
        self.knn_model = KNeighborsRegressor(n_neighbors=self.n_neighbors, **self.knn_kwargs)
        self.knn_model.fit(train_X_pca, train_y)
        return self

    def predict(self, test_X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        test_X_pca = self.pca_model.transform(test_X)
        preds = self.knn_model.predict(test_X_pca)
        return pd.DataFrame(preds, index=test_X.index, columns=self.knn_model._y.columns if hasattr(self.knn_model, '_y') else None)
        
    def fit_predict(self, train_X: pd.DataFrame, train_y: pd.DataFrame, test_X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.fit(train_X, train_y, **kwargs)
        return self.predict(test_X, **kwargs) 
        