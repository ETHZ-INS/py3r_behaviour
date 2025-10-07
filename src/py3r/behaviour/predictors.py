import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
import numpy as np

try:
    from annoy import AnnoyIndex
except ImportError:
    AnnoyIndex = None


class BasePredictor:
    """
    Abstract base class for predictors.
    """

    def fit(
        self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs
    ) -> "BasePredictor":
        raise NotImplementedError

    def predict(self, test_X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def fit_predict(
        self,
        train_X: pd.DataFrame,
        train_y: pd.DataFrame,
        test_X: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        self.fit(train_X, train_y, **kwargs)
        return self.predict(test_X, **kwargs)


class KNNPredictor(BasePredictor):
    """
    Ordinary k-Nearest Neighbors regressor predictor.

    Notes
    -----
    - During fit, rows with any NaNs in train_X or train_y are dropped.
    - During predict, only rows with no NaNs in test_X are predicted; others are filled with NaN in the output.
    """

    def __init__(self, n_neighbors: int = 5, **kwargs):
        self.n_neighbors = n_neighbors
        self.model = None
        self.model_kwargs = kwargs

    def fit(
        self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs
    ) -> "KNNPredictor":
        # Drop rows with NaNs in either train_X or train_y
        valid_mask = train_X.notna().all(axis=1) & train_y.notna().all(axis=1)
        train_X_valid = train_X[valid_mask]
        train_y_valid = train_y[valid_mask]
        self.model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors, **self.model_kwargs
        )
        self.model.fit(train_X_valid, train_y_valid)
        # Store columns for output
        self._output_columns = train_y.columns if hasattr(train_y, "columns") else None
        return self

    def predict(self, test_X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Only predict for rows with no NaNs
        valid_mask = test_X.notna().all(axis=1)
        test_X_valid = test_X[valid_mask]
        # Prepare output
        n_rows = len(test_X)
        if self._output_columns is not None:
            n_cols = len(self._output_columns)
        else:
            # Fallback: try to infer from model
            n_cols = 1
        preds = np.full((n_rows, n_cols), np.nan)
        if len(test_X_valid) > 0:
            y_pred = self.model.predict(test_X_valid)
            # Handle 1D output
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            preds[valid_mask, : y_pred.shape[1]] = y_pred
        columns = self._output_columns if self._output_columns is not None else None
        return pd.DataFrame(preds, index=test_X.index, columns=columns)

    def fit_predict(
        self,
        train_X: pd.DataFrame,
        train_y: pd.DataFrame,
        test_X: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        self.fit(train_X, train_y, **kwargs)
        return self.predict(test_X, **kwargs)


class KNNPredictorPCA(BasePredictor):
    """
    K-Nearest Neighbors regressor predictor with PCA preprocessing.

    Notes
    -----
    - During fit, rows with any NaNs in train_X or train_y are dropped.
    - During predict, only rows with no NaNs in test_X are predicted; others are filled with NaN in the output.
    """

    def __init__(self, n_neighbors: int = 5, n_components: int = 10, **kwargs):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.knn_model = None
        self.pca_model = None
        # Split kwargs by prefix
        self.knn_kwargs = {}
        self.pca_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith("knn__"):
                param = k[5:]
                if param == "n_neighbors":
                    raise ValueError(
                        "Pass 'n_neighbors' only as a top-level argument, not as 'knn__n_neighbors'."
                    )
                self.knn_kwargs[param] = v
            elif k.startswith("pca__"):
                param = k[5:]
                if param == "n_components":
                    raise ValueError(
                        "Pass 'n_components' only as a top-level argument, not as 'pca__n_components'."
                    )
                self.pca_kwargs[param] = v
        # Defensive: check for direct collision in kwargs
        if "n_neighbors" in self.knn_kwargs:
            raise ValueError(
                "Pass 'n_neighbors' only as a top-level argument, not as 'knn__n_neighbors'."
            )
        if "n_components" in self.pca_kwargs:
            raise ValueError(
                "Pass 'n_components' only as a top-level argument, not as 'pca__n_components'."
            )

    def fit(
        self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs
    ) -> "KNNPredictorPCA":
        # Drop rows with NaNs in either train_X or train_y
        valid_mask = train_X.notna().all(axis=1) & train_y.notna().all(axis=1)
        train_X_valid = train_X[valid_mask]
        train_y_valid = train_y[valid_mask]
        self.pca_model = PCA(n_components=self.n_components, **self.pca_kwargs)
        train_X_pca = self.pca_model.fit_transform(train_X_valid)
        self.knn_model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors, **self.knn_kwargs
        )
        self.knn_model.fit(train_X_pca, train_y_valid)
        # Store columns for output
        self._output_columns = train_y.columns if hasattr(train_y, "columns") else None
        return self

    def predict(self, test_X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Only predict for rows with no NaNs
        valid_mask = test_X.notna().all(axis=1)
        test_X_valid = test_X[valid_mask]
        n_rows = len(test_X)
        if self._output_columns is not None:
            n_cols = len(self._output_columns)
        else:
            n_cols = 1
        preds = np.full((n_rows, n_cols), np.nan)
        if len(test_X_valid) > 0:
            test_X_pca = self.pca_model.transform(test_X_valid)
            preds_valid = self.knn_model.predict(test_X_pca)
            if preds_valid.ndim == 1:
                preds_valid = preds_valid.reshape(-1, 1)
            preds[valid_mask, : preds_valid.shape[1]] = preds_valid
        columns = self._output_columns if self._output_columns is not None else None
        return pd.DataFrame(preds, index=test_X.index, columns=columns)

    def fit_predict(
        self,
        train_X: pd.DataFrame,
        train_y: pd.DataFrame,
        test_X: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        self.fit(train_X, train_y, **kwargs)
        return self.predict(test_X, **kwargs)


class KNNPredictorPCAnnoy(BasePredictor):
    """
    Fast approximate kNN regressor using Annoy, with optional PCA preprocessing.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for regression.
    n_components : int or None, default=None
        If set, use PCA to reduce dimensionality to n_components before building Annoy index.
    n_trees : int, default=10
        Number of trees to build in Annoy index.
    search_k : int or None, default=None
        Number of nodes to inspect during search (higher = more accurate, slower). If None, Annoy default is used.
    metric : str, default='euclidean'
        Distance metric for Annoy ('euclidean', 'manhattan', etc.).
    **kwargs :
        Ignored (for API compatibility).

    Notes
    -----
    - During fit, rows with any NaNs in train_X or train_y are dropped.
    - During predict, only rows with no NaNs in test_X are predicted; others are filled with NaN in the output.
    - Requires the `annoy` package: pip install annoy
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        n_components: int | None = 10,
        n_trees=10,
        search_k=None,
        metric="euclidean",
        **kwargs,
    ):
        if AnnoyIndex is None:
            raise ImportError(
                "KNNPredictorPCAnnoy requires the 'annoy' package. Install with 'pip install annoy'."
            )
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.n_trees = n_trees
        self.search_k = search_k
        self.metric = metric
        self.pca_model = None
        self.annoy_index = None
        self._train_X = None
        self._train_y = None
        self._output_columns = None

    def fit(
        self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs
    ) -> "KNNPredictorPCAnnoy":
        # Drop rows with NaNs in either train_X or train_y
        valid_mask = train_X.notna().all(axis=1) & train_y.notna().all(axis=1)
        train_X_valid = train_X[valid_mask]
        train_y_valid = train_y[valid_mask]
        # Optionally fit PCA
        if self.n_components is not None:
            from sklearn.decomposition import PCA

            self.pca_model = PCA(n_components=self.n_components)
            train_X_vecs = self.pca_model.fit_transform(train_X_valid)
        else:
            train_X_vecs = train_X_valid.values
        # Build Annoy index
        self.annoy_index = AnnoyIndex(train_X_vecs.shape[1], self.metric)
        for i, v in enumerate(train_X_vecs):
            self.annoy_index.add_item(i, v)
        self.annoy_index.build(self.n_trees)
        self._train_X = train_X_valid.reset_index(drop=True)
        self._train_y = train_y_valid.reset_index(drop=True)
        self._output_columns = train_y.columns if hasattr(train_y, "columns") else None
        return self

    def predict(self, test_X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Only predict for rows with no NaNs
        valid_mask = test_X.notna().all(axis=1)
        test_X_valid = test_X[valid_mask]
        n_rows = len(test_X)
        if self._output_columns is not None:
            n_cols = len(self._output_columns)
        else:
            n_cols = 1
        preds = np.full((n_rows, n_cols), np.nan)
        if len(test_X_valid) > 0:
            # Transform with PCA if needed
            if self.pca_model is not None:
                test_X_vecs = self.pca_model.transform(test_X_valid)
            else:
                test_X_vecs = test_X_valid.values
            # For each query, get neighbors and average their train_y
            for idx, (i, v) in enumerate(zip(test_X_valid.index, test_X_vecs)):
                if self.search_k is not None:
                    nn_idx = self.annoy_index.get_nns_by_vector(
                        v, self.n_neighbors, search_k=self.search_k
                    )
                else:
                    nn_idx = self.annoy_index.get_nns_by_vector(v, self.n_neighbors)
                y_neighbors = self._train_y.iloc[nn_idx].values
                y_pred = y_neighbors.mean(axis=0)
                preds[test_X.index.get_loc(i), : y_pred.shape[0]] = y_pred
        columns = self._output_columns if self._output_columns is not None else None
        return pd.DataFrame(preds, index=test_X.index, columns=columns)

    def fit_predict(
        self,
        train_X: pd.DataFrame,
        train_y: pd.DataFrame,
        test_X: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        self.fit(train_X, train_y, **kwargs)
        return self.predict(test_X, **kwargs)
