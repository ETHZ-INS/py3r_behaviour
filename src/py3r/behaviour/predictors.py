from __future__ import annotations

import json
import os
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

try:
    import joblib
except Exception:  # joblib is a transitive dep of sklearn, but be defensive
    joblib = None

try:
    from annoy import AnnoyIndex
except ImportError:
    AnnoyIndex = None


# Type alias for nan_policy: "drop", "impute" (median), or a constant float
NanPolicy = Literal["drop", "impute"] | float


def _impute_df(df: pd.DataFrame, fill_values: pd.Series) -> pd.DataFrame:
    """Fill NaNs in df using fill_values Series (indexed by column)."""
    return df.fillna(fill_values)


def _observed_fraction(df: pd.DataFrame) -> pd.Series:
    """Compute fraction of non-NaN values per row."""
    n_cols = df.shape[1] if df.shape[1] > 0 else 1
    return (df.notna().sum(axis=1) / n_cols).astype(np.float32)


class BasePredictor:
    """
    Abstract base class for predictors with optional NaN imputation.

    Parameters
    ----------
    nan_policy : {"drop", "impute"} or float, default="drop"
        How to handle NaN values:

        - ``"drop"``: Pass data through unchanged (subclass handles NaNs).
        - ``"impute"``: Fill NaNs with per-column medians from training data.
        - ``float``: Fill NaNs with this constant value.

        The same policy is applied to both X and y.
    """

    def __init__(self, *, nan_policy: NanPolicy = "drop"):
        self.nan_policy = nan_policy
        self._fill_values_X: pd.Series | None = None
        self._fill_values_y: pd.Series | None = None

    def _prepare_train(
        self, train_X: pd.DataFrame, train_y: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply nan_policy to training data, compute and store fill values."""
        if self.nan_policy == "drop":
            self._fill_values_X = None
            self._fill_values_y = None
            return train_X, train_y

        if self.nan_policy == "impute":
            self._fill_values_X = train_X.median()
            self._fill_values_y = train_y.median()
        else:
            # Constant fill
            fill_val = float(self.nan_policy)
            self._fill_values_X = pd.Series(fill_val, index=train_X.columns)
            self._fill_values_y = pd.Series(fill_val, index=train_y.columns)

        return _impute_df(train_X, self._fill_values_X), _impute_df(train_y, self._fill_values_y)

    def _prepare_test(self, test_X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Apply nan_policy to test data, return (imputed_X, observed_fraction)."""
        obs_frac = _observed_fraction(test_X)

        if self.nan_policy == "drop" or self._fill_values_X is None:
            return test_X, obs_frac

        return _impute_df(test_X, self._fill_values_X), obs_frac

    def fit(self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs) -> BasePredictor:
        raise NotImplementedError

    def predict(self, test_X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def predict_with_confidence(
        self, test_X: pd.DataFrame, **kwargs
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Predict and also return observed_fraction for each row."""
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

    # --- Serialization ---

    def _export_state(self) -> dict:
        state = {"nan_policy": self.nan_policy}
        if self._fill_values_X is not None:
            state["fill_values_X"] = self._fill_values_X.to_dict()
        if self._fill_values_y is not None:
            state["fill_values_y"] = self._fill_values_y.to_dict()
        return state

    def _import_state(self, state: dict) -> None:
        self.nan_policy = state.get("nan_policy", "drop")
        fv_x = state.get("fill_values_X")
        self._fill_values_X = pd.Series(fv_x) if fv_x is not None else None
        fv_y = state.get("fill_values_y")
        self._fill_values_y = pd.Series(fv_y) if fv_y is not None else None

    def _save_artifacts(self, dir_path: str) -> dict:
        raise NotImplementedError

    def _load_artifacts(
        self, dir_path: str, files: dict, manifest: dict, mmap: bool = True
    ) -> None:
        raise NotImplementedError

    def save(
        self,
        dir_path: str,
        *,
        input_columns: list[str] | None = None,
        output_columns: list[str] | None = None,
        input_normalization: dict | None = None,
        manifest_extra: dict | None = None,
    ) -> None:
        """Persist the trained predictor to a directory."""
        os.makedirs(dir_path, exist_ok=True)

        if input_columns is None and hasattr(self, "_train_X") and self._train_X is not None:
            try:
                input_columns = list(self._train_X.columns)
            except Exception:
                pass
        if (
            output_columns is None
            and hasattr(self, "_output_columns")
            and self._output_columns is not None
        ):
            try:
                output_columns = list(self._output_columns)
            except Exception:
                pass

        files = self._save_artifacts(dir_path)
        state = self._export_state()

        manifest = {
            "predictor_class": self.__class__.__name__,
            "predictor_module": self.__class__.__module__,
            "state": state,
            "files": files,
            "input_columns": input_columns,
            "output_columns": output_columns,
            "input_normalization": input_normalization,
        }
        if manifest_extra:
            manifest["extra"] = manifest_extra

        with open(os.path.join(dir_path, "manifest.json"), "w") as f:
            json.dump(manifest, f)

    @staticmethod
    def _read_manifest(dir_path: str) -> dict:
        with open(os.path.join(dir_path, "manifest.json")) as f:
            return json.load(f)

    @classmethod
    def load(cls, dir_path: str, mmap: bool = True) -> BasePredictor:
        """Load a predictor bundle previously saved via save()."""
        manifest = cls._read_manifest(dir_path)
        manifest_class = manifest.get("predictor_class")
        if manifest_class is not None and manifest_class != cls.__name__:
            raise ValueError(
                f"Manifest was saved for {manifest_class}, but load() called on {cls.__name__}."
            )
        inst = cls.__new__(cls)
        inst._import_state(manifest.get("state", {}))
        inst._load_artifacts(dir_path, manifest.get("files", {}), manifest, mmap=mmap)
        return inst


class KNNPredictor(BasePredictor):
    """
    Ordinary k-Nearest Neighbors regressor predictor.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    nan_policy : {"drop", "impute"} or float, default="drop"
        NaN handling policy. See BasePredictor.
    **kwargs
        Additional arguments passed to sklearn KNeighborsRegressor.
    """

    def __init__(self, n_neighbors: int = 5, *, nan_policy: NanPolicy = "drop", **kwargs):
        super().__init__(nan_policy=nan_policy)
        self.n_neighbors = n_neighbors
        self.model_kwargs = kwargs
        self.model = None
        self._output_columns = None

    def fit(self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs) -> KNNPredictor:
        train_X, train_y = self._prepare_train(train_X, train_y)
        # Drop any remaining NaNs (when policy is "drop")
        valid = train_X.notna().all(axis=1) & train_y.notna().all(axis=1)
        train_X, train_y = train_X[valid], train_y[valid]

        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, **self.model_kwargs)
        self.model.fit(train_X, train_y)
        self._output_columns = train_y.columns if hasattr(train_y, "columns") else None
        self._train_X = train_X
        return self

    def predict(self, test_X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        test_X, _ = self._prepare_test(test_X)
        valid = test_X.notna().all(axis=1)

        n_rows = len(test_X)
        n_cols = len(self._output_columns) if self._output_columns is not None else 1
        preds = np.full((n_rows, n_cols), np.nan)

        if valid.any():
            y_pred = self.model.predict(test_X[valid])
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            preds[valid.values] = y_pred

        return pd.DataFrame(preds, index=test_X.index, columns=self._output_columns)

    def predict_with_confidence(
        self, test_X: pd.DataFrame, **kwargs
    ) -> tuple[pd.DataFrame, pd.Series]:
        test_X_prep, obs_frac = self._prepare_test(test_X)
        valid = test_X_prep.notna().all(axis=1)

        n_rows = len(test_X_prep)
        n_cols = len(self._output_columns) if self._output_columns is not None else 1
        preds = np.full((n_rows, n_cols), np.nan)

        if valid.any():
            y_pred = self.model.predict(test_X_prep[valid])
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            preds[valid.values] = y_pred

        result = pd.DataFrame(preds, index=test_X.index, columns=self._output_columns)
        return result, obs_frac

    def _export_state(self) -> dict:
        state = super()._export_state()
        state["n_neighbors"] = self.n_neighbors
        state["model_kwargs"] = self.model_kwargs
        return state

    def _import_state(self, state: dict) -> None:
        super()._import_state(state)
        self.n_neighbors = state.get("n_neighbors", 5)
        self.model_kwargs = state.get("model_kwargs", {})
        self.model = None
        self._output_columns = None
        self._train_X = None

    def _save_artifacts(self, dir_path: str) -> dict:
        if self.model is None:
            raise ValueError("Predictor not fitted.")
        if joblib is None:
            raise ImportError("joblib required to save.")
        path = os.path.join(dir_path, "model.joblib")
        joblib.dump(self.model, path)
        return {"model": "model.joblib"}

    def _load_artifacts(
        self, dir_path: str, files: dict, manifest: dict, mmap: bool = True
    ) -> None:
        if joblib is None:
            raise ImportError("joblib required to load.")
        self.model = joblib.load(os.path.join(dir_path, files["model"]))
        self._output_columns = manifest.get("output_columns")
        self._train_X = None


class KNNPredictorPCA(BasePredictor):
    """
    K-Nearest Neighbors regressor with PCA preprocessing.

    Parameters
    ----------
    n_neighbors : int, default=5
    n_components : int, default=10
    nan_policy : {"drop", "impute"} or float, default="drop"
    **kwargs
        Use ``knn__`` prefix for KNN args, ``pca__`` prefix for PCA args.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        n_components: int = 10,
        *,
        nan_policy: NanPolicy = "drop",
        **kwargs,
    ):
        super().__init__(nan_policy=nan_policy)
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.knn_model = None
        self.pca_model = None
        self._output_columns = None
        self._train_X = None
        # Split kwargs
        self.knn_kwargs = {}
        self.pca_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith("knn__"):
                self.knn_kwargs[k[5:]] = v
            elif k.startswith("pca__"):
                self.pca_kwargs[k[5:]] = v

    def fit(self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs) -> KNNPredictorPCA:
        train_X, train_y = self._prepare_train(train_X, train_y)
        valid = train_X.notna().all(axis=1) & train_y.notna().all(axis=1)
        train_X, train_y = train_X[valid], train_y[valid]

        self.pca_model = PCA(n_components=self.n_components, **self.pca_kwargs)
        train_X_pca = self.pca_model.fit_transform(train_X)
        self.knn_model = KNeighborsRegressor(n_neighbors=self.n_neighbors, **self.knn_kwargs)
        self.knn_model.fit(train_X_pca, train_y)
        self._output_columns = train_y.columns if hasattr(train_y, "columns") else None
        self._train_X = train_X
        return self

    def predict(self, test_X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        test_X, _ = self._prepare_test(test_X)
        valid = test_X.notna().all(axis=1)

        n_rows = len(test_X)
        n_cols = len(self._output_columns) if self._output_columns is not None else 1
        preds = np.full((n_rows, n_cols), np.nan)

        if valid.any():
            test_pca = self.pca_model.transform(test_X[valid])
            y_pred = self.knn_model.predict(test_pca)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            preds[valid.values] = y_pred

        return pd.DataFrame(preds, index=test_X.index, columns=self._output_columns)

    def predict_with_confidence(
        self, test_X: pd.DataFrame, **kwargs
    ) -> tuple[pd.DataFrame, pd.Series]:
        test_X_prep, obs_frac = self._prepare_test(test_X)
        valid = test_X_prep.notna().all(axis=1)

        n_rows = len(test_X_prep)
        n_cols = len(self._output_columns) if self._output_columns is not None else 1
        preds = np.full((n_rows, n_cols), np.nan)

        if valid.any():
            test_pca = self.pca_model.transform(test_X_prep[valid])
            y_pred = self.knn_model.predict(test_pca)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            preds[valid.values] = y_pred

        result = pd.DataFrame(preds, index=test_X.index, columns=self._output_columns)
        return result, obs_frac

    def _export_state(self) -> dict:
        state = super()._export_state()
        state["n_neighbors"] = self.n_neighbors
        state["n_components"] = self.n_components
        state["knn_kwargs"] = self.knn_kwargs
        state["pca_kwargs"] = self.pca_kwargs
        return state

    def _import_state(self, state: dict) -> None:
        super()._import_state(state)
        self.n_neighbors = state.get("n_neighbors", 5)
        self.n_components = state.get("n_components", 10)
        self.knn_kwargs = state.get("knn_kwargs", {})
        self.pca_kwargs = state.get("pca_kwargs", {})
        self.knn_model = None
        self.pca_model = None
        self._output_columns = None
        self._train_X = None

    def _save_artifacts(self, dir_path: str) -> dict:
        if self.knn_model is None or self.pca_model is None:
            raise ValueError("Predictor not fitted.")
        if joblib is None:
            raise ImportError("joblib required.")
        joblib.dump(self.pca_model, os.path.join(dir_path, "pca.joblib"))
        joblib.dump(self.knn_model, os.path.join(dir_path, "knn.joblib"))
        return {"pca": "pca.joblib", "knn": "knn.joblib"}

    def _load_artifacts(
        self, dir_path: str, files: dict, manifest: dict, mmap: bool = True
    ) -> None:
        if joblib is None:
            raise ImportError("joblib required.")
        self.pca_model = joblib.load(os.path.join(dir_path, files["pca"]))
        self.knn_model = joblib.load(os.path.join(dir_path, files["knn"]))
        self._output_columns = manifest.get("output_columns")
        self._train_X = None


class KNNPredictorPCAnnoy(BasePredictor):
    """
    Fast approximate kNN regressor using Annoy, with optional PCA.

    Parameters
    ----------
    n_neighbors : int, default=5
    n_components : int or None, default=10
    n_trees : int, default=10
    search_k : int or None, default=None
    metric : str, default='euclidean'
    nan_policy : {"drop", "impute"} or float, default="drop"
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        n_components: int | None = 10,
        n_trees: int = 10,
        search_k: int | None = None,
        metric: str = "euclidean",
        *,
        nan_policy: NanPolicy = "drop",
        **kwargs,
    ):
        if AnnoyIndex is None:
            raise ImportError("KNNPredictorPCAnnoy requires 'annoy'. pip install annoy")
        super().__init__(nan_policy=nan_policy)
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

    def fit(self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs) -> KNNPredictorPCAnnoy:
        train_X, train_y = self._prepare_train(train_X, train_y)
        valid = train_X.notna().all(axis=1) & train_y.notna().all(axis=1)
        train_X, train_y = train_X[valid], train_y[valid]

        if self.n_components is not None:
            self.pca_model = PCA(n_components=self.n_components)
            vecs = self.pca_model.fit_transform(train_X)
        else:
            vecs = train_X.values

        self.annoy_index = AnnoyIndex(vecs.shape[1], self.metric)
        for i, v in enumerate(vecs):
            self.annoy_index.add_item(i, v)
        self.annoy_index.build(self.n_trees)

        self._train_X = train_X.reset_index(drop=True)
        self._train_y = train_y.reset_index(drop=True)
        self._output_columns = train_y.columns if hasattr(train_y, "columns") else None
        return self

    def predict(self, test_X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        test_X, _ = self._prepare_test(test_X)
        valid = test_X.notna().all(axis=1)

        n_rows = len(test_X)
        n_cols = len(self._output_columns) if self._output_columns is not None else 1
        preds = np.full((n_rows, n_cols), np.nan)

        if valid.any():
            test_valid = test_X[valid]
            if self.pca_model is not None:
                vecs = self.pca_model.transform(test_valid)
            else:
                vecs = test_valid.values

            for orig_idx, v in zip(test_valid.index, vecs, strict=True):
                if self.search_k is not None:
                    nn_idx = self.annoy_index.get_nns_by_vector(
                        v, self.n_neighbors, search_k=self.search_k
                    )
                else:
                    nn_idx = self.annoy_index.get_nns_by_vector(v, self.n_neighbors)
                y_pred = self._train_y.iloc[nn_idx].values.mean(axis=0)
                preds[test_X.index.get_loc(orig_idx)] = y_pred

        return pd.DataFrame(preds, index=test_X.index, columns=self._output_columns)

    def predict_with_confidence(
        self, test_X: pd.DataFrame, **kwargs
    ) -> tuple[pd.DataFrame, pd.Series]:
        test_X_prep, obs_frac = self._prepare_test(test_X)
        valid = test_X_prep.notna().all(axis=1)

        n_rows = len(test_X_prep)
        n_cols = len(self._output_columns) if self._output_columns is not None else 1
        preds = np.full((n_rows, n_cols), np.nan)

        if valid.any():
            test_valid = test_X_prep[valid]
            if self.pca_model is not None:
                vecs = self.pca_model.transform(test_valid)
            else:
                vecs = test_valid.values

            for orig_idx, v in zip(test_valid.index, vecs, strict=True):
                if self.search_k is not None:
                    nn_idx = self.annoy_index.get_nns_by_vector(
                        v, self.n_neighbors, search_k=self.search_k
                    )
                else:
                    nn_idx = self.annoy_index.get_nns_by_vector(v, self.n_neighbors)
                y_pred = self._train_y.iloc[nn_idx].values.mean(axis=0)
                preds[test_X_prep.index.get_loc(orig_idx)] = y_pred

        result = pd.DataFrame(preds, index=test_X.index, columns=self._output_columns)
        return result, obs_frac

    def _export_state(self) -> dict:
        state = super()._export_state()
        state.update(
            {
                "n_neighbors": self.n_neighbors,
                "n_components": self.n_components,
                "n_trees": self.n_trees,
                "search_k": self.search_k,
                "metric": self.metric,
            }
        )
        return state

    def _import_state(self, state: dict) -> None:
        super()._import_state(state)
        self.n_neighbors = state.get("n_neighbors", 5)
        self.n_components = state.get("n_components", None)
        self.n_trees = state.get("n_trees", 10)
        self.search_k = state.get("search_k", None)
        self.metric = state.get("metric", "euclidean")
        self.pca_model = None
        self.annoy_index = None
        self._train_X = None
        self._train_y = None
        self._output_columns = None

    def _save_artifacts(self, dir_path: str) -> dict:
        if self.annoy_index is None:
            raise ValueError("Predictor not fitted.")
        files: dict = {}
        if self.pca_model is not None:
            if joblib is None:
                raise ImportError("joblib required.")
            joblib.dump(self.pca_model, os.path.join(dir_path, "pca.joblib"))
            files["pca"] = "pca.joblib"
        self.annoy_index.save(os.path.join(dir_path, "index.ann"))
        files["ann"] = "index.ann"
        if self._train_y is None:
            raise ValueError("Predictor not fitted.")
        np.save(os.path.join(dir_path, "train_y.npy"), self._train_y.values.astype(np.float64))
        files["train_y"] = "train_y.npy"
        return files

    def _load_artifacts(
        self, dir_path: str, files: dict, manifest: dict, mmap: bool = True
    ) -> None:
        pca_file = files.get("pca")
        if pca_file is not None:
            if joblib is None:
                raise ImportError("joblib required.")
            self.pca_model = joblib.load(os.path.join(dir_path, pca_file))
        else:
            self.pca_model = None

        input_columns = manifest.get("input_columns")
        dim = (
            self.n_components
            if self.n_components is not None
            else (len(input_columns) if input_columns else None)
        )
        if dim is None:
            raise ValueError("Cannot determine Annoy index dimensionality.")
        if AnnoyIndex is None:
            raise ImportError("annoy required.")
        self.annoy_index = AnnoyIndex(dim, self.metric)
        self.annoy_index.load(os.path.join(dir_path, files["ann"]))

        mmap_mode = "r" if mmap else None
        y_arr = np.load(os.path.join(dir_path, files["train_y"]), mmap_mode=mmap_mode)
        output_columns = manifest.get("output_columns") or list(range(y_arr.shape[1]))
        self._train_y = pd.DataFrame(y_arr, columns=output_columns)
        self._output_columns = output_columns
        self._train_X = None
