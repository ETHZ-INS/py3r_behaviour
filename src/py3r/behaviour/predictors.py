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


class BasePredictor:
    """
    Abstract base class for predictors with centralized NaN handling.

    Parameters
    ----------
    nan_policy : {"drop", "impute"} or float, default="drop"
        How to handle NaN values in input features:

        - ``"drop"``: Skip rows with any NaN during fit; return NaN for such rows
          during predict (original behavior).
        - ``"impute"``: Fill NaNs with per-column medians computed from training data.
        - ``float``: Fill NaNs with this constant value (e.g., 0.0).

    Notes
    -----
    - During fit, rows with NaN in train_y are always dropped (cannot learn from
      missing targets). The nan_policy only affects handling of NaNs in train_X.
    - When nan_policy is "impute" or a float, predict() will return predictions
      for all rows, filling NaNs before computing distances.
    - Use ``return_confidence=True`` in predict() to also get the observed_fraction
      Series indicating what fraction of features were non-NaN for each row.
    """

    def __init__(self, *, nan_policy: NanPolicy = "drop"):
        self.nan_policy = nan_policy
        self._impute_medians: pd.Series | None = None
        self._output_columns: list | None = None
        self._train_X: pd.DataFrame | None = None  # for serialization column inference

    def fit(self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs) -> BasePredictor:
        """
        Fit the predictor on training data.

        Parameters
        ----------
        train_X : pd.DataFrame
            Input features. NaN handling depends on nan_policy.
        train_y : pd.DataFrame
            Target values. Rows with any NaN are always dropped.
        **kwargs
            Additional arguments passed to _fit_impl.

        Returns
        -------
        self
        """
        # Always drop rows where train_y has NaNs (can't learn from missing targets)
        valid_y = train_y.notna().all(axis=1)

        if self.nan_policy == "drop":
            # Drop rows with any NaN in train_X or train_y
            valid_mask = train_X.notna().all(axis=1) & valid_y
            train_X_clean = train_X[valid_mask]
            train_y_clean = train_y[valid_mask]
            self._impute_medians = None
        elif self.nan_policy == "impute":
            # Compute medians from train_X before any filtering
            self._impute_medians = train_X.median()
            # Impute train_X, then filter by valid_y
            train_X_clean = train_X.fillna(self._impute_medians)[valid_y]
            train_y_clean = train_y[valid_y]
        else:
            # nan_policy is a constant float
            fill_value = float(self.nan_policy)
            self._impute_medians = pd.Series(fill_value, index=train_X.columns)
            train_X_clean = train_X.fillna(fill_value)[valid_y]
            train_y_clean = train_y[valid_y]

        self._output_columns = list(train_y.columns) if hasattr(train_y, "columns") else None
        self._train_X = train_X  # store for serialization column inference

        self._fit_impl(train_X_clean, train_y_clean, **kwargs)
        return self

    def _fit_impl(self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs) -> None:
        """
        Subclasses implement actual model fitting on clean (NaN-free) data.

        Parameters
        ----------
        train_X : pd.DataFrame
            Clean input features (no NaNs).
        train_y : pd.DataFrame
            Clean target values (no NaNs).
        **kwargs
            Additional fitting arguments.
        """
        raise NotImplementedError

    def predict(
        self,
        test_X: pd.DataFrame,
        *,
        return_confidence: bool = False,
        **kwargs,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
        """
        Predict outputs for test data.

        Parameters
        ----------
        test_X : pd.DataFrame
            Input features. NaN handling depends on nan_policy.
        return_confidence : bool, default=False
            If True, also return observed_fraction Series indicating what fraction
            of features were non-NaN for each row (1.0 = complete, <1.0 = had NaNs).
        **kwargs
            Additional arguments passed to _predict_impl.

        Returns
        -------
        pd.DataFrame or tuple[pd.DataFrame, pd.Series]
            Predictions, and optionally observed_fraction if return_confidence=True.
        """
        n_rows = len(test_X)
        n_cols = len(self._output_columns) if self._output_columns else 1

        if self.nan_policy == "drop":
            # Only predict complete rows
            valid_mask = test_X.notna().all(axis=1)
            test_X_clean = test_X[valid_mask]
            observed_frac = valid_mask.astype(np.float32)  # 1.0 or 0.0
        else:
            # Impute and predict all rows
            n_features = test_X.shape[1] if test_X.shape[1] > 0 else 1
            observed_frac = (test_X.notna().sum(axis=1) / n_features).astype(np.float32)
            if self._impute_medians is not None:
                test_X_clean = test_X.fillna(self._impute_medians)
            else:
                # Fallback: shouldn't happen if fit() was called, but be defensive
                fill_val = (
                    float(self.nan_policy) if isinstance(self.nan_policy, (int, float)) else 0.0
                )
                test_X_clean = test_X.fillna(fill_val)
            valid_mask = pd.Series(True, index=test_X.index)

        # Get predictions for clean data
        if len(test_X_clean) > 0:
            preds_clean = self._predict_impl(test_X_clean, **kwargs)
            if preds_clean.ndim == 1:
                preds_clean = preds_clean.reshape(-1, 1)
        else:
            preds_clean = np.empty((0, n_cols))

        # Reconstruct full output array
        preds = np.full((n_rows, n_cols), np.nan)
        if self.nan_policy == "drop":
            preds[valid_mask.values] = preds_clean
        else:
            preds[:] = preds_clean

        result = pd.DataFrame(preds, index=test_X.index, columns=self._output_columns)

        if return_confidence:
            return result, observed_frac
        return result

    def _predict_impl(self, test_X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Subclasses implement actual prediction on clean (NaN-free) data.

        Parameters
        ----------
        test_X : pd.DataFrame
            Clean input features (no NaNs).
        **kwargs
            Additional prediction arguments.

        Returns
        -------
        np.ndarray
            Predictions as numpy array.
        """
        raise NotImplementedError

    def fit_predict(
        self,
        train_X: pd.DataFrame,
        train_y: pd.DataFrame,
        test_X: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """Fit on training data and predict on test data."""
        self.fit(train_X, train_y, **kwargs)
        return self.predict(test_X, **kwargs)

    # --- Serialization helpers ---

    def _export_base_state(self) -> dict:
        """Export base class state for serialization."""
        state = {"nan_policy": self.nan_policy}
        if self._impute_medians is not None:
            state["impute_medians"] = self._impute_medians.to_dict()
        return state

    def _import_base_state(self, state: dict) -> None:
        """Import base class state from serialization."""
        self.nan_policy = state.get("nan_policy", "drop")
        medians_dict = state.get("impute_medians")
        if medians_dict is not None:
            self._impute_medians = pd.Series(medians_dict)
        else:
            self._impute_medians = None
        self._output_columns = None
        self._train_X = None

    def _export_state(self) -> dict:
        """
        Subclasses should override to add their own state, calling super first.
        Returns a JSON-serializable dict capturing hyperparameters and small state.
        """
        return self._export_base_state()

    def _import_state(self, state: dict) -> None:
        """
        Subclasses should override to restore their own state, calling super first.
        """
        self._import_base_state(state)

    def _save_artifacts(self, dir_path: str) -> dict:
        """
        Subclasses should write any large artifacts to dir_path and return a dict
        mapping logical names to filenames, e.g. {"pca": "pca.joblib"}.
        """
        raise NotImplementedError

    def _load_artifacts(
        self, dir_path: str, files: dict, manifest: dict, mmap: bool = True
    ) -> None:
        """
        Subclasses should load artifacts previously saved by _save_artifacts.
        """
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
        """
        Persist the trained predictor to a directory as a bundle consisting of:
        - manifest.json (generic metadata)
        - zero or more artifact files (subclass-defined)

        If input/output columns are omitted, attempts to infer from attributes
        commonly set during fit.
        """
        os.makedirs(dir_path, exist_ok=True)

        # Try to infer columns if not explicitly provided
        if input_columns is None and self._train_X is not None:
            try:
                input_columns = list(self._train_X.columns)
            except Exception:
                pass
        if output_columns is None and self._output_columns is not None:
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
        """
        Load a predictor bundle previously saved via save().
        This method assumes the predictor class matches the class on which it is called.
        """
        manifest = cls._read_manifest(dir_path)
        # Defensive: ensure caller class matches manifest's class
        manifest_class = manifest.get("predictor_class")
        if manifest_class is not None and manifest_class != cls.__name__:
            raise ValueError(
                f"Manifest was saved for {manifest_class}, but load() called on {cls.__name__}."
            )
        inst = cls.__new__(cls)
        inst._import_state(manifest.get("state", {}))
        inst._load_artifacts(dir_path, manifest.get("files", {}), manifest, mmap=mmap)
        # Restore output columns from manifest
        inst._output_columns = manifest.get("output_columns")
        return inst


class KNNPredictor(BasePredictor):
    """
    Ordinary k-Nearest Neighbors regressor predictor.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    nan_policy : {"drop", "impute"} or float, default="drop"
        How to handle NaN values. See BasePredictor for details.
    **kwargs
        Additional arguments passed to sklearn KNeighborsRegressor.
    """

    def __init__(self, n_neighbors: int = 5, *, nan_policy: NanPolicy = "drop", **kwargs):
        super().__init__(nan_policy=nan_policy)
        self.n_neighbors = n_neighbors
        self.model_kwargs = kwargs
        self.model = None

    def _fit_impl(self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs) -> None:
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, **self.model_kwargs)
        self.model.fit(train_X, train_y)

    def _predict_impl(self, test_X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.model.predict(test_X)

    def _export_state(self) -> dict:
        state = super()._export_state()
        state.update(
            {
                "n_neighbors": self.n_neighbors,
                "model_kwargs": self.model_kwargs,
            }
        )
        return state

    def _import_state(self, state: dict) -> None:
        super()._import_state(state)
        self.n_neighbors = state.get("n_neighbors", 5)
        self.model_kwargs = state.get("model_kwargs", {})
        self.model = None

    def _save_artifacts(self, dir_path: str) -> dict:
        if self.model is None:
            raise ValueError("Predictor not fitted: model is missing.")
        if joblib is None:
            raise ImportError("joblib is required to save KNNPredictor.")
        model_path = os.path.join(dir_path, "knn_model.joblib")
        joblib.dump(self.model, model_path)
        return {"model": "knn_model.joblib"}

    def _load_artifacts(
        self, dir_path: str, files: dict, manifest: dict, mmap: bool = True
    ) -> None:
        model_file = files.get("model")
        if model_file is None:
            raise ValueError("KNN model file not listed in manifest.")
        if joblib is None:
            raise ImportError("joblib is required to load KNNPredictor.")
        self.model = joblib.load(os.path.join(dir_path, model_file))


class KNNPredictorPCA(BasePredictor):
    """
    K-Nearest Neighbors regressor predictor with PCA preprocessing.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    n_components : int, default=10
        Number of PCA components.
    nan_policy : {"drop", "impute"} or float, default="drop"
        How to handle NaN values. See BasePredictor for details.
    **kwargs
        Additional arguments. Use ``knn__`` prefix for KNN args,
        ``pca__`` prefix for PCA args.
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
        # Split kwargs by prefix
        self.knn_kwargs = {}
        self.pca_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith("knn__"):
                param = k[5:]
                if param == "n_neighbors":
                    raise ValueError("Pass 'n_neighbors' as top-level arg, not 'knn__n_neighbors'.")
                self.knn_kwargs[param] = v
            elif k.startswith("pca__"):
                param = k[5:]
                if param == "n_components":
                    raise ValueError(
                        "Pass 'n_components' as top-level arg, not 'pca__n_components'."
                    )
                self.pca_kwargs[param] = v

    def _fit_impl(self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs) -> None:
        self.pca_model = PCA(n_components=self.n_components, **self.pca_kwargs)
        train_X_pca = self.pca_model.fit_transform(train_X)
        self.knn_model = KNeighborsRegressor(n_neighbors=self.n_neighbors, **self.knn_kwargs)
        self.knn_model.fit(train_X_pca, train_y)

    def _predict_impl(self, test_X: pd.DataFrame, **kwargs) -> np.ndarray:
        test_X_pca = self.pca_model.transform(test_X)
        return self.knn_model.predict(test_X_pca)

    def _export_state(self) -> dict:
        state = super()._export_state()
        state.update(
            {
                "n_neighbors": self.n_neighbors,
                "n_components": self.n_components,
                "knn_kwargs": self.knn_kwargs,
                "pca_kwargs": self.pca_kwargs,
            }
        )
        return state

    def _import_state(self, state: dict) -> None:
        super()._import_state(state)
        self.n_neighbors = state.get("n_neighbors", 5)
        self.n_components = state.get("n_components", 10)
        self.knn_kwargs = state.get("knn_kwargs", {})
        self.pca_kwargs = state.get("pca_kwargs", {})
        self.knn_model = None
        self.pca_model = None

    def _save_artifacts(self, dir_path: str) -> dict:
        if self.knn_model is None or self.pca_model is None:
            raise ValueError("Predictor not fitted: models are missing.")
        if joblib is None:
            raise ImportError("joblib is required to save KNNPredictorPCA.")
        files = {}
        pca_path = os.path.join(dir_path, "pca.joblib")
        joblib.dump(self.pca_model, pca_path)
        files["pca"] = "pca.joblib"
        knn_path = os.path.join(dir_path, "knn_model.joblib")
        joblib.dump(self.knn_model, knn_path)
        files["knn"] = "knn_model.joblib"
        return files

    def _load_artifacts(
        self, dir_path: str, files: dict, manifest: dict, mmap: bool = True
    ) -> None:
        if joblib is None:
            raise ImportError("joblib is required to load KNNPredictorPCA.")
        pca_file = files.get("pca")
        if pca_file is not None:
            self.pca_model = joblib.load(os.path.join(dir_path, pca_file))
        knn_file = files.get("knn")
        if knn_file is None:
            raise ValueError("KNN model file not listed in manifest.")
        self.knn_model = joblib.load(os.path.join(dir_path, knn_file))


class KNNPredictorPCAnnoy(BasePredictor):
    """
    Fast approximate kNN regressor using Annoy, with optional PCA preprocessing.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for regression.
    n_components : int or None, default=10
        If set, use PCA to reduce dimensionality before building Annoy index.
    n_trees : int, default=10
        Number of trees to build in Annoy index.
    search_k : int or None, default=None
        Nodes to inspect during search (higher = more accurate, slower).
    metric : str, default='euclidean'
        Distance metric for Annoy ('euclidean', 'manhattan', etc.).
    nan_policy : {"drop", "impute"} or float, default="drop"
        How to handle NaN values. See BasePredictor for details.
    **kwargs
        Ignored (for API compatibility).

    Notes
    -----
    Requires the ``annoy`` package: pip install annoy
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
            raise ImportError("KNNPredictorPCAnnoy requires 'annoy'. Install: pip install annoy")
        super().__init__(nan_policy=nan_policy)
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.n_trees = n_trees
        self.search_k = search_k
        self.metric = metric
        self.pca_model = None
        self.annoy_index = None
        self._train_y_internal: pd.DataFrame | None = None

    def _fit_impl(self, train_X: pd.DataFrame, train_y: pd.DataFrame, **kwargs) -> None:
        # Optionally fit PCA
        if self.n_components is not None:
            self.pca_model = PCA(n_components=self.n_components)
            train_X_vecs = self.pca_model.fit_transform(train_X)
        else:
            train_X_vecs = train_X.values
        # Build Annoy index
        self.annoy_index = AnnoyIndex(train_X_vecs.shape[1], self.metric)
        for i, v in enumerate(train_X_vecs):
            self.annoy_index.add_item(i, v)
        self.annoy_index.build(self.n_trees)
        # Store train_y for neighbor aggregation (reset index for iloc access)
        self._train_y_internal = train_y.reset_index(drop=True)

    def _predict_impl(self, test_X: pd.DataFrame, **kwargs) -> np.ndarray:
        # Transform with PCA if needed
        if self.pca_model is not None:
            test_X_vecs = self.pca_model.transform(test_X)
        else:
            test_X_vecs = test_X.values

        n_cols = self._train_y_internal.shape[1]
        preds = np.empty((len(test_X_vecs), n_cols))

        # For each query, get neighbors and average their train_y
        for i, v in enumerate(test_X_vecs):
            if self.search_k is not None:
                nn_idx = self.annoy_index.get_nns_by_vector(
                    v, self.n_neighbors, search_k=self.search_k
                )
            else:
                nn_idx = self.annoy_index.get_nns_by_vector(v, self.n_neighbors)
            y_neighbors = self._train_y_internal.iloc[nn_idx].values
            preds[i] = y_neighbors.mean(axis=0)

        return preds

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
        self._train_y_internal = None

    def _save_artifacts(self, dir_path: str) -> dict:
        if self.annoy_index is None:
            raise ValueError("Predictor not fitted: Annoy index is missing.")
        files: dict = {}
        # Save PCA if present
        if self.pca_model is not None:
            if joblib is None:
                raise ImportError("joblib is required to save PCA model.")
            pca_path = os.path.join(dir_path, "pca.joblib")
            joblib.dump(self.pca_model, pca_path)
            files["pca"] = "pca.joblib"
        # Save Annoy index
        ann_path = os.path.join(dir_path, "index.ann")
        self.annoy_index.save(ann_path)
        files["ann"] = "index.ann"
        # Save train_y (required at inference to compute neighbor aggregate)
        if self._train_y_internal is None:
            raise ValueError("Predictor not fitted: train_y is missing.")
        y_path = os.path.join(dir_path, "train_y.npy")
        np.save(y_path, self._train_y_internal.values.astype(np.float64))
        files["train_y"] = "train_y.npy"
        return files

    def _load_artifacts(
        self, dir_path: str, files: dict, manifest: dict, mmap: bool = True
    ) -> None:
        # Load PCA if present
        pca_file = files.get("pca")
        if pca_file is not None:
            if joblib is None:
                raise ImportError("joblib is required to load PCA model.")
            self.pca_model = joblib.load(os.path.join(dir_path, pca_file))
        else:
            self.pca_model = None
        # Rebuild Annoy index
        input_columns = manifest.get("input_columns")
        dim = (
            self.n_components
            if self.n_components is not None
            else (len(input_columns) if input_columns is not None else None)
        )
        if dim is None:
            raise ValueError("Cannot determine Annoy index dimensionality from manifest/state.")
        if AnnoyIndex is None:
            raise ImportError("annoy package is required to load Annoy index.")
        self.annoy_index = AnnoyIndex(dim, self.metric)
        ann_file = files.get("ann")
        if ann_file is None:
            raise ValueError("Annoy index file not listed in manifest files.")
        self.annoy_index.load(os.path.join(dir_path, ann_file))
        # Load train_y
        y_file = files.get("train_y")
        if y_file is None:
            raise ValueError("train_y file not listed in manifest files.")
        mmap_mode = "r" if mmap else None
        y_arr = np.load(os.path.join(dir_path, y_file), mmap_mode=mmap_mode)
        output_columns = manifest.get("output_columns")
        if output_columns is None:
            output_columns = list(range(y_arr.shape[1]))
        self._train_y_internal = pd.DataFrame(y_arr, columns=output_columns)
