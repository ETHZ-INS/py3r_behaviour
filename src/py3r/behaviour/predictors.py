import os
import json
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
import numpy as np

try:
    import joblib
except Exception:  # joblib is a transitive dep of sklearn, but be defensive
    joblib = None

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

    # --- Serialization helpers (generic manifest with subclass hooks) ---
    def _export_state(self) -> dict:
        """
        Subclasses should return a JSON-serializable dict capturing hyperparameters
        and any small state needed to reconstruct the predictor, excluding large
        artifacts saved via _save_artifacts.
        """
        raise NotImplementedError

    def _import_state(self, state: dict) -> None:
        """
        Subclasses should restore internal parameters from the provided state dict.
        """
        raise NotImplementedError

    def _save_artifacts(self, dir_path: str) -> dict:
        """
        Subclasses should write any large artifacts to dir_path and return a dict
        mapping logical names to filenames, e.g. {"pca": "pca.joblib", "ann": "index.ann"}.
        """
        raise NotImplementedError

    def _load_artifacts(
        self, dir_path: str, files: dict, manifest: dict, mmap: bool = True
    ) -> None:
        """
        Subclasses should load artifacts previously saved by _save_artifacts and
        attach them to self. The manifest is provided for context (e.g. input/output columns).
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
        commonly set during fit (self._train_X.columns and self._output_columns).
        """
        os.makedirs(dir_path, exist_ok=True)

        # Try to infer columns if not explicitly provided
        if (
            input_columns is None
            and hasattr(self, "_train_X")
            and getattr(self, "_train_X") is not None
        ):
            try:
                input_columns = list(self._train_X.columns)
            except Exception:
                pass
        if (
            output_columns is None
            and hasattr(self, "_output_columns")
            and getattr(self, "_output_columns") is not None
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
        with open(os.path.join(dir_path, "manifest.json"), "r") as f:
            return json.load(f)

    @classmethod
    def load(cls, dir_path: str, mmap: bool = True) -> "BasePredictor":
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
        # Ensure __init__ isn't required for these attrs; _import_state will set parameters
        inst._import_state(manifest.get("state", {}))
        inst._load_artifacts(dir_path, manifest.get("files", {}), manifest, mmap=mmap)
        return inst


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

    # ---- Serialization hooks ----
    def _export_state(self) -> dict:
        return {
            "n_neighbors": self.n_neighbors,
            "n_components": self.n_components,
            "n_trees": self.n_trees,
            "search_k": self.search_k,
            "metric": self.metric,
        }

    def _import_state(self, state: dict) -> None:
        # Recreate minimal parameter state; artifacts loaded separately
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
        # Recreate AnnoyIndex to ensure save works (existing object is fine)
        self.annoy_index.save(ann_path)
        files["ann"] = "index.ann"
        # Save train_y (required at inference to compute neighbor aggregate)
        if self._train_y is None:
            raise ValueError("Predictor not fitted: train_y is missing.")
        y_path = os.path.join(dir_path, "train_y.npy")
        # Store as float32 to balance size/precision
        np.save(y_path, self._train_y.values.astype(np.float64))
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
            raise ValueError(
                "Cannot determine Annoy index dimensionality from manifest/state."
            )
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
            # Fallback: numeric columns
            output_columns = list(range(y_arr.shape[1]))
        # Wrap memmap/ndarray as DataFrame for convenient iloc/values behavior
        self._train_y = pd.DataFrame(y_arr, columns=output_columns)
        self._output_columns = output_columns
