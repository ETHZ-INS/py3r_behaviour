from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from py3r.behaviour.features import Features


class BaseClassifier:
    def predict(self, features: Features, **kwargs):
        # User implements: extract and format X from features as needed
        raise NotImplementedError

    def fit(self, features: Features, **kwargs):
        # User implements: fit the model
        raise NotImplementedError


class OnnxClassifier(BaseClassifier):
    """Run a pre-trained ONNX model on a Features embedding.

    Args:
        model_path: Path to the ``.onnx`` model file.
        embedding_dict: Mapping of feature column name to time shifts, passed
            directly to :meth:`Features.embedding_df`. Determines the input
            columns and their temporal context fed to the model.

    Requires ``onnxruntime`` (``pip install onnxruntime``). The model must
    have been exported to ONNX from the original training framework (Keras,
    PyTorch, scikit-learn, etc.) before use.
    """

    def __init__(self, model_path: str | Path, embedding_dict: dict[str, list[int]]):
        try:
            import onnxruntime as ort
        except ImportError as err:
            raise ImportError(
                "OnnxClassifier requires onnxruntime. pip install onnxruntime"
            ) from err
        self.embedding_dict = embedding_dict
        self._session = ort.InferenceSession(str(model_path))
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, features: Features, **kwargs) -> pd.Series | pd.DataFrame:
        """Run inference on the embedding derived from *features*.

        Args:
            features: A fitted :class:`~py3r.behaviour.features.Features`
                instance containing the columns referenced by ``embedding_dict``.
            **kwargs: Unused; accepted for interface compatibility.

        Returns:
            A :class:`pd.Series` (single output) or :class:`pd.DataFrame`
                (multiple outputs) indexed identically to the embedding, so
                results align with the original Features data.
        """
        df = features.embedding_df(self.embedding_dict)
        X = df.values.astype(np.float32)
        raw = self._session.run(None, {self._input_name: X})[0]
        if raw.ndim == 1:
            return pd.Series(raw, index=df.index)
        return pd.DataFrame(raw, index=df.index)

    def fit(self, features: Features, **kwargs):
        raise NotImplementedError(
            "OnnxClassifier is inference-only. Train and export the model to ONNX "
            "using your training framework (e.g. tf2onnx for Keras, torch.onnx.export "
            "for PyTorch) before loading it here."
        )


class KerasClassifierExample(BaseClassifier):
    """
    Example/template for using a Keras model with py3r_behaviour.
    Not intended for direct use. Copy and adapt this class outside of the py3r.behaviour package.
    Requires user to install keras and a backend (e.g., tensorflow).
    """

    def __init__(self, model, embedding_dict):
        import importlib.util

        if importlib.util.find_spec("keras") is None:
            raise ImportError(
                "KerasClassifierExample requires keras and an appropriate backend. "
                "See https://keras.io/getting_started/"
            )
        if not hasattr(model, "predict"):
            raise ValueError("model must be a Keras model with a .predict() method")

        self.model = model
        self.embedding_dict = embedding_dict

    @classmethod
    def from_file(cls, filepath, embedding_dict):
        try:
            from keras.models import load_model
        except ImportError as err:
            raise ImportError(
                "KerasClassifier requires keras and an appropriate backend. "
                "See https://keras.io/getting_started/"
            ) from err
        model = load_model(filepath)
        return cls(model, embedding_dict)

    def predict(self, features: Features, **kwargs):
        # User controls how to extract X
        X = features.embedding_df(self.embedding_dict).values
        # User controls any further formatting of input to model.predict
        raise NotImplementedError
        base_result = self.model.predict(X, **kwargs)
        # restructure base_result to be a pd.Series with the same index as Features.data
        return base_result

    def fit(self, features: Features, **kwargs):
        # User implements: fit the model
        raise NotImplementedError
