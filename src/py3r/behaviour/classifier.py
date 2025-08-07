from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py3r.behaviour.features import Features


class BaseClassifier:
    def predict(self, features: Features, **kwargs):
        # User implements: extract and format X from features as needed
        raise NotImplementedError

    def fit(self, features: Features, **kwargs):
        # User implements: fit the model
        raise NotImplementedError


class KerasClassifierExample(BaseClassifier):
    """
    Example/template for using a Keras model with py3r_behaviour.
    Not intended for direct use. Copy and adapt this class outside of the py3r.behaviour package.
    Requires user to install keras and a backend (e.g., tensorflow).
    """

    def __init__(self, model, embedding_dict):
        try:
            import keras
        except ImportError:
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
        except ImportError:
            raise ImportError(
                "KerasClassifier requires keras and an appropriate backend. "
                "See https://keras.io/getting_started/"
            )
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
