from __future__ import annotations
import pandas as pd


class FeaturesResult(pd.Series):
    def __init__(self, series, features_obj, column_name, params):
        super().__init__(series)
        self._features_obj = features_obj
        self._column_name = column_name
        self._params = params
        self.name = column_name  # Set the Series name for plotting/legend

    def store(self, name=None, meta=None, overwrite=False):
        if name is None:
            name = self._column_name
        if meta is None:
            meta = self._params
        self._features_obj.store(self, name, overwrite=overwrite, meta=meta)
        return name

    @property
    def _constructor(self):
        # Ensures pandas operations return pd.Series, not FeaturesResult
        return pd.Series
