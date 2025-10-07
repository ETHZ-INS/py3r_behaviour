from __future__ import annotations
import json
import os
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

    def save(self, filepath: str) -> None:
        """
        Save the feature to a CSV alongside a JSON metadata file.

        The CSV contains the series with its name as header. A companion
        `<base>_meta.json` is written next to it containing the stored params.

        Examples:
            >>> import tempfile, os, pandas as pd
            >>> class DummyFeatures:
            ...     def __init__(self):
            ...         self.meta = {}
            ...     def store(self, s, name, overwrite=False, meta=None):
            ...         self.meta[name] = meta or {}
            ...
            >>> fr = FeaturesResult(pd.Series([1, 2, 3], index=[0,1,2]), DummyFeatures(), 'feat_a', {'k': 'v'})
            >>> with tempfile.TemporaryDirectory() as d:
            ...     path = os.path.join(d, 'feat_a.csv')
            ...     fr.save(path)
            ...     os.path.exists(path) and os.path.exists(path.replace('.csv', '_meta.json'))
            True
        """
        base = filepath[:-4] if filepath.endswith(".csv") else filepath
        csv_path = base + ".csv"
        meta_path = base + "_meta.json"

        # Ensure the Series name is set for a proper CSV header
        series_to_save = self.copy()
        series_to_save.name = self._column_name
        series_to_save.to_csv(os.path.expanduser(csv_path))

        # Prefer stored meta if available, otherwise fall back to params
        meta = None
        try:
            meta = self._features_obj.meta.get(self._column_name)
        except Exception:
            meta = None
        if meta is None:
            meta = self._params

        with open(os.path.expanduser(meta_path), "w") as f:
            json.dump(meta, f)

    @property
    def _constructor(self):
        # Ensures pandas operations return pd.Series, not FeaturesResult
        return pd.Series
