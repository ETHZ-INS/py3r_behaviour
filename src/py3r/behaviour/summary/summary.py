from __future__ import annotations
import warnings
from copy import deepcopy
import os
from typing import Any, List

import numpy as np
import pandas as pd

from py3r.behaviour.features.features import Features
from py3r.behaviour.summary.summary_result import SummaryResult
from py3r.behaviour.util.io_utils import (
    SchemaVersion,
    begin_save,
    write_manifest,
    read_manifest,
    write_dataframe,
    read_dataframe,
)


class Summary:
    """
    stores and computes summary statistics from features objects
    """

    def __init__(self, trackingfeatures: Features) -> None:
        self.features = trackingfeatures
        self.data = dict()
        self.meta = dict()
        self.handle = trackingfeatures.handle
        self.tags = trackingfeatures.tags
        if "usermeta" in trackingfeatures.meta:
            self.meta["usermeta"] = trackingfeatures.meta["usermeta"]

    # Full round-trip persistence
    def save(
        self,
        dirpath: str,
        *,
        data_format: str = "parquet",
        overwrite: bool = False,
    ) -> None:
        """
        Save this Summary (including nested Features/Tracking) to a directory.

        Examples
        --------
        ```pycon
        >>> import tempfile, os, pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> # add a boolean feature for summary methods to use
        >>> mask = pd.Series([True, False] * (len(t.data)//2 + 1))[:len(t.data)]
        >>> mask.index = t.data.index
        >>> f.store(mask, 'mask', meta={})
        >>> s = Summary(f)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     s.save(d, data_format='csv', overwrite=True)
        ...     os.path.exists(os.path.join(d, 'manifest.json'))
        True

        ```
        """
        target = begin_save(dirpath, overwrite)
        # Save nested features (which saves nested tracking)
        features_sub = os.path.join(target, "features")
        self.features.save(features_sub, data_format=data_format, overwrite=True)
        # Save summary data: scalars inline; pandas as separate files
        data_dir = os.path.join(target, "data")
        os.makedirs(data_dir, exist_ok=True)
        scalars = {}
        frames = {}
        for name, value in self.data.items():
            if isinstance(value, pd.DataFrame):
                spec = write_dataframe(
                    data_dir,
                    value,
                    filename=f"{name}.parquet"
                    if data_format == "parquet"
                    else f"{name}.csv",
                    format=data_format,
                )
                frames[name] = {"type": "dataframe", **spec, "subdir": "data"}
            elif isinstance(value, pd.Series):
                spec = write_dataframe(
                    data_dir,
                    value,
                    filename=f"{name}.parquet"
                    if data_format == "parquet"
                    else f"{name}.csv",
                    format=data_format,
                )
                frames[name] = {"type": "series", **spec, "subdir": "data"}
            else:
                scalars[name] = value
        manifest = {
            "schema_version": SchemaVersion,
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
            "handle": self.handle,
            "tags": self.tags,
            "meta": self.meta,
            "features_path": "features",
            "scalars": scalars,
            "frames": frames,
        }
        write_manifest(target, manifest)

    @classmethod
    def load(cls, dirpath: str) -> "Summary":
        """
        Load a Summary previously saved with save().

        Examples
        --------
        ```pycon
        >>> import tempfile, pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> f.store(pd.Series(range(len(t.data)), index=t.data.index), 'counter', meta={})
        >>> s = Summary(f)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     s.save(d, data_format='csv', overwrite=True)
        ...     s2 = Summary.load(d)
        >>> isinstance(s2, Summary) and 'counter' in s2.features.data.columns
        True

        ```
        """
        manifest = read_manifest(dirpath)
        features = Features.load(os.path.join(dirpath, manifest["features_path"]))
        obj = cls(features)
        obj.meta = manifest.get("meta", {})
        obj.handle = manifest.get("handle", obj.handle)
        obj.tags = manifest.get("tags", obj.tags)
        # restore data
        obj.data = {}
        scalars = manifest.get("scalars", {})
        frames = manifest.get("frames", {})
        obj.data.update(scalars)
        for name, spec in frames.items():
            subdir = spec.get("subdir", "")
            base = os.path.join(dirpath, subdir) if subdir else dirpath
            df = read_dataframe(base, spec)
            if spec.get("type") == "series":
                obj.data[name] = df.iloc[:, 0]
            else:
                obj.data[name] = df
        return obj

    def count_onset(self, column: str) -> SummaryResult:
        """
        counts number of times boolean series in the given column changes from False to True, ignoring nan values
        if first non nan value in series is true, this counts as an onset
        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> mask = pd.Series([False, True, False, True, False][:len(t.data)], index=t.data.index)
        >>> f.store(mask, 'mask', meta={})
        >>> s = Summary(f)
        >>> res = s.count_onset('mask')
        >>> bool(res.value == 2)
        True

        ```
        """
        if column not in self.features.data.columns:
            raise ValueError(f"Column '{column}' not found in features.data")
        series = self.features.data[column]
        nonan = pd.Series(list(series.dropna()))
        if nonan.dtype != "bool":
            raise Exception("count_onset requires boolean series as input")
        count = (nonan & (nonan != nonan.shift(-1))).sum()
        meta = {"function": "count_onset", "column": column}
        return SummaryResult(count, self, f"count_onset_{column}", meta)

    def time_true(self, column: str) -> SummaryResult:
        """
        returns time in seconds that condition in the given column is true

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> mask = pd.Series([True] * len(t.data), index=t.data.index)
        >>> f.store(mask, 'mask', meta={})
        >>> s = Summary(f)
        >>> res = s.time_true('mask')
        >>> bool(res.value > 0)
        True

        ```
        """
        if column not in self.features.data.columns:
            raise ValueError(f"Column '{column}' not found in features.data")
        series = self.features.data[column]
        nonan = pd.Series(list(series.dropna()))
        if nonan.dtype != "bool":
            raise Exception("time_true requires boolean series as input")
        time = nonan.sum() / self.features.tracking.meta["fps"]
        meta = {"function": "time_true", "column": column}
        return SummaryResult(time, self, f"time_true_{column}", meta)

    def time_false(self, column: str) -> SummaryResult:
        """
        returns time in seconds that condition in the given column is false

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> mask = pd.Series([False] * len(t.data), index=t.data.index)
        >>> f.store(mask, 'mask', meta={})
        >>> s = Summary(f)
        >>> res = s.time_false('mask')
        >>> bool(res.value > 0)
        True

        ```
        """
        if column not in self.features.data.columns:
            raise ValueError(f"Column '{column}' not found in features.data")
        series = self.features.data[column]
        nonan = pd.Series(list(series.dropna()))
        if nonan.dtype != "bool":
            raise Exception("time_true requires boolean series as input")
        time = (~nonan).sum() / self.features.tracking.meta["fps"]
        meta = {"function": "time_false", "column": column}
        return SummaryResult(time, self, f"time_false_{column}", meta)

    def total_distance(
        self, point: str, startframe: int | None = None, endframe: int | None = None
    ) -> SummaryResult:
        """
        returns total distance traveled by a tracked point between optional start and end frames
        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> s = Summary(f)
        >>> res = s.total_distance('p1', 0, 4)
        >>> isinstance(res.value, (int, float))
        True

        ```
        """
        # Slice uses None gracefully to indicate full-range on that side
        distance_change = self.features.distance_change(point).loc[startframe:endframe]
        value = distance_change.sum()

        # Build name: include frames only if explicitly provided by the caller
        if startframe is not None and endframe is not None:
            name = f"total_distance_{point}_{startframe}_to_{endframe}"
        elif startframe is not None:
            name = f"total_distance_{point}_from_{startframe}"
        elif endframe is not None:
            name = f"total_distance_{point}_to_{endframe}"
        else:
            name = f"total_distance_{point}"

        meta = {
            "function": "total_distance",
            "point": point,
            "startframe": startframe,
            "endframe": endframe,
        }
        return SummaryResult(value, self, name, meta)

    def _apply_column(self, column: str, func, **kwargs) -> SummaryResult:
        """
        Internal method to apply aggregation function to a column
        """
        if column not in self.features.data.columns:
            raise ValueError(f"Column '{column}' not found in features.data")

        if callable(func):
            value = func(self.features.data[column], **kwargs)
            meta = {"function": f"{func.__name__}_column", "column": column}
            return SummaryResult(value, self, f"{func.__name__}_{column}", meta)

        raise TypeError("func must be callable.")


    def sum_column(self, column: str) -> SummaryResult:
        """
        Sum all non-NaN values in a `features.data` column and return as a SummaryResult.

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> s = pd.Series([1, 2, 3, 4, 5][:len(t.data)], index=t.data.index)
        >>> f.store(s, 'x', meta={})
        >>> summ = Summary(f)
        >>> res = summ.sum_column('x')
        >>> bool(res.value == 15)
        True

        ```
        """
        return self._apply_column(column, pd.Series.sum, skipna=True)

    def mean_column(self, column: str) -> SummaryResult:
        """
        Mean of all non-NaN values in a `features.data` column and return as a SummaryResult.

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> s = pd.Series([1, 2, 3, 4, 5][:len(t.data)], index=t.data.index)
        >>> f.store(s, 'x', meta={})
        >>> summ = Summary(f)
        >>> res = summ.mean_column('x')
        >>> bool(res.value == 3)
        True

        ```
        """
        return self._apply_column(column, pd.Series.mean, skipna=True)

    def median_column(self, column: str) -> SummaryResult:
        """
        Median of all non-NaN values in a `features.data` column and return as a SummaryResult.

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> s = pd.Series([1, 2, 3, 4, 5][:len(t.data)], index=t.data.index)
        >>> f.store(s, 'x', meta={})
        >>> summ = Summary(f)
        >>> res = summ.median_column('x')
        >>> bool(res.value == 3)
        True

        ```
        """
        return self._apply_column(column, pd.Series.median, skipna=True)

    def max_column(self, column: str) -> SummaryResult:
        """
        Max of all non-NaN values in a `features.data` column and return as a SummaryResult.

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> s = pd.Series([1, 2, 3, 4, 5][:len(t.data)], index=t.data.index)
        >>> f.store(s, 'x', meta={})
        >>> summ = Summary(f)
        >>> res = summ.max_column('x')
        >>> bool(res.value == 5)
        True

        ```
        """
        return self._apply_column(column, pd.Series.max, skipna=True)

    def min_column(self, column: str) -> SummaryResult:
        """
        Min of all non-NaN values in a `features.data` column and return as a SummaryResult.

        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> s = pd.Series([1, 2, 3, 4, 5][:len(t.data)], index=t.data.index)
        >>> f.store(s, 'x', meta={})
        >>> summ = Summary(f)
        >>> res = summ.min_column('x')
        >>> bool(res.value == 1)
        True

        ```
        """
        return self._apply_column(column, pd.Series.min, skipna=True)

    def store(
        self, summarystat: Any, name: str, overwrite: bool = False, meta: Any = None
    ) -> None:
        """
        stores a summary statistic and optional metadata, with optional overwrite protection
        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> s = Summary(f)
        >>> s.store(123, 'num_events', meta={'unit':'count'})
        >>> s.data['num_events'] == 123 and s.meta['num_events']['unit'] == 'count'
        True

        ```
        """
        if name in self.data:
            if overwrite:
                self.data[name] = summarystat
                warnings.warn(f"summarystat {name} overwritten")
            else:
                raise Exception(
                    f"summarystat with name {name} already stored. set overwrite=True to overwrite"
                )
        else:
            self.data[name] = summarystat
        self.meta[name] = meta

    def make_bin(self, startframe: int, endframe: int) -> "Summary":
        """
        creates a copy of the Summary object with the dataframes
        restricted from startframe to endframe, inclusive
        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> s = Summary(f)
        >>> b = s.make_bin(0, 2)
        >>> isinstance(b, Summary)
        True

        ```
        """
        # make deep copy of the Summary object
        bin_out = deepcopy(self)

        # trim the tracking dataframe
        bin_out.features.tracking.data = self.features.tracking.data.loc[
            startframe:endframe
        ].copy()

        # trim the features dataframe
        bin_out.features.data = self.features.data.loc[startframe:endframe].copy()

        # clear data and meta
        bin_out.data = dict()
        bin_out.meta = dict()

        return bin_out

    def make_bins(self, numbins: int) -> List[Summary]:
        """
        creates a list of Summary objects, with frames restricted into
        numbins even intervals.
        start/endpoints are duplicated between intervals to ensure no loss
        in e.g. distance calculations
        Examples
        --------
        ```pycon
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> s = Summary(f)
        >>> bins = s.make_bins(3)
        >>> len(bins) == 3 and all(isinstance(b, Summary) for b in bins)
        True

        ```
        """
        startframe = self.features.tracking.data.index.values[0]
        endframe = self.features.tracking.data.index.values[-1]

        binboundaries = np.linspace(startframe, endframe, numbins + 1).astype(int)

        out = [
            self.make_bin(binboundaries[i], binboundaries[i + 1])
            for i in range(numbins)
        ]

        return out

    def transition_matrix(self, column: str, all_states=None) -> SummaryResult:
        """
        Returns a transition matrix for a given column in self.features.data,
        with rows and columns as the unique values of the column or as specified by all_states.
        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> states = pd.Series(['A','A','B','B','A'][:len(t.data)], index=t.data.index)
        >>> f.store(states, 'state', meta={})
        >>> s = Summary(f)
        >>> res = s.transition_matrix('state')
        >>> isinstance(res.value, pd.DataFrame)
        True

        ```
        """
        if column not in self.features.data.columns:
            raise ValueError(f"Column '{column}' not found in features.data")
        states = self.features.data[column]
        transitions = states != states.shift()
        prev_states = states.shift()[transitions]
        curr_states = states[transitions]
        trans_df = pd.DataFrame(
            {"previous": prev_states, "current": curr_states}
        ).dropna()
        if all_states is None:
            all_states = pd.unique(states.dropna())
        transition_matrix = pd.crosstab(
            trans_df["previous"], trans_df["current"], dropna=False
        ).reindex(index=all_states, columns=all_states, fill_value=0)
        meta = {
            "function": "transition_matrix",
            "column": column,
            "all_states": all_states,
        }
        return SummaryResult(
            transition_matrix, self, f"transition_matrix_{column}", meta
        )

    def count_state_onsets(self, column: str) -> SummaryResult:
        """
        counts the number of times a state is entered in a given column
        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> states = pd.Series(['A','A','B','B','A'][:len(t.data)], index=t.data.index)
        >>> f.store(states, 'state', meta={})
        >>> s = Summary(f)
        >>> res = s.count_state_onsets('state')
        >>> hasattr(res, 'value')
        True

        ```
        """
        if column not in self.features.data.columns:
            raise ValueError(f"Column '{column}' not found in features.data")
        states = self.features.data[column]
        transitions = states != states.shift()
        transition_states = states[transitions]
        state_counts = transition_states.value_counts()
        meta = {"function": "count_state_onsets", "column": column}
        return SummaryResult(state_counts, self, f"count_state_onsets_{column}", meta)

    def time_in_state(self, column: str) -> SummaryResult:
        """
        returns the time spent in each state in a given column
        Examples
        --------
        ```pycon
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking import Tracking
        >>> from py3r.behaviour.features.features import Features
        >>> from py3r.behaviour.summary.summary import Summary
        >>> with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...     t = Tracking.from_dlc(str(p), handle='ex', fps=30)
        >>> f = Features(t)
        >>> states = pd.Series(['A','A','B','B','A'][:len(t.data)], index=t.data.index)
        >>> f.store(states, 'state', meta={})
        >>> s = Summary(f)
        >>> res = s.time_in_state('state')
        >>> hasattr(res, 'value')
        True

        ```
        """
        if column not in self.features.data.columns:
            raise ValueError(f"Column '{column}' not found in features.data")
        states = self.features.data[column]
        time_in_state = states.value_counts() / self.features.tracking.meta["fps"]
        meta = {"function": "time_in_state", "column": column}
        return SummaryResult(time_in_state, self, f"time_in_state_{column}", meta)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.data)} summary statistics>"
