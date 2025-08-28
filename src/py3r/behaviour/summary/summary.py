from __future__ import annotations
import warnings
from copy import deepcopy
from typing import Any, List

import numpy as np
import pandas as pd

from py3r.behaviour.features.features import Features
from py3r.behaviour.summary.summary_result import SummaryResult


class Summary:
    """
    stores and computes summary statistics from features objects
    """

    def __init__(self, trackingfeatures: Features) -> None:
        self.features = trackingfeatures
        self.data = dict()
        self.meta = dict()
        self.handle = trackingfeatures.handle
        if "usermeta" in trackingfeatures.meta:
            self.meta["usermeta"] = trackingfeatures.meta["usermeta"]

    def count_onset(self, column: str) -> SummaryResult:
        """
        counts number of times boolean series in the given column changes from False to True, ignoring nan values
        if first non nan value in series is true, this counts as an onset
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
        """returns time in seconds that condition in the given column is true"""
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
        """returns time in seconds that condition in the given column is false"""
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
        """
        distance_change = self.features.distance_change(point).loc[startframe:endframe]
        value = distance_change.sum()
        name = f"total_distance_{point}_{startframe}_to_{endframe}"
        meta = {
            "function": "total_distance",
            "point": point,
            "startframe": startframe,
            "endframe": endframe,
        }
        return SummaryResult(value, self, name, meta)

    def store(
        self, summarystat: Any, name: str, overwrite: bool = False, meta: Any = None
    ) -> None:
        """
        stores a summary statistic and optional metadata, with optional overwrite protection
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
        """
        if column not in self.features.data.columns:
            raise ValueError(f"Column '{column}' not found in features.data")
        states = self.features.data[column]
        time_in_state = states.value_counts() / self.features.tracking.meta["fps"]
        meta = {"function": "time_in_state", "column": column}
        return SummaryResult(time_in_state, self, f"time_in_state_{column}", meta)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.data)} summary statistics>"
