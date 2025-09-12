from __future__ import annotations

import numpy as np
import pandas as pd

from py3r.behaviour.summary.summary_collection import SummaryCollection
from py3r.behaviour.summary.summary import Summary
from py3r.behaviour.summary.summary_result import SummaryResult
from py3r.behaviour.features.multiple_features_collection import (
    MultipleFeaturesCollection,
)
from py3r.behaviour.util.base_collection import BaseMultipleCollection


class MultipleSummaryCollection(BaseMultipleCollection):
    """
    collection of SummaryCollection objects
    (e.g. for comparison between groups)
    """

    _element_type = SummaryCollection
    _multiple_collection_type = "MultipleSummaryCollection"

    def __init__(self, dict_of_summary_collections: dict[str, SummaryCollection]):
        super().__init__(dict_of_summary_collections)

    @property
    def dict_of_summary_collections(self):
        return self._obj_dict

    @classmethod
    def from_multiple_features_collection(
        cls,
        multiple_features_collection: MultipleFeaturesCollection,
        summary_cls=Summary,
    ):
        """
        creates a MultipleSummaryCollections from a MultipleFeaturesCollection
        """
        multiple_summary_collection = {}
        for (
            handle,
            features_collection,
        ) in multiple_features_collection.features_collections.items():
            multiple_summary_collection[handle] = (
                SummaryCollection.from_features_collection(
                    features_collection, summary_cls=summary_cls
                )
            )
        return cls(multiple_summary_collection)

    def make_bin(self, startframe, endframe):
        """returns a new MultipleSummaryCollection with binned summaries"""
        binned = {
            k: v.make_bin(startframe, endframe)
            for k, v in self.dict_of_summary_collections.items()
        }
        return MultipleSummaryCollection(binned)

    def make_bins(self, numbins):
        """returns a list of MultipleSummaryCollection, one per bin"""
        bins = {
            k: v.make_bins(numbins) for k, v in self.dict_of_summary_collections.items()
        }  # {group: [SummaryCollection, ...]}
        nbins = len(next(iter(bins.values())))
        return [
            MultipleSummaryCollection({k: bins[k][i] for k in bins})
            for i in range(nbins)
        ]

    def bfa(self, column: str, all_states=None, numshuffles: int = 1000):
        from itertools import combinations

        # batch calculate transition matrix for each summary object
        transition_matrices_result = self.transition_matrix(column, all_states)
        # Extract the .value from each SummaryResult in the nested dict
        transition_matrices = {
            group: {k: v.value for k, v in d.items()}
            for group, d in transition_matrices_result.items()
        }
        # calculate manhattan distance for each group pair
        distances = {}
        for group1, group2 in combinations(self.dict_of_summary_collections.keys(), 2):
            _ = {}
            list1 = list(transition_matrices[group1].values())
            list2 = list(transition_matrices[group2].values())
            _["observed"] = self._manhattan_distance_twogroups(list1, list2)
            _["surrogates"] = [
                self._manhattan_distance_twogroups(*self.shuffle_lists(list1, list2))
                for i in range(numshuffles)
            ]
            distances[group1 + "_vs_" + group2] = _
        return distances

    @staticmethod
    def bfa_stats(
        bfa_results: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        def percentile(observed: float, surrogates: list[float]) -> float:
            return sum(observed > pd.Series(surrogates)) / (len(surrogates) + 1)

        def zscore(observed: float, surrogates: list[float]) -> float:
            return (observed - np.mean(surrogates)) / np.std(surrogates)

        def right_tail_p(observed: float, surrogates: list[float]) -> float:
            from math import erf

            return 0.5 * (1 - erf(zscore(observed, surrogates) / np.sqrt(2)))

        stats = {}
        for group, result in bfa_results.items():
            observed = result["observed"]
            surrogates = result["surrogates"]
            stats[group] = {
                "percentile": percentile(observed, surrogates),
                "zscore": zscore(observed, surrogates),
                "right_tail_p": right_tail_p(observed, surrogates),
            }
        return stats

    @staticmethod
    def _manhattan_distance(
        transition_matrix1: pd.DataFrame, transition_matrix2: pd.DataFrame
    ) -> float:
        # check that transition_matrix1 and transition_matrix2 have the same index and columns
        if not transition_matrix1.index.equals(transition_matrix2.index):
            raise ValueError(
                "transition_matrix1 and transition_matrix2 must have the same index"
            )
        if not transition_matrix1.columns.equals(transition_matrix2.columns):
            raise ValueError(
                "transition_matrix1 and transition_matrix2 must have the same columns"
            )
        difference = transition_matrix1 - transition_matrix2
        return difference.abs().sum(axis=1).sum()

    @staticmethod
    def _mean_transition_matrix(matrices: list[pd.DataFrame]) -> pd.DataFrame:
        summed_matrix = sum(matrices)
        mean_matrix = summed_matrix / len(matrices)
        return mean_matrix

    def _manhattan_distance_twogroups(
        self, list1: list[pd.DataFrame], list2: list[pd.DataFrame]
    ) -> float:
        # calculate manhattan distance between two lists of transition matrices
        distance = self._manhattan_distance(
            self._mean_transition_matrix(list1), self._mean_transition_matrix(list2)
        )
        return distance

    @staticmethod
    def shuffle_lists(group1: list, group2: list) -> tuple[list, list]:
        import random

        n1 = len(group1)
        combined = group1 + group2
        random.shuffle(combined)
        new_group1 = combined[:n1]
        new_group2 = combined[n1:]
        return new_group1, new_group2

    def store(
        self,
        results_dict: dict[str, dict[str, SummaryResult]],
        name: str = None,
        meta: dict = None,
        overwrite: bool = False,
    ):
        """
        Store all SummaryResult objects in a two-layer dict (as returned by batch methods).
        Example:
            results = multiple_summary_collection.time_true('is_running')
            multiple_summary_collection.store(results)
        """
        for group_dict in results_dict.values():
            for v in group_dict.values():
                if hasattr(v, "store"):
                    v.store(name=name, meta=meta, overwrite=overwrite)
                else:
                    raise ValueError(f"{v} is not a SummaryResult object")
