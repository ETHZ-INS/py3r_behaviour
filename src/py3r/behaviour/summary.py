from __future__ import annotations
from copy import deepcopy
import pandas as pd
import numpy as np
import warnings
import logging
import sys
from typing import Any, List
from py3r.behaviour.features import Features, FeaturesCollection, MultipleFeaturesCollection
from py3r.behaviour.exceptions import BatchProcessError


logger = logging.getLogger(__name__)
logformat = '%(funcName)s(): %(message)s'
logging.basicConfig(stream=sys.stdout, format=logformat)
logger.setLevel(logging.INFO)

class Summary():
    '''
    stores and computes summary statistics from features objects
    '''
    def __init__(self, trackingfeatures:Features) -> None:
        self.features = trackingfeatures
        self.data = dict()
        self.meta = dict()
        self.handle = trackingfeatures.handle
        if 'usermeta' in trackingfeatures.meta:
            self.meta['usermeta'] = trackingfeatures.meta['usermeta']

    @staticmethod
    def count_onset(series:pd.Series) -> int:
        '''
        counts number of times boolean series changes from False to True, ignoring nan values
        if first non nan value in series is true, this counts as an onset
        '''
        nonan = pd.Series(list(series.dropna()))

        if nonan.dtype != 'bool':
            raise Exception('count_onset requires boolean series as input')
        
        count = (nonan & (nonan != nonan.shift(-1))).sum()
        return count
        
    def time_true(self, series:pd.Series) -> float:
        '''returns time in seconds that condition is true'''

        nonan = pd.Series(list(series.dropna()))

        if nonan.dtype != 'bool':
            raise Exception('time_true requires boolean series as input')
        
        time = nonan.sum() / self.features.tracking.meta['fps']
        return time
    
    def total_distance(self, point:str, startframe: int|None = None, endframe: int|None = None) -> float:
        '''
        returns total distance traveled by a tracked point between optional start and end frames
        '''
        distance_change = self.features.distance_change(point).loc[startframe:endframe]
        return distance_change.sum()
    
    def store(self, summarystat: Any, name: str, overwrite: bool = False, meta: Any = None) -> None:
        '''
        stores a summary statistic and optional metadata, with optional overwrite protection
        '''
        if name in self.data:
            if overwrite:
                self.data[name] = summarystat
                warnings.warn('summarystat \''+name+'\' overwritten')
            else:
                raise Exception('summarystat with name \''+name+'\' already stored. set overwrite=True to overwrite')
        else:
            self.data[name] = summarystat

        self.meta[name] = meta

    def make_bin(self, startframe:int, endframe:int) -> 'Summary':
        '''
        creates a copy of the Summary object with the dataframes 
        restricted from startframe to endframe, inclusive
        '''
        # make deep copy of the Summary object
        bin_out = deepcopy(self)

        #trim the tracking dataframe
        bin_out.features.tracking.data = self.features.tracking.data.loc[startframe:endframe].copy()

        #trim the features dataframe
        bin_out.features.data = self.features.data.loc[startframe:endframe].copy()

        #clear data and meta
        bin_out.data = dict()
        bin_out.meta = dict()
        
        return(bin_out)
    
    def make_bins(self, numbins: int) -> List[Summary]:
        '''
        creates a list of Summary objects, with frames restricted into
        numbins even intervals. 
        start/endpoints are duplicated between intervals to ensure no loss
        in e.g. distance calculations
        '''
        startframe = self.features.tracking.data.index.values[0]
        endframe = self.features.tracking.data.index.values[-1]

        binboundaries = np.linspace(startframe, endframe, numbins+1).astype(int)

        out = [self.make_bin(binboundaries[i],binboundaries[i+1]) for i in range(numbins)]
        
        return(out)

    def transition_matrix(self, column: str, all_states=None) -> pd.DataFrame:
        '''
        Returns a transition matrix for a given column in self.features.data,
        with rows and columns as the unique values of the column or as specified by all_states.
        '''
        if column not in self.features.data.columns:
            raise ValueError(f"Column '{column}' not found in features.data")
        
        states = self.features.data[column]
        transitions = states != states.shift()
        prev_states = states.shift()[transitions]
        curr_states = states[transitions]
        trans_df = pd.DataFrame({'previous': prev_states, 'current': curr_states}).dropna()

        # Use provided all_states or infer from data
        if all_states is None:
            all_states = pd.unique(states.dropna())

        # Build the transition matrix with all possible clusters as rows/columns, fill missing with 0
        transition_matrix = pd.crosstab(
            trans_df['previous'],
            trans_df['current'],
            dropna=False
        ).reindex(index=all_states, columns=all_states, fill_value=0)

        return transition_matrix
    
    def count_state_onsets(self, column: str) -> pd.Series:
        '''
        counts the number of times a state is entered in a given column
        '''
        #check if column is in features.data
        if column not in self.features.data.columns:
            raise ValueError(f"Column '{column}' not found in features.data")
        
        states = self.features.data[column]
        transitions = states != states.shift()
        transition_states = states[transitions]
        # count occurrences of each state in transition_states
        state_counts = transition_states.value_counts()
        return state_counts
    
    def time_in_state(self, column: str) -> pd.Series:
        '''
        returns the time spent in each state in a given column
        '''
        #check if column is in features.data
        if column not in self.features.data.columns:
            raise ValueError(f"Column '{column}' not found in features.data")
        
        states = self.features.data[column]
        return states.value_counts() / self.features.tracking.meta['fps']

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.data)} summary statistics>"
    

class SummaryCollection:
    '''
    collection of Summary objects
    (e.g. for grouping individuals)
    note: type-hints refer to Summary, but factory methods allow for other classes
    these are intended ONLY for subclasses of Summary, and this is enforced
    '''
    summary_dict: dict[str, Summary]
    
    def __init__(self, summary_dict: dict[str, Summary]):
        self.summary_dict = summary_dict

    def __getattr__(self, name):
        # Only called if the attribute is not found on the collection itself
        def batch_method(*args, **kwargs):
            results = {}
            for key, obj in self.summary_dict.items():
                try:
                    method = getattr(obj, name)
                    results[key] = method(*args, **kwargs)
                except Exception as e:
                    raise BatchProcessError(
                        collection_name=None,
                        object_name=key,
                        method=e.method,
                        original_exception=e.original_exception
                    ) from e
            return results
        return batch_method

    @classmethod
    def from_features_collection(cls, features_collection: FeaturesCollection, summary_cls = Summary):
        '''
        creates a SummaryCollection from a FeaturesCollection
        '''
        if not issubclass(summary_cls, Summary):
            raise TypeError(f"summary_cls must be Summary or a subclass, got {summary_cls}")
        #check that dict handles match tracking handles
        for handle, f in features_collection.features_dict.items():
            if handle != f.handle:
                raise ValueError(f"Key '{handle}' does not match object's handle '{f.handle}'")
        return cls({handle: summary_cls(f) for handle, f in features_collection.features_dict.items()})

    @classmethod
    def from_list(cls, summary_list: list[Summary]):
        '''
        creates a SummaryCollection from a list of Summary objects, keyed by handle
        '''
        handles = [obj.handle for obj in summary_list]
        if len(handles) != len(set(handles)):
            raise Exception('handles must be unique')
        summary_dict = {obj.handle: obj for obj in summary_list}
        return cls(summary_dict)

class MultipleSummaryCollection:
    '''
    collection of SummaryCollection objects
    (e.g. for comparison between groups)
    '''
    dict_of_summary_collections: dict[str, SummaryCollection]
    
    def __init__(self, dict_of_summary_collections: dict[str, SummaryCollection]):
        self.dict_of_summary_collections = dict_of_summary_collections

    def __getattr__(self, name):
        # Only called if the attribute is not found on the collection itself
        def batch_method(*args, **kwargs):
            results = {}
            for key, obj in self.dict_of_summary_collections.items():
                try:
                    method = getattr(obj, name)
                    results[key] = method(*args, **kwargs)
                except Exception as e:
                    raise BatchProcessError(
                        collection_name=key,
                        object_name=e.object_name,
                        method=e.method,
                        original_exception=e.original_exception
                    ) from e
            return results
        return batch_method
    
    @classmethod
    def from_multiple_features_collection(cls, multiple_features_collection: MultipleFeaturesCollection):
        '''
        creates a MultipleSummaryCollections from a MultipleFeaturesCollection
        '''
        multiple_summary_collection = {}
        for handle, features_collection in multiple_features_collection.features_collections.items():
            multiple_summary_collection[handle] = SummaryCollection.from_features_collection(features_collection)
        return cls(multiple_summary_collection)

    def bfa(self, column: str, all_states=None, numshuffles: int = 1000):
        from itertools import combinations
        #batch calculate transition matrix for each summary object
        transition_matrices = self.transition_matrix(column, all_states)
        #calculate manhattan distance for each group pair
        distances = {}
        for group1, group2 in combinations(self.dict_of_summary_collections.keys(), 2):
            _ = {}
            list1 = list(transition_matrices[group1].values())
            list2 = list(transition_matrices[group2].values())
            _['observed'] = self._manhattan_distance_twogroups(list1, list2)
            _['surrogates'] = [self._manhattan_distance_twogroups(*self.shuffle_lists(list1, list2)) 
                               for i in range(numshuffles)]
            distances[group1+'_vs_'+group2] = _
        return distances
    
    @staticmethod
    def bfa_stats(bfa_results: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        
        def percentile(observed: float, surrogates: list[float]) -> float:
            return sum(observed > pd.Series(surrogates))/(len(surrogates) + 1)
        
        def zscore(observed: float, surrogates: list[float]) -> float:
            return (observed - np.mean(surrogates))/np.std(surrogates)
        
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
                "right_tail_p": right_tail_p(observed, surrogates)
            }
        return stats

    @staticmethod
    def _manhattan_distance(transition_matrix1: pd.DataFrame, transition_matrix2: pd.DataFrame) -> float:
        #check that transition_matrix1 and transition_matrix2 have the same index and columns
        if not transition_matrix1.index.equals(transition_matrix2.index):
            raise ValueError('transition_matrix1 and transition_matrix2 must have the same index')
        if not transition_matrix1.columns.equals(transition_matrix2.columns):
            raise ValueError('transition_matrix1 and transition_matrix2 must have the same columns')
        difference = transition_matrix1 - transition_matrix2
        return difference.abs().sum(axis=1).sum()
    
    @staticmethod
    def _mean_transition_matrix(matrices: list[pd.DataFrame]) -> pd.DataFrame:
        summed_matrix = sum(matrices)
        mean_matrix = summed_matrix / len(matrices)
        return mean_matrix
    
    def _manhattan_distance_twogroups(self, list1: list[pd.DataFrame], list2: list[pd.DataFrame]) -> float:
        #calculate manhattan distance between two lists of transition matrices
        distance = self._manhattan_distance(
            self._mean_transition_matrix(list1), 
            self._mean_transition_matrix(list2)
            )
        return distance

    @staticmethod
    def shuffle_lists(group1: list, group2: list) -> tuple[list, list]:
        import random
        n1 = len(group1)
        n2 = len(group2)
        combined = group1 + group2
        random.shuffle(combined)
        new_group1 = combined[:n1]
        new_group2 = combined[n1:]
        return new_group1, new_group2
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.summary_collection_list)} summary collections>"

    
    