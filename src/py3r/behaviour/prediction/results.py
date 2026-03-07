from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from py3r.behaviour.util.collection_utils import BatchResult


@dataclass(frozen=True)
class PredictionComparisonResult:
    """
    Standalone result payload for cross-prediction comparison.

    - frame outputs are returned as BatchResult[FeaturesResult]
    - summary/stats are collection-level tables/dicts
    """

    within: BatchResult
    between: BatchResult
    ratio: BatchResult
    log_ratio: BatchResult
    summary_table: pd.DataFrame
    stats: dict
