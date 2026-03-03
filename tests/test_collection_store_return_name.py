from __future__ import annotations

import pandas as pd
import pytest

from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.summary.summary_collection import SummaryCollection
from py3r.behaviour.tracking.tracking import Tracking


def _make_features(handle: str) -> Features:
    tracking_df = pd.DataFrame(
        {
            "bp.x": [0.0, 1.0, 2.0, 3.0],
            "bp.y": [0.0, 0.0, 0.0, 0.0],
        },
        index=pd.RangeIndex(4, name="frame"),
    )
    tracking = Tracking(
        tracking_df,
        {"fps": 30.0, "rescale_distance_method": "dummy"},
        handle=handle,
    )
    features = Features(tracking)
    features.store(
        pd.Series([True, False, True, False], index=tracking_df.index),
        "flag",
        meta={},
    )
    return features


def test_featurescollection_store_returns_resolved_name():
    fc = FeaturesCollection.from_list([_make_features("A"), _make_features("B")])
    results = fc.each.distance_change("bp")
    expected = next(iter(results.values()))._column_name

    stored_name = results.store()

    assert stored_name == expected
    assert all(expected in f.data.columns for f in fc.values())


def test_featurescollection_store_raises_on_mixed_auto_names():
    fc = FeaturesCollection.from_list([_make_features("A"), _make_features("B")])
    before_cols = {h: set(f.data.columns) for h, f in fc.items()}
    mixed = {
        "A": fc["A"].distance_change("bp"),
        "B": fc["B"].speed("bp"),
    }

    with pytest.raises(ValueError, match="resolved to multiple names"):
        fc.store(mixed)

    after_cols = {h: set(f.data.columns) for h, f in fc.items()}
    assert before_cols == after_cols


def test_summarycollection_store_returns_resolved_name():
    fc = FeaturesCollection.from_list([_make_features("A"), _make_features("B")])
    sc = SummaryCollection.from_features_collection(fc)
    results = sc.each.total_distance("bp")
    expected = next(iter(results.values()))._func_name

    stored_name = results.store()

    assert stored_name == expected
    assert all(stored_name in s.data for s in sc.values())


def test_summarycollection_store_raises_on_mixed_auto_names():
    fc = FeaturesCollection.from_list([_make_features("A"), _make_features("B")])
    sc = SummaryCollection.from_features_collection(fc)
    before_keys = {h: set(s.data.keys()) for h, s in sc.items()}
    mixed = {
        "A": sc["A"].total_distance("bp"),
        "B": sc["B"].time_true("flag"),
    }

    with pytest.raises(ValueError, match="resolved to multiple names"):
        sc.store(mixed)

    after_keys = {h: set(s.data.keys()) for h, s in sc.items()}
    assert before_keys == after_keys
