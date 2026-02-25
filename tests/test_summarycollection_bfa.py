from __future__ import annotations

import random

import pandas as pd

from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.summary.summary_collection import SummaryCollection
from py3r.behaviour.tracking.tracking import Tracking


def _make_features_with_states(handle: str, group: str, states: list[str]) -> Features:
    n_frames = len(states)
    tracking_df = pd.DataFrame(
        {
            "bp.x": [float(i) for i in range(n_frames)],
            "bp.y": [0.0 for _ in range(n_frames)],
        },
        index=pd.RangeIndex(n_frames, name="frame"),
    )
    tracking = Tracking(
        tracking_df,
        {"fps": 30.0, "rescale_distance_method": "dummy"},
        handle=handle,
    )
    tracking.add_tag("group", group)
    features = Features(tracking)
    state_series = pd.Series(states, index=tracking_df.index, name="state")
    features.store(state_series, "state", meta={})
    return features


def _make_grouped_summary_collection() -> SummaryCollection:
    features = [
        _make_features_with_states("g1_a", "G1", ["A", "A", "B", "A", "B", "B", "A", "A"]),
        _make_features_with_states("g1_b", "G1", ["A", "B", "B", "A", "A", "B", "A", "B"]),
        _make_features_with_states("g2_a", "G2", ["B", "B", "A", "B", "A", "A", "B", "B"]),
        _make_features_with_states("g2_b", "G2", ["B", "A", "A", "B", "B", "A", "B", "A"]),
    ]
    fc = FeaturesCollection.from_list(features)
    return SummaryCollection.from_features_collection(fc.groupby("group"))


def _make_non_degenerate_grouped_summary_collection() -> SummaryCollection:
    features = [
        _make_features_with_states(
            "g1_a",
            "G1",
            ["A", "B", "C", "A", "B", "C", "A", "A", "B", "C", "B", "A"],
        ),
        _make_features_with_states(
            "g1_b",
            "G1",
            ["A", "A", "C", "B", "C", "A", "B", "C", "A", "B", "B", "C"],
        ),
        _make_features_with_states(
            "g1_c",
            "G1",
            ["C", "A", "B", "A", "C", "B", "A", "C", "B", "A", "C", "B"],
        ),
        _make_features_with_states(
            "g2_a",
            "G2",
            ["C", "B", "A", "C", "B", "A", "C", "C", "B", "A", "B", "C"],
        ),
        _make_features_with_states(
            "g2_b",
            "G2",
            ["B", "C", "A", "B", "A", "C", "B", "A", "C", "B", "A", "A"],
        ),
        _make_features_with_states(
            "g2_c",
            "G2",
            ["A", "C", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B"],
        ),
    ]
    fc = FeaturesCollection.from_list(features)
    return SummaryCollection.from_features_collection(fc.groupby("group"))


def test_bfa_random_state_reproducible():
    sc = _make_grouped_summary_collection()
    all_states = ["A", "B"]

    out1 = sc.bfa("state", all_states=all_states, numshuffles=50, random_state=123)
    out2 = sc.bfa("state", all_states=all_states, numshuffles=50, random_state=123)

    assert out1 == out2


def test_bfa_random_state_changes_surrogates():
    group1 = list(range(10))
    group2 = list(range(10, 20))

    shuffled1 = SummaryCollection._shuffle_lists(group1, group2, random.Random(123))
    shuffled2 = SummaryCollection._shuffle_lists(group1, group2, random.Random(456))

    assert shuffled1 != shuffled2


def test_bfa_random_state_changes_surrogates_non_degenerate():
    sc = _make_non_degenerate_grouped_summary_collection()
    all_states = ["A", "B", "C"]

    out1 = sc.bfa("state", all_states=all_states, numshuffles=100, random_state=123)
    out2 = sc.bfa("state", all_states=all_states, numshuffles=100, random_state=456)

    compare = next(iter(out1.keys()))
    assert out1[compare]["surrogates"] != out2[compare]["surrogates"]
