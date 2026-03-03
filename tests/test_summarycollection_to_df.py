from __future__ import annotations

import pandas as pd

from py3r.behaviour.features.features import Features
from py3r.behaviour.summary.summary import Summary
from py3r.behaviour.summary.summary_collection import SummaryCollection
from py3r.behaviour.tracking.tracking import Tracking


def _make_summary(handle: str, score: float, group: str) -> Summary:
    tracking_df = pd.DataFrame(
        {"bp.x": [0.0, 1.0], "bp.y": [0.0, 0.0]},
        index=pd.RangeIndex(2, name="frame"),
    )
    tracking = Tracking(
        tracking_df,
        {"fps": 30.0, "rescale_distance_method": "dummy"},
        handle=handle,
    )
    tracking.add_tag("group", group)
    summary = Summary(Features(tracking))
    summary.store(score, "score")
    return summary


def test_to_df_default_ignores_series_entries():
    s1 = _make_summary("A", 1.0, "G1")
    s2 = _make_summary("B", 2.0, "G2")
    s1.store(pd.Series({"state1": 10.0, "state2": 20.0}), "speed_by_state")
    s2.store(pd.Series({"state1": 30.0, "state2": 40.0}), "speed_by_state")
    sc = SummaryCollection.from_list([s1, s2])

    df = sc.to_df(include_tags=True)

    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == ["A", "B"]
    assert "score" in df.columns
    assert "tag_group" in df.columns
    assert "speed_by_state" not in df.columns


def test_to_df_series_separate_returns_tuple_with_series_tables():
    s1 = _make_summary("A", 1.0, "G1")
    s2 = _make_summary("B", 2.0, "G2")
    s1.store(pd.Series({"state1": 10.0, "state2": 20.0}), "speed_by_state")
    s2.store(pd.Series({"state1": 30.0, "state2": 40.0}), "speed_by_state")
    sc = SummaryCollection.from_list([s1, s2])

    scalars_df, series_tables = sc.to_df(series="separate")

    assert isinstance(scalars_df, pd.DataFrame)
    assert "score" in scalars_df.columns
    assert "speed_by_state" not in scalars_df.columns

    assert set(series_tables.keys()) == {"speed_by_state"}
    speed_df = series_tables["speed_by_state"]
    assert isinstance(speed_df, pd.DataFrame)
    assert speed_df.index.name == "handle"
    assert speed_df.loc["A", "state1"] == 10.0
    assert speed_df.loc["A", "state2"] == 20.0
    assert speed_df.loc["B", "state1"] == 30.0
    assert speed_df.loc["B", "state2"] == 40.0


def test_to_df_series_separate_includes_tags_in_series_tables():
    s1 = _make_summary("A", 1.0, "G1")
    s2 = _make_summary("B", 2.0, "G2")
    s1.store(pd.Series({"state1": 10.0, "state2": 20.0}), "speed_by_state")
    s2.store(pd.Series({"state1": 30.0, "state2": 40.0}), "speed_by_state")
    sc = SummaryCollection.from_list([s1, s2])

    scalars_df, series_tables = sc.to_df(include_tags=True, series="separate")

    assert scalars_df.loc["A", "tag_group"] == "G1"
    assert scalars_df.loc["B", "tag_group"] == "G2"

    speed_df = series_tables["speed_by_state"]
    assert speed_df.loc["A", "tag_group"] == "G1"
    assert speed_df.loc["B", "tag_group"] == "G2"
