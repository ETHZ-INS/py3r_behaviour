from __future__ import annotations

import pandas as pd

from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.summary.summary import Summary
from py3r.behaviour.summary.summary_collection import SummaryCollection
from py3r.behaviour.tracking.tracking import Tracking
from py3r.behaviour.tracking.tracking_collection import TrackingCollection


def _make_tracking(handle: str, group: str | None = None) -> Tracking:
    df = pd.DataFrame(
        {
            "bp.x": [0.0, 1.0, 2.0],
            "bp.y": [0.0, 1.0, 0.0],
        },
        index=pd.RangeIndex(3, name="frame"),
    )
    tracking = Tracking(df, {"fps": 30.0}, handle=handle)
    if group is not None:
        tracking.add_tag("group", group)
    return tracking


def test_tracking_to_features_and_to_summary():
    tracking = _make_tracking("A")

    features = tracking.to_features()
    summary = tracking.to_summary()

    assert isinstance(features, Features)
    assert features.handle == "A"
    assert isinstance(summary, Summary)
    assert summary.handle == "A"
    assert isinstance(summary.features, Features)


def test_trackingcollection_to_features_and_to_summary():
    tc = TrackingCollection({"A": _make_tracking("A"), "B": _make_tracking("B")})

    fc = tc.to_features()
    sc = tc.to_summary()

    assert isinstance(fc, FeaturesCollection)
    assert set(fc.keys()) == {"A", "B"}
    assert all(isinstance(v, Features) for v in fc.values())
    assert isinstance(sc, SummaryCollection)
    assert set(sc.keys()) == {"A", "B"}
    assert all(isinstance(v, Summary) for v in sc.values())


def test_features_to_summary_and_collection_to_summary_grouped():
    tc = TrackingCollection(
        {
            "g1_a": _make_tracking("g1_a", group="G1"),
            "g1_b": _make_tracking("g1_b", group="G1"),
            "g2_a": _make_tracking("g2_a", group="G2"),
        }
    )
    grouped_tc = tc.groupby("group")

    grouped_fc = grouped_tc.to_features()
    grouped_sc = grouped_fc.to_summary()

    assert grouped_fc.is_grouped
    assert grouped_sc.is_grouped
    assert set(grouped_sc.keys()) == {("G1",), ("G2",)}
    assert all(isinstance(sub, SummaryCollection) for sub in grouped_sc.values())

    single_summary = grouped_fc[("G1",)]["g1_a"].to_summary()
    assert isinstance(single_summary, Summary)
