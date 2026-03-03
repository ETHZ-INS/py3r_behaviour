from __future__ import annotations

import pandas as pd

from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.summary.summary import Summary
from py3r.behaviour.summary.summary_collection import SummaryCollection
from py3r.behaviour.tracking.tracking import Tracking
from py3r.behaviour.tracking.tracking_collection import TrackingCollection


def _make_tracking(
    handle: str,
    columns: dict[str, list[float]],
) -> Tracking:
    df = pd.DataFrame(columns, index=pd.RangeIndex(3, name="frame"))
    return Tracking(
        df,
        {"fps": 30.0, "rescale_distance_method": "dummy"},
        handle=handle,
    )


def test_trackingcollection_stored_info_reports_points_counts_and_dims():
    t1 = _make_tracking(
        "A",
        {
            "p1.x": [0.0, 1.0, 2.0],
            "p1.y": [0.0, 0.0, 0.0],
            "p1.likelihood": [0.9, 0.8, 0.7],
            "p2.x": [1.0, 1.0, 1.0],
            "p2.y": [2.0, 2.0, 2.0],
            "p2.likelihood": [0.5, 0.6, 0.7],
        },
    )
    t2 = _make_tracking(
        "B",
        {
            "p1.x": [0.0, 1.0, 2.0],
            "p1.y": [0.0, 0.0, 0.0],
        },
    )
    tc = TrackingCollection.from_list([t1, t2])

    info = tc.stored_info()

    assert list(info.columns) == ["attached_to", "missing_from", "dims"]
    assert info.index.name == "point_name"
    assert int(info.loc["p1", "attached_to"]) == 2
    assert int(info.loc["p1", "missing_from"]) == 0
    assert info.loc["p1", "dims"] == ["x", "y"]
    assert int(info.loc["p2", "attached_to"]) == 1
    assert int(info.loc["p2", "missing_from"]) == 1
    assert info.loc["p2", "dims"] == ["x", "y"]


def test_featurescollection_stored_info_reports_features_counts_and_type():
    t1 = _make_tracking("A", {"bp.x": [0.0, 1.0, 2.0], "bp.y": [0.0, 0.0, 0.0]})
    t2 = _make_tracking("B", {"bp.x": [1.0, 2.0, 3.0], "bp.y": [0.0, 0.0, 0.0]})
    f1 = Features(t1)
    f2 = Features(t2)
    f1.store(pd.Series([True, False, True], index=t1.data.index), "flag", meta={})
    f1.store(pd.Series([0.1, 0.2, 0.3], index=t1.data.index), "score", meta={})
    f2.store(pd.Series([1, 2, 3], index=t2.data.index), "score", meta={})
    fc = FeaturesCollection.from_list([f1, f2])

    info = fc.stored_info()

    assert list(info.columns) == ["attached_to", "missing_from", "type"]
    assert info.index.name == "feature"
    assert int(info.loc["flag", "attached_to"]) == 1
    assert int(info.loc["flag", "missing_from"]) == 1
    assert info.loc["flag", "type"] == "bool"
    assert int(info.loc["score", "attached_to"]) == 2
    assert int(info.loc["score", "missing_from"]) == 0
    assert info.loc["score", "type"] == ["float64", "int64"]


def test_summarycollection_stored_info_reports_summary_counts_and_type():
    t1 = _make_tracking("A", {"bp.x": [0.0, 1.0, 2.0], "bp.y": [0.0, 0.0, 0.0]})
    t2 = _make_tracking("B", {"bp.x": [1.0, 2.0, 3.0], "bp.y": [0.0, 0.0, 0.0]})
    s1 = Summary(Features(t1))
    s2 = Summary(Features(t2))
    s1.store(1.0, "score", meta={})
    s2.store(1, "score", meta={})
    s1.store(5, "events", meta={})
    sc = SummaryCollection.from_list([s1, s2])

    info = sc.stored_info()

    assert list(info.columns) == ["attached_to", "missing_from", "type"]
    assert info.index.name == "summary"
    assert int(info.loc["events", "attached_to"]) == 1
    assert int(info.loc["events", "missing_from"]) == 1
    assert info.loc["events", "type"] == "int"
    assert int(info.loc["score", "attached_to"]) == 2
    assert int(info.loc["score", "missing_from"]) == 0
    assert info.loc["score", "type"] == ["float", "int"]
