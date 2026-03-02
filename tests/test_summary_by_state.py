from __future__ import annotations

import pandas as pd
import pytest

from py3r.behaviour.features.features import Features
from py3r.behaviour.summary.summary import Summary
from py3r.behaviour.tracking.tracking import Tracking


def _make_summary_with_state_and_value(
    states: list[str], values: list[float], fps: float = 2.0
) -> Summary:
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
        {"fps": fps, "rescale_distance_method": "dummy"},
        handle="ex",
    )
    features = Features(tracking)
    features.store(pd.Series(states, index=tracking_df.index, name="state"), "state", meta={})
    features.store(pd.Series(values, index=tracking_df.index, name="x"), "x", meta={})
    return Summary(features)


def test_by_state_min_column_returns_series_per_state():
    summary = _make_summary_with_state_and_value(["A", "A", "B", "A"], [2, 1, 9, 3])

    result = summary.by_state("state").min_column("x")

    assert set(result.value.index.tolist()) == {"A", "B"}
    assert result.value["A"] == 1
    assert result.value["B"] == 9


def test_by_state_sum_column_respects_all_states_order():
    summary = _make_summary_with_state_and_value(["A", "A", "B", "A"], [2, 1, 9, 3])

    result = summary.by_state("state", all_states=["B", "C", "A"]).sum_column("x")

    assert result.value.index.tolist() == ["B", "C", "A"]
    assert result.value["B"] == 9
    assert result.value["C"] == 0
    assert result.value["A"] == 6
    assert result._params["all_states"] == ["B", "C", "A"]


def test_by_state_missing_state_column_raises():
    summary = _make_summary_with_state_and_value(["A", "A", "B"], [1, 2, 3])

    with pytest.raises(ValueError, match="Column 'missing' not found in features.data"):
        summary.by_state("missing")


def test_by_state_missing_value_column_raises():
    summary = _make_summary_with_state_and_value(["A", "A", "B"], [1, 2, 3])

    with pytest.raises(ValueError, match="Column 'missing' not found in features.data"):
        summary.by_state("state").mean_column("missing")


def test_by_state_returns_summary_type():
    summary = _make_summary_with_state_and_value(["A", "A", "B"], [1, 2, 3])

    scoped = summary.by_state("state")

    assert not isinstance(scoped, Summary)
    assert "min_column" in dir(scoped)


def test_unscoped_only_method_raises_on_state_scoped_summary():
    summary = _make_summary_with_state_and_value(["A", "A", "B"], [1, 2, 3])

    with pytest.raises(NotImplementedError, match="not marked as by_state-compatible"):
        summary.by_state("state").time_in_state("state")


def test_by_state_supports_time_true_and_time_false():
    summary = _make_summary_with_state_and_value(["A", "A", "B", "B"], [1, 2, 3, 4])
    summary.features.store(
        pd.Series([True, False, True, True], index=summary.features.tracking.data.index),
        "flag",
        meta={},
    )

    true_res = summary.by_state("state").time_true("flag")
    false_res = summary.by_state("state").time_false("flag")

    assert true_res.value["A"] == 0.5
    assert true_res.value["B"] == 1.0
    assert false_res.value["A"] == 0.5
    assert false_res.value["B"] == 0.0
    assert true_res._ylabel == "Time (s)"
    assert false_res._ylabel == "Time (s)"


def test_by_state_all_states_is_inclusive_for_missing_state():
    summary = _make_summary_with_state_and_value(["A", "A", "B", "B"], [1, 2, 3, 4])
    summary.features.store(
        pd.Series([True, False, True, True], index=summary.features.tracking.data.index),
        "flag",
        meta={},
    )

    true_res = summary.by_state("state", all_states=["A", "B", "C"]).time_true("flag")

    assert true_res.value.index.tolist() == ["A", "B", "C"]
    assert true_res.value["A"] == 0.5
    assert true_res.value["B"] == 1.0
    assert true_res.value["C"] == 0.0


def test_by_state_supports_total_distance():
    summary = _make_summary_with_state_and_value(["A", "A", "B", "B"], [1, 2, 3, 4])

    result = summary.by_state("state").total_distance("bp")

    assert set(result.value.index.tolist()) == {"A", "B"}
    assert result.value["A"] > 0
    assert result.value["B"] > 0
    assert result._ylabel == "Distance (a.u.)"


def test_by_state_dir_lists_only_marked_methods():
    summary = _make_summary_with_state_and_value(["A", "A", "B"], [1, 2, 3])

    names = dir(summary.by_state("state"))

    assert "min_column" in names
    assert "time_true" in names
    assert "time_in_state" not in names


def test_by_state_max_states_guard_raises_helpful_error():
    summary = _make_summary_with_state_and_value(["A", "A", "B"], [1, 2, 3])

    with pytest.raises(ValueError, match="exceeding max_states=1"):
        summary.by_state("state", max_states=1).min_column("x")
