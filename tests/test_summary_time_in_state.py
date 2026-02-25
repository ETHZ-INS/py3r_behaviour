from __future__ import annotations

import pandas as pd

from py3r.behaviour.features.features import Features
from py3r.behaviour.summary.summary import Summary
from py3r.behaviour.tracking.tracking import Tracking


def _make_summary_with_states(states: list[str], fps: float = 2.0) -> Summary:
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
    return Summary(features)


def test_time_in_state_default_returns_observed_states_only():
    summary = _make_summary_with_states(["A", "A", "B", "A"], fps=2.0)

    result = summary.time_in_state("state")

    assert set(result.value.index.tolist()) == {"A", "B"}
    assert result.value["A"] == 1.5  # 3 frames / 2 fps
    assert result.value["B"] == 0.5  # 1 frame / 2 fps


def test_time_in_state_all_states_includes_missing_and_respects_order():
    summary = _make_summary_with_states(["A", "A", "B", "A"], fps=2.0)

    result = summary.time_in_state("state", all_states=["B", "C", "A"])

    assert result.value.index.tolist() == ["B", "C", "A"]
    assert result.value["B"] == 0.5
    assert result.value["C"] == 0.0
    assert result.value["A"] == 1.5
    assert result._params["all_states"] == ["B", "C", "A"]


def test_time_in_state_all_states_can_exclude_unwanted_states():
    summary = _make_summary_with_states(["A", "A", "B", "A"], fps=2.0)

    result = summary.time_in_state("state", all_states=["A"])

    assert result.value.index.tolist() == ["A"]
    assert result.value["A"] == 1.5


def test_count_state_onsets_default_returns_observed_states_only():
    summary = _make_summary_with_states(["A", "A", "B", "A"], fps=2.0)

    result = summary.count_state_onsets("state")

    assert set(result.value.index.tolist()) == {"A", "B"}
    assert result.value["A"] == 2
    assert result.value["B"] == 1


def test_count_state_onsets_all_states_includes_missing_and_respects_order():
    summary = _make_summary_with_states(["A", "A", "B", "A"], fps=2.0)

    result = summary.count_state_onsets("state", all_states=["B", "C", "A"])

    assert result.value.index.tolist() == ["B", "C", "A"]
    assert result.value["B"] == 1
    assert result.value["C"] == 0
    assert result.value["A"] == 2
    assert result._params["all_states"] == ["B", "C", "A"]


def test_count_state_onsets_all_states_can_exclude_unwanted_states():
    summary = _make_summary_with_states(["A", "A", "B", "A"], fps=2.0)

    result = summary.count_state_onsets("state", all_states=["A"])

    assert result.value.index.tolist() == ["A"]
    assert result.value["A"] == 2
