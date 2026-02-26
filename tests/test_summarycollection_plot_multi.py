from __future__ import annotations

import pandas as pd
import pytest

from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.summary.summary_collection import SummaryCollection
from py3r.behaviour.tracking.tracking import Tracking
from py3r.behaviour.util.collection_utils import BatchResult


def _make_features(
    handle: str,
    states: list[str],
    states_alt: list[str],
    flags: list[bool],
    group: str | None = None,
):
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
        {"fps": 2.0, "rescale_distance_method": "dummy"},
        handle=handle,
    )
    if group is not None:
        tracking.add_tag("group", group)
    features = Features(tracking)
    features.store(pd.Series(states, index=tracking_df.index, name="state"), "state", meta={})
    features.store(
        pd.Series(states_alt, index=tracking_df.index, name="state_alt"),
        "state_alt",
        meta={},
    )
    features.store(pd.Series(flags, index=tracking_df.index, name="flag"), "flag", meta={})
    return features


def _store_plot_metrics(sc: SummaryCollection):
    for summary in sc.flatten().values():
        summary.time_in_state("state", all_states=["A", "B"]).store("time_state")
        summary.time_in_state("state_alt", all_states=["A", "B"]).store("time_state_alt")
        summary.time_true("flag").store("time_true")
        summary.count_state_onsets("state", all_states=["A", "B"]).store("onsets_state")


def _make_ungrouped_sc() -> SummaryCollection:
    fc = FeaturesCollection.from_list(
        [
            _make_features(
                "A",
                states=["A", "A", "B", "A", "B", "B", "A", "A"],
                states_alt=["B", "A", "A", "B", "A", "B", "B", "A"],
                flags=[True, False, True, True, False, False, True, False],
            ),
            _make_features(
                "B",
                states=["B", "A", "B", "B", "A", "A", "B", "A"],
                states_alt=["A", "B", "A", "A", "B", "B", "A", "B"],
                flags=[False, True, True, False, True, False, False, True],
            ),
        ]
    )
    sc = SummaryCollection.from_features_collection(fc)
    _store_plot_metrics(sc)
    return sc


def _make_grouped_sc() -> SummaryCollection:
    fc = FeaturesCollection.from_list(
        [
            _make_features(
                "g1_a",
                states=["A", "A", "B", "A", "B", "B", "A", "A"],
                states_alt=["B", "A", "A", "B", "A", "B", "B", "A"],
                flags=[True, False, True, True, False, False, True, False],
                group="G1",
            ),
            _make_features(
                "g1_b",
                states=["B", "A", "B", "B", "A", "A", "B", "A"],
                states_alt=["A", "B", "A", "A", "B", "B", "A", "B"],
                flags=[False, True, True, False, True, False, False, True],
                group="G1",
            ),
            _make_features(
                "g2_a",
                states=["A", "B", "A", "B", "A", "B", "A", "B"],
                states_alt=["B", "A", "B", "A", "B", "A", "B", "A"],
                flags=[True, True, False, False, True, True, False, False],
                group="G2",
            ),
            _make_features(
                "g2_b",
                states=["B", "B", "A", "A", "B", "A", "B", "A"],
                states_alt=["A", "A", "B", "B", "A", "B", "A", "B"],
                flags=[False, False, True, True, False, True, False, True],
                group="G2",
            ),
        ]
    )
    sc = SummaryCollection.from_features_collection(fc.groupby("group"))
    _store_plot_metrics(sc)
    return sc


def test_snsbar_accepts_list_metrics_for_multi_path():
    sc = _make_ungrouped_sc()

    fig, ax, df = sc.snsbar(["time_state", "time_state_alt"], show=False)

    assert "_metric" not in df.columns
    components = set(df["component"].astype(str).tolist())
    assert any(c.startswith("time_state::") for c in components)
    assert any(c.startswith("time_state_alt::") for c in components)
    assert fig is not None and ax is not None


def test_snsbar_single_item_metric_list_delegates_to_single_path():
    sc = _make_ungrouped_sc()

    _, _, df = sc.snsbar(["time_state"], show=False)

    assert "_metric" not in df.columns


def test_multi_metric_fast_fail_for_mixed_ylabels():
    sc = _make_ungrouped_sc()

    with pytest.raises(ValueError, match="identical y-axis labels"):
        sc.snsbar(["time_state", "onsets_state"], show=False)


def test_snsbar_multi_accepts_mixed_str_and_batchresult_items():
    sc = _make_ungrouped_sc()
    br = sc.each.time_true("flag")
    assert isinstance(br, BatchResult)

    _, _, df = sc.snsbar(["time_state", br], show=False)

    components = set(df["component"].astype(str).tolist())
    assert any(c.startswith("time_state::") for c in components)
    assert "time_true_flag::value" in components


def test_grouped_multi_metric_returns_one_plot_per_group():
    sc = _make_grouped_sc()

    fig, ax, df = sc.snsbar(["time_state", "time_state_alt"], show=False)

    assert fig is not None and ax is not None
    assert "_group" in df.columns
    components = set(df["component"].astype(str).tolist())
    assert any(c.startswith("time_state::") for c in components)
    assert any(c.startswith("time_state_alt::") for c in components)


def test_snsbar_merge_by_component_reorders_merged_labels():
    sc = _make_ungrouped_sc()

    _, _, df = sc.snsbar(["time_state", "time_state_alt"], merge_by="component", show=False)

    components = set(df["component"].astype(str).tolist())
    assert any(c.startswith("A::time_state") for c in components)
    assert any(c.startswith("A::time_state_alt") for c in components)


def test_snsbar_merge_by_invalid_value_raises():
    sc = _make_ungrouped_sc()

    with pytest.raises(ValueError, match="merge_by must be"):
        sc.snsbar(["time_state", "time_state_alt"], merge_by="bad_value", show=False)
