from __future__ import annotations

import pandas as pd
import pytest

from py3r.behaviour.exceptions import BatchProcessError
from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.tracking.tracking import Tracking
from py3r.behaviour.util.collection_utils import BatchResult


def _make_features(handle: str, n_frames: int = 4) -> Features:
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
    return Features(tracking)


def test_features_compose_state_from_booleans_priority_and_none_label():
    features = _make_features("A")
    idx = features.tracking.data.index
    features.store(pd.Series([True, False, True, False], index=idx), "in_corner")
    features.store(pd.Series([False, True, True, False], index=idx), "in_food")

    state = features.compose_state_from_booleans(
        {"corner": "in_corner", "food": "in_food"},
        priority=["food", "corner"],
        none_label="none",
    )

    assert state.tolist() == ["corner", "food", "food", "none"]
    assert state.name == "state_from_booleans"


def test_features_compose_state_from_booleans_coerces_bool_like_object_series():
    features = _make_features("A", n_frames=3)
    object_bool = pd.Series([True, False, None], index=features.tracking.data.index, dtype="object")

    with pytest.warns(UserWarning, match="coercing to nullable boolean"):
        state = features.compose_state_from_booleans({"corner": object_bool})

    assert state.tolist() == ["corner", "none", "none"]


def test_features_compose_state_from_booleans_rejects_non_boolean_content():
    features = _make_features("A", n_frames=3)
    bad = pd.Series([1, 0, 1], index=features.tracking.data.index, dtype="int64")

    with pytest.raises(TypeError, match="must be boolean/nullable-boolean"):
        features.compose_state_from_booleans({"corner": bad})


def test_features_collection_each_compose_state_from_booleans_with_mixed_dict_mapping():
    fa = _make_features("A")
    fb = _make_features("B")
    fa.store(pd.Series([True, False, False, False], index=fa.tracking.data.index), "in_corner")
    fb.store(pd.Series([False, True, False, False], index=fb.tracking.data.index), "in_corner")

    fc = FeaturesCollection({"A": fa, "B": fb})
    in_food = BatchResult(
        {
            "A": pd.Series([False, True, False, False], index=fa.tracking.data.index),
            "B": pd.Series([True, False, False, False], index=fb.tracking.data.index),
        },
        fc,
    )

    states = fc.each.compose_state_from_booleans(
        {"corner": "in_corner", "food": in_food},
        priority=["food", "corner"],
    )

    assert isinstance(states, BatchResult)
    assert states["A"].tolist() == ["corner", "food", "none", "none"]
    assert states["B"].tolist() == ["food", "corner", "none", "none"]


def test_features_collection_each_compose_state_from_booleans_rejects_key_mismatch():
    fa = _make_features("A")
    fb = _make_features("B")
    fc = FeaturesCollection({"A": fa, "B": fb})
    bad_source = BatchResult(
        {"A": pd.Series([True, False, True, False], index=fa.tracking.data.index)},
        fc,
    )

    with pytest.raises(BatchProcessError, match="BatchResult mapping keys"):
        fc.each.compose_state_from_booleans({"corner": bad_source})


def test_grouped_collection_each_compose_state_from_booleans_maps_per_group_handle():
    fa = _make_features("A")
    fb = _make_features("B")
    fa.tags["cohort"] = "g1"
    fb.tags["cohort"] = "g2"
    fc = FeaturesCollection({"A": fa, "B": fb})
    grouped = fc.groupby("cohort")

    source = BatchResult(
        {
            "A": pd.Series([True, False, False, False], index=fa.tracking.data.index),
            "B": pd.Series([False, True, False, False], index=fb.tracking.data.index),
        },
        fc,
    )

    states = grouped.each.compose_state_from_booleans({"zone": source})

    assert isinstance(states, BatchResult)
    assert states[("g1",)]["A"].tolist() == ["zone", "none", "none", "none"]
    assert states[("g2",)]["B"].tolist() == ["none", "zone", "none", "none"]
