"""Tests for Features.axis_intersects_boundary.

Geometry
--------
All frames share a horizontal axis  A=(0,0) → B=(10,0)  (except frames that
test degenerate/NaN behaviour).  Zone classification by scalar projection t:

    behind : t ≤ 0   (at or before A)
    within : 0 < t < 1  (strictly between A and B)
    front  : t ≥ 1   (at or beyond B)

Frame layout
------------
0 : both crossings within (t≈0.37, t≈0.63)   → within=T, behind=F, front=F
1 : both crossings behind (t≈−0.37, t≈−0.63) → within=F, behind=T, front=F
2 : both crossings front  (t≈1.37, t≈1.63)   → within=F, behind=F, front=T
3 : one behind (t=−0.1),  one within (t=0.3)  → within=T, behind=T, front=F
4 : one within (t=0.7),   one front (t=1.1)   → within=T, behind=F, front=T
5 : no crossing (triangle entirely above y=0)  → all False
6 : degenerate axis A==B                       → all <NA>
7 : NaN in axis reference point                → all <NA>
8 : crossing exactly at A (t=0, t=0.1)         → behind=T, within=T, front=F
9 : crossing exactly at B (t=1.0, t=1.1)       → within=F, behind=F, front=T
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from py3r.behaviour.features.features import Features
from py3r.behaviour.tracking.tracking import Tracking

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tracking():
    data = pd.DataFrame(
        {
            # Axis reference points
            "a.x": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, np.nan, 0.0, 0.0],
            "a.y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "b.x": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 5.0, 10.0, 10.0, 10.0],
            "b.y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # Boundary triangle vertices
            #   p1: first vertex (below axis in frames 0-4, above in frame 5)
            "p1.x": [3.0, -3.0, 13.0, -2.0, 6.0, 3.0, 3.0, 3.0, -1.0, 9.0],
            "p1.y": [-2.0, -2.0, -2.0, -2.0, -2.0, 2.0, -2.0, -2.0, -1.0, -1.0],
            "p2.x": [7.0, -7.0, 17.0, 4.0, 12.0, 7.0, 7.0, 7.0, 1.0, 11.0],
            "p2.y": [-2.0, -2.0, -2.0, -2.0, -2.0, 2.0, -2.0, -2.0, 1.0, 1.0],
            "p3.x": [5.0, -5.0, 15.0, 1.0, 9.0, 5.0, 5.0, 5.0, 1.0, 11.0],
            "p3.y": [4.0, 4.0, 4.0, 4.0, 4.0, 6.0, 4.0, 4.0, -1.0, -1.0],
        },
        index=pd.RangeIndex(10, name="frame"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Tracking(data, {"fps": 30.0}, handle="test")


@pytest.fixture
def f():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Features(_make_tracking())


@pytest.fixture
def dyn_axis(f):
    return f.define_dynamic_axis("a", "b", dims=("x", "y"))


@pytest.fixture
def dyn_boundary(f):
    return f.define_dynamic_boundary(["p1", "p2", "p3"], dims=("x", "y"), name="tri")


def _call(f, *args, **kwargs):
    """Call axis_intersects_boundary suppressing calibration/smoothing warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f.axis_intersects_boundary(*args, **kwargs)


# ---------------------------------------------------------------------------
# Core zone classification
# ---------------------------------------------------------------------------


class TestZoneClassification:
    def test_within_true_both_crossings_in_segment(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary, zones={"within"})
        assert res.iloc[0]

    def test_within_false_all_crossings_behind(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary, zones={"within"})
        assert not res.iloc[1]

    def test_within_false_all_crossings_front(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary, zones={"within"})
        assert not res.iloc[2]

    def test_behind_true_crossings_before_A(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary, zones={"behind"})
        assert res.iloc[1]

    def test_behind_false_crossings_front(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary, zones={"behind"})
        assert not res.iloc[2]

    def test_front_true_crossings_beyond_B(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary, zones={"front"})
        assert res.iloc[2]

    def test_front_false_crossings_within(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary, zones={"front"})
        assert not res.iloc[0]

    def test_mixed_behind_and_within(self, f, dyn_axis, dyn_boundary):
        # frame 3: t=-0.1 (behind) and t=0.3 (within)
        assert _call(f, dyn_axis, dyn_boundary, zones={"behind"}).iloc[3]
        assert _call(f, dyn_axis, dyn_boundary, zones={"within"}).iloc[3]
        assert not _call(f, dyn_axis, dyn_boundary, zones={"front"}).iloc[3]

    def test_mixed_within_and_front(self, f, dyn_axis, dyn_boundary):
        # frame 4: t=0.7 (within) and t=1.1 (front)
        assert not _call(f, dyn_axis, dyn_boundary, zones={"behind"}).iloc[4]
        assert _call(f, dyn_axis, dyn_boundary, zones={"within"}).iloc[4]
        assert _call(f, dyn_axis, dyn_boundary, zones={"front"}).iloc[4]

    def test_false_when_no_crossing(self, f, dyn_axis, dyn_boundary):
        # frame 5: triangle entirely above y=0
        res = _call(f, dyn_axis, dyn_boundary)
        assert not res.iloc[5]

    def test_default_zones_equals_all_three_explicit(self, f, dyn_axis, dyn_boundary):
        default = _call(f, dyn_axis, dyn_boundary)
        explicit = _call(f, dyn_axis, dyn_boundary, zones={"front", "within", "behind"})
        pd.testing.assert_series_equal(pd.Series(default), pd.Series(explicit))


# ---------------------------------------------------------------------------
# Zone boundary cases: t=0 is "behind", t=1 is "front"
# ---------------------------------------------------------------------------


class TestZoneBoundaries:
    def test_t_zero_classified_as_behind_not_within(self, f, dyn_axis, dyn_boundary):
        # frame 8: edge crosses axis exactly at A → t=0 → behind
        assert _call(f, dyn_axis, dyn_boundary, zones={"behind"}).iloc[8]
        # second crossing at t=0.1 makes within also True for this frame
        assert _call(f, dyn_axis, dyn_boundary, zones={"within"}).iloc[8]
        assert not _call(f, dyn_axis, dyn_boundary, zones={"front"}).iloc[8]

    def test_t_one_classified_as_front_not_within(self, f, dyn_axis, dyn_boundary):
        # frame 9: edge crosses axis exactly at B → t=1 → front; second crossing at t=1.1 also front
        assert _call(f, dyn_axis, dyn_boundary, zones={"front"}).iloc[9]
        assert not _call(f, dyn_axis, dyn_boundary, zones={"within"}).iloc[9]
        assert not _call(f, dyn_axis, dyn_boundary, zones={"behind"}).iloc[9]


# ---------------------------------------------------------------------------
# NaN and degenerate axis handling
# ---------------------------------------------------------------------------


class TestNaNAndDegenerate:
    def test_degenerate_axis_gives_na(self, f, dyn_axis, dyn_boundary):
        # frame 6: A == B
        res = _call(f, dyn_axis, dyn_boundary)
        assert pd.isna(res.iloc[6])

    def test_nan_axis_gives_na(self, f, dyn_axis, dyn_boundary):
        # frame 7: a.x is NaN
        res = _call(f, dyn_axis, dyn_boundary)
        assert pd.isna(res.iloc[7])

    def test_result_dtype_is_nullable_boolean(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary)
        assert isinstance(res.dtype, pd.BooleanDtype)

    def test_valid_frames_are_not_na(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary)
        for i in range(6):
            assert not pd.isna(res.iloc[i]), f"frame {i} unexpectedly NA"


# ---------------------------------------------------------------------------
# Static axis and static boundary variants
# ---------------------------------------------------------------------------


class TestStaticVariants:
    def test_static_axis_matches_dynamic_on_valid_frames(self, f, dyn_boundary):
        static_ax = f.import_static_axis([(0.0, 0.0), (10.0, 0.0)], dims=("x", "y"))
        dyn_ax = f.define_dynamic_axis("a", "b", dims=("x", "y"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = f.axis_intersects_boundary(static_ax, dyn_boundary, zones={"within"})
            d = f.axis_intersects_boundary(dyn_ax, dyn_boundary, zones={"within"})
        # Static axis is always valid, so frames 6 and 7 differ (dyn gives NA).
        for i in range(6):
            assert s.iloc[i] == d.iloc[i], f"frame {i} differs between static and dynamic axis"

    def test_static_boundary_all_frames_same_result(self, f, dyn_axis):
        # Triangle (3,-2),(7,-2),(5,4) crosses within on every frame where axis is valid.
        static_b = f.import_static_boundary([(3.0, -2.0), (7.0, -2.0), (5.0, 4.0)], dims=("x", "y"))
        res = _call(f, dyn_axis, static_b, zones={"within"})
        for i in range(6):
            assert res.iloc[i], f"frame {i}: expected True"
        assert pd.isna(res.iloc[6])  # degenerate axis
        assert pd.isna(res.iloc[7])  # NaN axis

    def test_accepts_axis_by_registered_name(self, f, dyn_boundary):
        f.define_dynamic_axis("a", "b", dims=("x", "y"), name="spine")
        res = _call(f, "spine", dyn_boundary, zones={"within"})
        assert res.iloc[0]

    def test_accepts_boundary_by_registered_name(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, "tri", zones={"within"})
        assert res.iloc[0]


# ---------------------------------------------------------------------------
# Validation and error cases
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zones_string_shorthand_matches_set(self, f, dyn_axis, dyn_boundary):
        str_result = _call(f, dyn_axis, dyn_boundary, zones="within")
        set_result = _call(f, dyn_axis, dyn_boundary, zones={"within"})
        pd.testing.assert_series_equal(pd.Series(str_result), pd.Series(set_result))

    def test_unknown_zone_raises(self, f, dyn_axis, dyn_boundary):
        with pytest.raises(ValueError, match="Unknown zone"):
            _call(f, dyn_axis, dyn_boundary, zones={"sideways"})

    def test_empty_zones_raises(self, f, dyn_axis, dyn_boundary):
        with pytest.raises(ValueError, match="must not be empty"):
            _call(f, dyn_axis, dyn_boundary, zones=set())

    def test_dims_mismatch_on_axis_raises(self, f, dyn_boundary):
        # Static axis with dims=("x","z"), calling with dims=("x","y")
        ax_xz = f.import_static_axis([(0.0, 0.0), (10.0, 0.0)], dims=("x", "z"))
        with pytest.raises(ValueError, match="axis dims"):
            _call(f, ax_xz, dyn_boundary, dims=("x", "y"))

    def test_dims_mismatch_on_boundary_raises(self, f):
        # Static axis with dims=("x","z"), static boundary with dims=("x","y"),
        # calling with dims=("x","z") → boundary dims mismatch
        ax_xz = f.import_static_axis([(0.0, 0.0), (10.0, 0.0)], dims=("x", "z"))
        b_xy = f.import_static_boundary([(3.0, -2.0), (7.0, -2.0), (5.0, 4.0)], dims=("x", "y"))
        with pytest.raises(ValueError, match="boundary dims"):
            _call(f, ax_xz, b_xy, dims=("x", "z"))

    def test_passing_boundary_as_axis_raises(self, f, dyn_boundary):
        with pytest.raises(TypeError):
            _call(f, dyn_boundary, dyn_boundary)

    def test_passing_axis_as_boundary_raises(self, f, dyn_axis):
        with pytest.raises(TypeError):
            _call(f, dyn_axis, dyn_axis)


# ---------------------------------------------------------------------------
# Feature name and meta
# ---------------------------------------------------------------------------


class TestFeatureNameAndMeta:
    def test_name_contains_function(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary, zones={"within"})
        assert "axis_intersects_boundary" in res.name

    def test_name_contains_named_axis_label(self, f, dyn_boundary):
        f.define_dynamic_axis("a", "b", dims=("x", "y"), name="spine")
        res = _call(f, "spine", dyn_boundary, zones={"within"})
        assert "spine" in res.name

    def test_name_contains_named_boundary_label(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, "tri", zones={"within"})
        assert "tri" in res.name

    def test_name_contains_zone(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary, zones={"within"})
        assert "within" in res.name

    def test_name_zones_are_sorted(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary, zones={"front", "behind"})
        assert "behind_front" in res.name

    def test_meta_function_key(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary)
        assert res._params["function"] == "axis_intersects_boundary"

    def test_meta_zones_are_sorted(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary, zones={"front", "within"})
        assert res._params["zones"] == sorted(["front", "within"])

    def test_meta_dims(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary)
        assert res._params["dims"] == ["x", "y"]

    def test_result_length_matches_tracking(self, f, dyn_axis, dyn_boundary):
        res = _call(f, dyn_axis, dyn_boundary)
        assert len(res) == len(f.tracking.data)
