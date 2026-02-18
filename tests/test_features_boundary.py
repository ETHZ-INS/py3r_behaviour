"""
Edge-case tests for the vectorized boundary methods in Features:
  - within_boundary_static
  - within_boundary_dynamic
  - distance_to_boundary_static
  - distance_to_boundary_dynamic
  - area_of_boundary (dynamic branch)

The test DataFrame has 8 rows, each designed to probe a specific scenario:
  row 0: query point clearly inside the boundary — all coords valid
  row 1: query point clearly outside the boundary — all coords valid
  row 2: query point x is NaN
  row 3: query point y is NaN
  row 4: both x and y of query point are NaN
  row 5: one dynamic boundary vertex is NaN (query point valid)
  row 6: all coordinates NaN
  row 7: query point on the boundary edge
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from py3r.behaviour.features.features import Features
from py3r.behaviour.tracking.tracking import Tracking


def _make_tracking():
    """
    Build a minimal Tracking with 4 tracked points: q, b1, b2, b3.

    The boundary triangle b1-b2-b3 forms a right triangle at the origin:
        b1=(0,0)  b2=(10,0)  b3=(0,10)

    q is the query point whose position varies per row to cover edge cases.
    Row 5 has a NaN in b3 to test dynamic-boundary NaN propagation.
    """
    data = pd.DataFrame(
        {
            # query point
            "q.x": [3.0, 20.0, np.nan, 3.0, np.nan, 3.0, np.nan, 5.0],
            "q.y": [3.0, 20.0, 3.0, np.nan, np.nan, 3.0, np.nan, 0.0],
            # boundary vertex 1
            "b1.x": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0],
            "b1.y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0],
            # boundary vertex 2
            "b2.x": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, np.nan, 10.0],
            "b2.y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0],
            # boundary vertex 3
            "b3.x": [0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan, 0.0],
            "b3.y": [10.0, 10.0, 10.0, 10.0, 10.0, np.nan, np.nan, 10.0],
        },
        index=pd.RangeIndex(8, name="frame"),
    )
    meta = {"fps": 30.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Tracking(data, meta, handle="test")


STATIC_BOUNDARY = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
BOUNDARY_NAMES = ["b1", "b2", "b3"]

# expected triangle area via shoelace: 0.5 * |10*10 - 0| = 50
EXPECTED_AREA = 50.0


@pytest.fixture
def features():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Features(_make_tracking())


# ── within_boundary_static ──────────────────────────────────────────


class TestWithinBoundaryStatic:
    def test_inside(self, features):
        res = features.within_boundary_static("q", STATIC_BOUNDARY)
        assert bool(res.iloc[0])

    def test_outside(self, features):
        res = features.within_boundary_static("q", STATIC_BOUNDARY)
        assert not bool(res.iloc[1])

    def test_point_x_nan(self, features):
        res = features.within_boundary_static("q", STATIC_BOUNDARY)
        assert pd.isna(res.iloc[2])

    def test_point_y_nan(self, features):
        res = features.within_boundary_static("q", STATIC_BOUNDARY)
        assert pd.isna(res.iloc[3])

    def test_point_both_nan(self, features):
        res = features.within_boundary_static("q", STATIC_BOUNDARY)
        assert pd.isna(res.iloc[4])

    def test_on_edge(self, features):
        """Point (5,0) lies on the b1-b2 edge; shapely.contains returns False for boundary."""
        res = features.within_boundary_static("q", STATIC_BOUNDARY)
        assert not bool(res.iloc[7])

    def test_boundary_with_nan_all_na(self, features):
        nan_boundary = [(0.0, 0.0), (10.0, 0.0), (np.nan, 10.0)]
        res = features.within_boundary_static("q", nan_boundary)
        assert res.isna().all()

    def test_result_dtype(self, features):
        res = features.within_boundary_static("q", STATIC_BOUNDARY)
        assert res.dtype == pd.BooleanDtype()

    def test_result_length(self, features):
        res = features.within_boundary_static("q", STATIC_BOUNDARY)
        assert len(res) == 8


# ── within_boundary_dynamic ─────────────────────────────────────────


class TestWithinBoundaryDynamic:
    def test_inside(self, features):
        res = features.within_boundary_dynamic("q", BOUNDARY_NAMES)
        assert bool(res.iloc[0])

    def test_outside(self, features):
        res = features.within_boundary_dynamic("q", BOUNDARY_NAMES)
        assert not bool(res.iloc[1])

    def test_point_nan_rows(self, features):
        res = features.within_boundary_dynamic("q", BOUNDARY_NAMES)
        for i in [2, 3, 4]:
            assert pd.isna(res.iloc[i]), f"row {i} should be NA"

    def test_boundary_vertex_nan(self, features):
        """Row 5: query valid but b3 is NaN → result should be NA."""
        res = features.within_boundary_dynamic("q", BOUNDARY_NAMES)
        assert pd.isna(res.iloc[5])

    def test_all_nan_row(self, features):
        res = features.within_boundary_dynamic("q", BOUNDARY_NAMES)
        assert pd.isna(res.iloc[6])

    def test_on_edge(self, features):
        res = features.within_boundary_dynamic("q", BOUNDARY_NAMES)
        assert not bool(res.iloc[7])

    def test_result_dtype(self, features):
        res = features.within_boundary_dynamic("q", BOUNDARY_NAMES)
        assert res.dtype == pd.BooleanDtype()


# ── distance_to_boundary_static ─────────────────────────────────────


class TestDistanceToBoundaryStatic:
    def test_inside_positive_distance(self, features):
        res = features.distance_to_boundary_static("q", STATIC_BOUNDARY)
        assert res.iloc[0] > 0  # (3,3) is 3 units from nearest edge

    def test_outside_positive_distance(self, features):
        res = features.distance_to_boundary_static("q", STATIC_BOUNDARY)
        assert res.iloc[1] > 0

    def test_on_edge_zero_distance(self, features):
        res = features.distance_to_boundary_static("q", STATIC_BOUNDARY)
        assert res.iloc[7] == pytest.approx(0.0)

    def test_known_distance(self, features):
        """(3,3) nearest edge is hypotenuse x+y=10 → distance = 4/√2 = 2√2."""
        res = features.distance_to_boundary_static("q", STATIC_BOUNDARY)
        assert res.iloc[0] == pytest.approx(2 * np.sqrt(2))

    def test_point_nan_rows(self, features):
        res = features.distance_to_boundary_static("q", STATIC_BOUNDARY)
        for i in [2, 3, 4]:
            assert np.isnan(res.iloc[i]), f"row {i} should be NaN"

    def test_boundary_with_nan_all_nan(self, features):
        nan_boundary = [(0.0, 0.0), (10.0, 0.0), (np.nan, 10.0)]
        res = features.distance_to_boundary_static("q", nan_boundary)
        assert res.isna().all()


# ── distance_to_boundary_dynamic ────────────────────────────────────


class TestDistanceToBoundaryDynamic:
    def test_inside_positive_distance(self, features):
        res = features.distance_to_boundary_dynamic("q", BOUNDARY_NAMES)
        assert res.iloc[0] > 0

    def test_known_distance(self, features):
        res = features.distance_to_boundary_dynamic("q", BOUNDARY_NAMES)
        assert res.iloc[0] == pytest.approx(2 * np.sqrt(2))

    def test_on_edge_zero_distance(self, features):
        res = features.distance_to_boundary_dynamic("q", BOUNDARY_NAMES)
        assert res.iloc[7] == pytest.approx(0.0)

    def test_point_nan_rows(self, features):
        res = features.distance_to_boundary_dynamic("q", BOUNDARY_NAMES)
        for i in [2, 3, 4]:
            assert np.isnan(res.iloc[i]), f"row {i} should be NaN"

    def test_boundary_vertex_nan(self, features):
        res = features.distance_to_boundary_dynamic("q", BOUNDARY_NAMES)
        assert np.isnan(res.iloc[5])

    def test_all_nan_row(self, features):
        res = features.distance_to_boundary_dynamic("q", BOUNDARY_NAMES)
        assert np.isnan(res.iloc[6])


# ── area_of_boundary (dynamic) ──────────────────────────────────────


class TestAreaOfBoundaryDynamic:
    def test_valid_area(self, features):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = features.area_of_boundary(BOUNDARY_NAMES, median=False)
        assert res.iloc[0] == pytest.approx(EXPECTED_AREA)
        assert res.iloc[1] == pytest.approx(EXPECTED_AREA)

    def test_nan_boundary_vertex_propagates(self, features):
        """Row 5 has NaN in b3 → area should be NaN via arithmetic propagation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = features.area_of_boundary(BOUNDARY_NAMES, median=False)
        assert np.isnan(res.iloc[5])

    def test_all_nan_row(self, features):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = features.area_of_boundary(BOUNDARY_NAMES, median=False)
        assert np.isnan(res.iloc[6])

    def test_result_length(self, features):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = features.area_of_boundary(BOUNDARY_NAMES, median=False)
        assert len(res) == 8


# ── cross-method consistency ────────────────────────────────────────


class TestStaticDynamicConsistency:
    """
    When the dynamic boundary columns happen to be constant across rows
    (rows 0-4, 7 have identical b1/b2/b3), the static and dynamic methods
    should agree on valid rows.
    """

    def test_contains_agree(self, features):
        static = features.within_boundary_static("q", STATIC_BOUNDARY)
        dynamic = features.within_boundary_dynamic("q", BOUNDARY_NAMES)
        for i in [0, 1, 7]:
            assert static.iloc[i] == dynamic.iloc[i], f"mismatch at row {i}"

    def test_distance_agree(self, features):
        static = features.distance_to_boundary_static("q", STATIC_BOUNDARY)
        dynamic = features.distance_to_boundary_dynamic("q", BOUNDARY_NAMES)
        for i in [0, 1, 7]:
            assert static.iloc[i] == pytest.approx(dynamic.iloc[i]), f"mismatch at row {i}"

    def test_nan_positions_agree_contains(self, features):
        static = features.within_boundary_static("q", STATIC_BOUNDARY)
        dynamic = features.within_boundary_dynamic("q", BOUNDARY_NAMES)
        for i in [2, 3, 4]:
            assert pd.isna(static.iloc[i]) and pd.isna(dynamic.iloc[i])

    def test_nan_positions_agree_distance(self, features):
        static = features.distance_to_boundary_static("q", STATIC_BOUNDARY)
        dynamic = features.distance_to_boundary_dynamic("q", BOUNDARY_NAMES)
        for i in [2, 3, 4]:
            assert np.isnan(static.iloc[i]) and np.isnan(dynamic.iloc[i])
