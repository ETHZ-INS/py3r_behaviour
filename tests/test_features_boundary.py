"""
Boundary API tests for Features.

Focus:
  - Current API: within_boundary / distance_to_boundary / area_of_boundary
  - Removed legacy API paths now raise NotImplementedError
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from py3r.behaviour.features.features import Features
from py3r.behaviour.tracking.tracking import Tracking


def _make_tracking():
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
EXPECTED_AREA = 50.0


@pytest.fixture
def features():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Features(_make_tracking())


class TestRemovedLegacyBoundaryApis:
    def test_define_boundary_removed(self, features):
        with pytest.raises(NotImplementedError, match="define_static_boundary"):
            features.define_boundary(BOUNDARY_NAMES, scaling=1.0)

    def test_within_boundary_static_removed(self, features):
        with pytest.raises(NotImplementedError, match="within_boundary"):
            features.within_boundary_static("q", STATIC_BOUNDARY)

    def test_within_boundary_dynamic_removed(self, features):
        with pytest.raises(NotImplementedError, match="within_boundary"):
            features.within_boundary_dynamic("q", BOUNDARY_NAMES)

    def test_distance_to_boundary_static_removed(self, features):
        with pytest.raises(NotImplementedError, match="distance_to_boundary"):
            features.distance_to_boundary_static("q", STATIC_BOUNDARY)

    def test_distance_to_boundary_dynamic_removed(self, features):
        with pytest.raises(NotImplementedError, match="distance_to_boundary"):
            features.distance_to_boundary_dynamic("q", BOUNDARY_NAMES)

    def test_area_of_boundary_deprecated_removed(self, features):
        with pytest.raises(NotImplementedError, match="area_of_boundary"):
            features.area_of_boundary_deprecated(BOUNDARY_NAMES, median=False)


class TestAreaOfBoundary:
    def test_valid_area_dynamic_boundary(self, features):
        boundary = features.define_dynamic_boundary(BOUNDARY_NAMES)
        res = features.area_of_boundary(boundary)
        assert res.iloc[0] == pytest.approx(EXPECTED_AREA)
        assert res.iloc[1] == pytest.approx(EXPECTED_AREA)

    def test_valid_area_static_boundary(self, features):
        boundary = features.define_static_boundary(BOUNDARY_NAMES)
        res = features.area_of_boundary(boundary)
        assert res.iloc[0] == pytest.approx(EXPECTED_AREA)
        assert res.iloc[1] == pytest.approx(EXPECTED_AREA)
        assert res.nunique(dropna=False) == 1

    def test_valid_area_named_dynamic_boundary(self, features):
        features.define_dynamic_boundary(BOUNDARY_NAMES, name="tri_dyn", overwrite=True)
        res = features.area_of_boundary("tri_dyn")
        assert res.iloc[0] == pytest.approx(EXPECTED_AREA)
        assert res.iloc[1] == pytest.approx(EXPECTED_AREA)

    def test_nan_boundary_vertex_propagates_dynamic_boundary(self, features):
        boundary = features.define_dynamic_boundary(BOUNDARY_NAMES)
        res = features.area_of_boundary(boundary)
        assert np.isnan(res.iloc[5])

    def test_all_nan_row_dynamic_boundary(self, features):
        boundary = features.define_dynamic_boundary(BOUNDARY_NAMES)
        res = features.area_of_boundary(boundary)
        assert np.isnan(res.iloc[6])

    def test_result_length(self, features):
        boundary = features.define_dynamic_boundary(BOUNDARY_NAMES)
        res = features.area_of_boundary(boundary)
        assert len(res) == 8

    def test_legacy_point_list_rejected_by_new_api(self, features):
        with pytest.raises(TypeError, match="no longer accepts point-name lists"):
            features.area_of_boundary(BOUNDARY_NAMES)

    def test_new_api_rejects_legacy_kwargs(self, features):
        boundary = features.define_dynamic_boundary(BOUNDARY_NAMES)
        with pytest.raises(TypeError, match="accepts only `boundary`"):
            features.area_of_boundary(boundary, median=False)
        with pytest.raises(TypeError, match="accepts only `boundary`"):
            features.area_of_boundary(boundary, boundary_name="x")


class TestWithinBoundaryNewApi:
    def test_accepts_static_boundary_object(self, features):
        boundary = features.define_static_boundary(BOUNDARY_NAMES)
        res = features.within_boundary("q", boundary)
        assert bool(res.iloc[0])

    def test_accepts_dynamic_boundary_object(self, features):
        boundary = features.define_dynamic_boundary(BOUNDARY_NAMES)
        res = features.within_boundary("q", boundary)
        assert bool(res.iloc[0])

    def test_accepts_stored_boundary_name(self, features):
        features.define_dynamic_boundary(BOUNDARY_NAMES, name="tri_dyn", overwrite=True)
        res = features.within_boundary("q", "tri_dyn")
        assert bool(res.iloc[0])

    def test_rejects_legacy_point_name_list(self, features):
        with pytest.raises(TypeError, match="Unsupported boundary value"):
            features.within_boundary("q", BOUNDARY_NAMES)

    def test_rejects_legacy_vertex_list(self, features):
        with pytest.raises(TypeError, match="Unsupported boundary value"):
            features.within_boundary("q", STATIC_BOUNDARY)


class TestDistanceToBoundaryNewApi:
    def test_accepts_static_boundary_object(self, features):
        boundary = features.define_static_boundary(BOUNDARY_NAMES)
        res = features.distance_to_boundary("q", boundary)
        assert res.iloc[0] == pytest.approx(2 * np.sqrt(2))

    def test_accepts_dynamic_boundary_object(self, features):
        boundary = features.define_dynamic_boundary(BOUNDARY_NAMES)
        res = features.distance_to_boundary("q", boundary)
        assert res.iloc[0] == pytest.approx(2 * np.sqrt(2))

    def test_accepts_stored_boundary_name(self, features):
        features.define_dynamic_boundary(BOUNDARY_NAMES, name="tri_dyn", overwrite=True)
        res = features.distance_to_boundary("q", "tri_dyn")
        assert res.iloc[0] == pytest.approx(2 * np.sqrt(2))

    def test_rejects_legacy_point_name_list(self, features):
        with pytest.raises(TypeError, match="Unsupported boundary value"):
            features.distance_to_boundary("q", BOUNDARY_NAMES)

    def test_rejects_legacy_vertex_list(self, features):
        with pytest.raises(TypeError, match="Unsupported boundary value"):
            features.distance_to_boundary("q", STATIC_BOUNDARY)
