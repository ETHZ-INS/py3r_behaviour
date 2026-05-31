"""Tests for the axis asset API in Features.

Covers:
  - StaticAxis / DynamicAxis factory methods and offset transform helpers
  - Generic asset access (get_asset, list_assets, _resolve_axis_ref)
  - Type guards: within_boundary / distance_to_boundary reject axis objects
  - distance_to_axis: unsigned (2-D and 3-D), signed (2-D), error paths
  - Serialisation: to_dict / from_dict roundtrip for both axis types
  - axes_to_arrays: shapes and per-frame values
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from py3r.behaviour.features.axis import (
    DynamicAxis,
    StaticAxis,
    _transform_axis_endpoints,
    _transform_axis_per_frame,
)
from py3r.behaviour.features.features import Features
from py3r.behaviour.tracking.tracking import Tracking

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tracking_2d():
    """4-frame session with a horizontal axis (a→b along y=5) and a query point.

    Frame layout:
      frame 0: q above the axis  →  unsigned dist = 3, signed = −3 (left)
      frame 1: q on the axis     →  unsigned dist = 0, signed = 0
      frame 2: q below the axis  →  unsigned dist = 3, signed = +3 (right)
      frame 3: axis ref pts NaN  →  distance should be NaN
    """
    data = pd.DataFrame(
        {
            "a.x": [0.0, 0.0, 0.0, np.nan],
            "a.y": [5.0, 5.0, 5.0, 5.0],
            "b.x": [10.0, 10.0, 10.0, np.nan],
            "b.y": [5.0, 5.0, 5.0, 5.0],
            "q.x": [5.0, 5.0, 5.0, 5.0],
            "q.y": [8.0, 5.0, 2.0, 3.0],
        },
        index=pd.RangeIndex(4, name="frame"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Tracking(data, {"fps": 30.0}, handle="test")


def _make_tracking_3d():
    """2-frame session with a 3-D axis and a query point."""
    data = pd.DataFrame(
        {
            "a.x": [0.0, 0.0],
            "a.y": [0.0, 0.0],
            "a.z": [0.0, 0.0],
            "b.x": [1.0, 1.0],
            "b.y": [0.0, 0.0],
            "b.z": [0.0, 0.0],
            # q at (0, 3, 4): perpendicular distance to x-axis = sqrt(9+16)=5
            "q.x": [0.0, 0.0],
            "q.y": [3.0, 3.0],
            "q.z": [4.0, 4.0],
        },
        index=pd.RangeIndex(2, name="frame"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Tracking(data, {"fps": 30.0}, handle="test3d")


@pytest.fixture
def f2d():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Features(_make_tracking_2d())


@pytest.fixture
def f3d():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Features(_make_tracking_3d())


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------


class TestTransformAxisEndpoints:
    def test_zero_offset_returns_copy(self):
        A = np.array([0.0, 0.0])
        B = np.array([10.0, 0.0])
        At, Bt = _transform_axis_endpoints(A, B, offset=0.0)
        assert np.allclose(At, A)
        assert np.allclose(Bt, B)

    def test_positive_offset_shifts_right(self):
        # Rightward axis A=(0,0)→B=(10,0); right-hand perp = (0,−1)
        A = np.array([0.0, 0.0])
        B = np.array([10.0, 0.0])
        At, Bt = _transform_axis_endpoints(A, B, offset=2.0)
        assert np.allclose(At, [0.0, -2.0])
        assert np.allclose(Bt, [10.0, -2.0])

    def test_negative_offset_shifts_left(self):
        A = np.array([0.0, 0.0])
        B = np.array([10.0, 0.0])
        At, Bt = _transform_axis_endpoints(A, B, offset=-3.0)
        assert np.allclose(At, [0.0, 3.0])
        assert np.allclose(Bt, [10.0, 3.0])

    def test_non_2d_with_offset_raises(self):
        A = np.array([0.0, 0.0, 0.0])
        B = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="2-D"):
            _transform_axis_endpoints(A, B, offset=1.0)

    def test_degenerate_axis_with_offset_raises(self):
        A = np.array([5.0, 5.0])
        with pytest.raises(ValueError, match="zero-length"):
            _transform_axis_endpoints(A, A.copy(), offset=1.0)


class TestTransformAxisPerFrame:
    def test_zero_offset_returns_copy(self):
        arr = np.array([[[0.0, 0.0], [10.0, 0.0]], [[1.0, 1.0], [11.0, 1.0]]], dtype=float)
        out = _transform_axis_per_frame(arr, offset=0.0)
        assert np.allclose(out, arr)
        assert out is not arr

    def test_positive_offset_shifts_right_per_frame(self):
        # Rightward axis each frame; perp_right = (0,−1); offset=2 → shift (0,−2)
        arr = np.zeros((3, 2, 2), dtype=float)
        arr[:, 1, 0] = 10.0  # B.x = 10 for all frames
        out = _transform_axis_per_frame(arr, offset=2.0)
        assert np.allclose(out[:, 0, 1], -2.0)  # A.y shifted to −2
        assert np.allclose(out[:, 1, 1], -2.0)  # B.y shifted to −2

    def test_degenerate_frame_gets_zero_shift(self):
        # A == B for frame 0 → no shift, no error
        arr = np.zeros((2, 2, 2), dtype=float)
        arr[1, 1, 0] = 10.0  # only frame 1 has a valid direction
        out = _transform_axis_per_frame(arr, offset=2.0)
        assert np.allclose(out[0], 0.0)  # degenerate: no shift applied

    def test_non_2d_with_offset_raises(self):
        arr = np.zeros((2, 2, 3), dtype=float)
        with pytest.raises(ValueError, match="2-D"):
            _transform_axis_per_frame(arr, offset=1.0)


# ---------------------------------------------------------------------------
# Factory: define_static_axis
# ---------------------------------------------------------------------------


class TestDefineStaticAxis:
    def test_returns_static_axis(self, f2d):
        ax = f2d.define_static_axis("a", "b")
        assert isinstance(ax, StaticAxis)

    def test_vertices_match_medians(self, f2d):
        ax = f2d.define_static_axis("a", "b")
        # Median of a: x=0, y=5; median of b: x=10, y=5 (frame 3 NaN ignored)
        assert np.allclose(ax.vertices[0], [0.0, 5.0])
        assert np.allclose(ax.vertices[1], [10.0, 5.0])

    def test_source_points_recorded(self, f2d):
        ax = f2d.define_static_axis("a", "b")
        assert ax.source_points == ("a", "b")

    def test_dims_stored(self, f2d):
        ax = f2d.define_static_axis("a", "b", dims=("x", "y"))
        assert ax.dims == ("x", "y")

    def test_registered_by_name(self, f2d):
        ax = f2d.define_static_axis("a", "b", name="midline")
        assert f2d.get_asset("midline") is ax

    def test_offset_baked_into_vertices(self, f2d):
        ax = f2d.define_static_axis("a", "b", offset=2.0)
        # Rightward axis along y=5; perp_right=(0,−1); offset=2 → y shifts to 3
        assert np.allclose(ax.vertices[0][1], 3.0)
        assert np.allclose(ax.vertices[1][1], 3.0)

    def test_overwrite_false_raises_on_duplicate(self, f2d):
        f2d.define_static_axis("a", "b", name="ax")
        with pytest.raises(ValueError, match="already exists"):
            f2d.define_static_axis("a", "b", name="ax", overwrite=False)

    def test_overwrite_true_replaces(self, f2d):
        f2d.define_static_axis("a", "b", name="ax")
        ax2 = f2d.define_static_axis("a", "b", offset=1.0, name="ax", overwrite=True)
        assert f2d.get_asset("ax") is ax2


# ---------------------------------------------------------------------------
# Factory: import_static_axis
# ---------------------------------------------------------------------------


class TestImportStaticAxis:
    def test_returns_static_axis(self, f2d):
        ax = f2d.import_static_axis([(0.0, 0.0), (1.0, 0.0)])
        assert isinstance(ax, StaticAxis)

    def test_vertices_stored(self, f2d):
        ax = f2d.import_static_axis([(1.0, 2.0), (3.0, 4.0)], name="custom")
        assert ax.vertices == ((1.0, 2.0), (3.0, 4.0))

    def test_wrong_vertex_count_raises(self, f2d):
        with pytest.raises(ValueError, match="2 reference points"):
            f2d.import_static_axis([(0.0, 0.0)])

    def test_three_vertices_raises(self, f2d):
        with pytest.raises(ValueError, match="2 reference points"):
            f2d.import_static_axis([(0, 0), (1, 0), (2, 0)])


# ---------------------------------------------------------------------------
# Factory: define_dynamic_axis
# ---------------------------------------------------------------------------


class TestDefineDynamicAxis:
    def test_returns_dynamic_axis(self, f2d):
        ax = f2d.define_dynamic_axis("a", "b")
        assert isinstance(ax, DynamicAxis)

    def test_points_stored(self, f2d):
        ax = f2d.define_dynamic_axis("a", "b")
        assert ax.points == ("a", "b")

    def test_dims_stored(self, f2d):
        ax = f2d.define_dynamic_axis("a", "b", dims=("x", "y"))
        assert ax.dims == ("x", "y")

    def test_offset_stored(self, f2d):
        ax = f2d.define_dynamic_axis("a", "b", offset=1.5)
        assert ax.offset == pytest.approx(1.5)

    def test_default_offset_is_zero(self, f2d):
        ax = f2d.define_dynamic_axis("a", "b")
        assert ax.offset == 0.0

    def test_registered_by_name(self, f2d):
        ax = f2d.define_dynamic_axis("a", "b", name="dyn_ax")
        assert f2d.get_asset("dyn_ax") is ax


# ---------------------------------------------------------------------------
# Asset access
# ---------------------------------------------------------------------------


class TestAssetAccess:
    def test_get_asset_returns_registered_axis(self, f2d):
        ax = f2d.define_static_axis("a", "b", name="midline")
        assert f2d.get_asset("midline") is ax

    def test_list_assets_includes_axis(self, f2d):
        f2d.define_static_axis("a", "b", name="midline")
        tbl = f2d.list_assets()
        assert "midline" in tbl.index
        assert tbl.loc["midline", "asset_type"] == "StaticAxis"

    def test_list_assets_includes_dynamic_axis(self, f2d):
        f2d.define_dynamic_axis("a", "b", name="dyn")
        tbl = f2d.list_assets()
        assert tbl.loc["dyn", "asset_type"] == "DynamicAxis"

    def test_resolve_axis_ref_by_name(self, f2d):
        ax = f2d.define_static_axis("a", "b", name="midline")
        assert f2d._resolve_axis_ref("midline") is ax

    def test_resolve_axis_ref_by_object(self, f2d):
        ax = f2d.define_static_axis("a", "b")
        assert f2d._resolve_axis_ref(ax) is ax

    def test_resolve_axis_ref_rejects_boundary(self, f2d):
        f2d.define_static_boundary(["a", "b", "q"], name="tri")
        with pytest.raises(TypeError, match="StaticAxis"):
            f2d._resolve_axis_ref("tri")

    def test_within_boundary_rejects_axis(self, f2d):
        ax = f2d.define_static_axis("a", "b")
        with pytest.raises(TypeError, match="distance_to_axis"):
            f2d.within_boundary("q", ax)

    def test_distance_to_boundary_rejects_axis(self, f2d):
        ax = f2d.define_dynamic_axis("a", "b")
        with pytest.raises(TypeError, match="distance_to_axis"):
            f2d.distance_to_boundary("q", ax)


# ---------------------------------------------------------------------------
# distance_to_axis: unsigned
# ---------------------------------------------------------------------------


class TestDistanceToAxisUnsigned:
    def test_static_axis_above(self, f2d):
        ax = f2d.define_static_axis("a", "b")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = f2d.distance_to_axis("q", ax)
        assert res.iloc[0] == pytest.approx(3.0)

    def test_static_axis_on_axis(self, f2d):
        ax = f2d.define_static_axis("a", "b")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = f2d.distance_to_axis("q", ax)
        assert res.iloc[1] == pytest.approx(0.0)

    def test_static_axis_below(self, f2d):
        ax = f2d.define_static_axis("a", "b")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = f2d.distance_to_axis("q", ax)
        assert res.iloc[2] == pytest.approx(3.0)

    def test_dynamic_axis_matches_static(self, f2d):
        static_ax = f2d.define_static_axis("a", "b")
        dyn_ax = f2d.define_dynamic_axis("a", "b")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_s = f2d.distance_to_axis("q", static_ax)
            res_d = f2d.distance_to_axis("q", dyn_ax)
        # Frames 0-2: identical (frame 3 has NaN ref pts, dyn will be NaN)
        assert res_s.iloc[0] == pytest.approx(res_d.iloc[0])
        assert res_s.iloc[1] == pytest.approx(res_d.iloc[1])

    def test_dynamic_axis_nan_ref_pts_gives_nan(self, f2d):
        dyn_ax = f2d.define_dynamic_axis("a", "b")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = f2d.distance_to_axis("q", dyn_ax)
        assert np.isnan(res.iloc[3])

    def test_3d_axis_distance(self, f3d):
        ax = f3d.define_static_axis("a", "b", dims=("x", "y", "z"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = f3d.distance_to_axis("q", ax)
        assert res.iloc[0] == pytest.approx(5.0)

    def test_result_length_matches_tracking(self, f2d):
        ax = f2d.define_static_axis("a", "b")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = f2d.distance_to_axis("q", ax)
        assert len(res) == len(f2d.tracking.data)

    def test_accepts_axis_by_name(self, f2d):
        f2d.define_static_axis("a", "b", name="midline")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = f2d.distance_to_axis("q", "midline")
        assert res.iloc[0] == pytest.approx(3.0)

    def test_result_name_format(self, f2d):
        f2d.define_static_axis("a", "b", name="midline")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = f2d.distance_to_axis("q", "midline")
        assert "distance_to_axis" in res.name
        assert "midline" in res.name
        assert "_signed" not in res.name


# ---------------------------------------------------------------------------
# distance_to_axis: signed
# ---------------------------------------------------------------------------


class TestDistanceToAxisSigned:
    def test_positive_right_of_axis(self, f2d):
        # frame 2: q below the rightward axis → right → positive
        ax = f2d.define_static_axis("a", "b")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = f2d.distance_to_axis("q", ax, signed=True)
        assert res.iloc[2] == pytest.approx(3.0)

    def test_negative_left_of_axis(self, f2d):
        # frame 0: q above the rightward axis → left → negative
        ax = f2d.define_static_axis("a", "b")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = f2d.distance_to_axis("q", ax, signed=True)
        assert res.iloc[0] == pytest.approx(-3.0)

    def test_zero_on_axis(self, f2d):
        ax = f2d.define_static_axis("a", "b")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = f2d.distance_to_axis("q", ax, signed=True)
        assert res.iloc[1] == pytest.approx(0.0)

    def test_signed_magnitude_equals_unsigned_off_axis(self, f2d):
        ax = f2d.define_static_axis("a", "b")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unsigned = f2d.distance_to_axis("q", ax, signed=False)
            signed = f2d.distance_to_axis("q", ax, signed=True)
        assert abs(signed.iloc[0]) == pytest.approx(unsigned.iloc[0])
        assert abs(signed.iloc[2]) == pytest.approx(unsigned.iloc[2])

    def test_signed_3d_raises(self, f3d):
        ax = f3d.define_static_axis("a", "b", dims=("x", "y", "z"))
        with pytest.raises(ValueError, match="signed=True requires a 2-D"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f3d.distance_to_axis("q", ax, signed=True)

    def test_signed_result_name_has_suffix(self, f2d):
        f2d.define_static_axis("a", "b", name="midline")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = f2d.distance_to_axis("q", "midline", signed=True)
        assert "_signed" in res.name

    def test_signed_meta_flag(self, f2d):
        ax = f2d.define_static_axis("a", "b")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = f2d.distance_to_axis("q", ax, signed=True)
        assert res._params["signed"] is True

    def test_unsigned_meta_flag(self, f2d):
        ax = f2d.define_static_axis("a", "b")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = f2d.distance_to_axis("q", ax, signed=False)
        assert res._params["signed"] is False


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestAxisSerialisation:
    def test_static_axis_kind_string(self, f2d):
        ax = f2d.define_static_axis("a", "b")
        assert ax.to_dict()["kind"] == "static_axis"

    def test_dynamic_axis_kind_string(self, f2d):
        ax = f2d.define_dynamic_axis("a", "b")
        assert ax.to_dict()["kind"] == "dynamic_axis"

    def test_static_axis_roundtrip(self, f2d):
        ax = f2d.define_static_axis("a", "b", name="midline", offset=1.0)
        d = ax.to_dict()
        ax2 = StaticAxis.from_dict(d)
        assert ax2.dims == ax.dims
        assert ax2.name == ax.name
        assert np.allclose(ax2.vertices[0], ax.vertices[0])
        assert np.allclose(ax2.vertices[1], ax.vertices[1])

    def test_dynamic_axis_roundtrip(self, f2d):
        ax = f2d.define_dynamic_axis("a", "b", offset=2.5, name="dyn")
        d = ax.to_dict()
        ax2 = DynamicAxis.from_dict(d)
        assert ax2.points == ax.points
        assert ax2.dims == ax.dims
        assert ax2.offset == pytest.approx(ax.offset)
        assert ax2.name == ax.name

    def test_static_axis_source_points_preserved(self, f2d):
        ax = f2d.define_static_axis("a", "b")
        d = ax.to_dict()
        ax2 = StaticAxis.from_dict(d)
        assert ax2.source_points == ("a", "b")


# ---------------------------------------------------------------------------
# axes_to_arrays
# ---------------------------------------------------------------------------


class TestAxesToArrays:
    def test_static_axis_shape(self, f2d):
        f2d.define_static_axis("a", "b", name="midline")
        arrays = f2d.axes_to_arrays(["midline"])
        assert len(arrays) == 1
        name, arr = arrays[0]
        assert name == "midline"
        assert arr.shape == (len(f2d.tracking.data), 2, 2)

    def test_static_axis_constant_across_frames(self, f2d):
        f2d.define_static_axis("a", "b", name="midline")
        _, arr = f2d.axes_to_arrays(["midline"])[0]
        # All frames should be identical for a static axis
        assert np.allclose(arr[0], arr[1])
        assert np.allclose(arr[0], arr[2])

    def test_dynamic_axis_shape(self, f2d):
        f2d.define_dynamic_axis("a", "b", name="dyn")
        _, arr = f2d.axes_to_arrays(["dyn"])[0]
        assert arr.shape == (len(f2d.tracking.data), 2, 2)

    def test_dynamic_axis_varies_with_data(self, f2d):
        # frames 0–2 have the same a/b coords, so arrays are equal
        # frame 3 has NaN → should propagate
        f2d.define_dynamic_axis("a", "b", name="dyn")
        _, arr = f2d.axes_to_arrays(["dyn"])[0]
        assert np.isnan(arr[3, 0, 0])  # a.x is NaN in frame 3

    def test_dims_mismatch_raises(self, f2d):
        f2d.define_static_axis("a", "b", dims=("x", "y"), name="midline")
        with pytest.raises(ValueError, match="dims"):
            f2d.axes_to_arrays(["midline"], dims=("x", "z"))

    def test_multiple_axes_returned_in_order(self, f2d):
        f2d.define_static_axis("a", "b", name="ax1")
        f2d.define_dynamic_axis("a", "b", name="ax2")
        arrays = f2d.axes_to_arrays(["ax1", "ax2"])
        assert [name for name, _ in arrays] == ["ax1", "ax2"]
