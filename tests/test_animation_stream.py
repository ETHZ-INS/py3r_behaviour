import types
import warnings

import numpy as np
import pandas as pd
import pytest

from py3r.behaviour.animation._projection import _clip_axis_to_canvas, _data_to_pixel_float
from py3r.behaviour.animation.animation_stream import (
    _format_overlay_value,
    build_animation_stream,
)
from py3r.behaviour.features.features import Features
from py3r.behaviour.tracking.tracking import Tracking


def _tracking_xy() -> Tracking:
    df = pd.DataFrame(
        {
            "nose.x": [10.0, 11.0, 12.0],
            "nose.y": [20.0, 21.0, 22.0],
            "tail.x": [5.0, 6.0, 7.0],
            "tail.y": [8.0, 9.0, 10.0],
        },
        index=pd.Index([100, 101, 102], name="frame"),
    )
    return Tracking(df, meta={"fps": 30.0}, handle="demo")


def test_tracking_animation_stream_calls_points_to_numpy(monkeypatch):
    t = _tracking_xy()
    captured = {}

    def fake_points_to_numpy(self, points, dims=("x", "y"), *, undo_meta_scaling=False):
        captured["points"] = list(points)
        captured["dims"] = tuple(dims)
        captured["undo_meta_scaling"] = bool(undo_meta_scaling)
        arr = np.zeros((len(self.data), len(points), len(dims)), dtype=float)
        return list(points), arr

    monkeypatch.setattr(Tracking, "points_to_numpy", fake_points_to_numpy)
    stream = t.animation_stream(
        points=["nose"],
        lines=[("tail", "nose")],
        dims=("x", "y"),
        pixel_coords=True,
    )

    assert captured["points"] == ["nose", "tail"]
    assert captured["dims"] == ("x", "y")
    assert captured["undo_meta_scaling"] is False
    assert stream.frame_count == 3
    assert stream.frame_ids.tolist() == [100, 101, 102]


def test_features_animation_stream_calls_tracking_points_to_numpy():
    t = _tracking_xy()
    f = Features(t)
    captured = {}

    def fake_points_to_numpy(self, points, dims=("x", "y"), *, undo_meta_scaling=False):
        captured["points"] = list(points)
        captured["dims"] = tuple(dims)
        captured["undo_meta_scaling"] = bool(undo_meta_scaling)
        arr = np.zeros((len(self.data), len(points), len(dims)), dtype=float)
        return list(points), arr

    t.points_to_numpy = types.MethodType(fake_points_to_numpy, t)
    stream = f.animation_stream(
        points=["nose"],
        lines=[("tail", "nose")],
        dims=("x", "y"),
        pixel_coords=True,
    )

    assert captured["points"] == ["nose", "tail"]
    assert captured["dims"] == ("x", "y")
    assert captured["undo_meta_scaling"] is False
    assert stream.frame_count == 3
    assert stream.frame_ids.tolist() == [100, 101, 102]


def test_geometry_animation_stream_read_reset_iter():
    points = np.array(
        [
            [[10.0, 10.0], [20.0, 20.0]],
            [[11.0, 11.0], [21.0, 21.0]],
        ],
        dtype=float,
    )
    stream = build_animation_stream(
        points=points,
        point_names=["nose", "tail"],
        draw_points=["nose"],
        lines=[("nose", "tail")],
        frame_ids=np.array([0, 1]),
        pixel_coords=True,
        canvas_size=(64, 48),
    )

    ok0, frame0 = stream.read()
    ok1, frame1 = stream.read()
    ok2, frame2 = stream.read()
    assert ok0 and frame0 is not None and frame0.shape == (48, 64, 3)
    assert ok1 and frame1 is not None and frame1.shape == (48, 64, 3)
    assert (ok2, frame2) == (False, None)

    stream.reset()
    first = next(stream)
    assert first.shape == (48, 64, 3)


def test_geometry_animation_stream_render_into_copy_modes():
    points = np.array([[[10.0, 10.0]]], dtype=float)
    stream = build_animation_stream(
        points=points,
        point_names=["nose"],
        draw_points=["nose"],
        frame_ids=np.array([0]),
        pixel_coords=True,
        canvas_size=(32, 24),
    )
    base = np.zeros((24, 32, 3), dtype=np.uint8)
    out_copy = stream.render_into(base, frame_idx=0, copy=True)
    assert out_copy is not base
    assert np.any(out_copy != base)

    base2 = np.zeros((24, 32, 3), dtype=np.uint8)
    out_inplace = stream.render_into(base2, frame_idx=0, copy=False)
    assert out_inplace is base2
    assert np.any(base2 != 0)


def test_tracking_points_to_numpy_undo_meta_scaling_inverts_meta_scaling():
    df = pd.DataFrame(
        {
            "nose.x": [20.0],
            "nose.y": [10.0],
            "nose.z": [9.0],
        }
    )
    meta = {"fps": 30.0, "aspectratio_correction": 2.0, "rescale_factor": {"x": 4.0, "y": 2.0}}
    tracking = Tracking(df, meta=meta, handle="demo")
    _, out = tracking.points_to_numpy(
        ["nose"],
        dims=("x", "y", "z"),
        undo_meta_scaling=True,
    )

    assert float(out[0, 0, 0]) == 2.5
    assert float(out[0, 0, 1]) == 5.0
    assert float(out[0, 0, 2]) == 9.0


def test_build_animation_stream_supports_3d_points():
    points = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.5], [2.0, 0.0, 0.5]],
        ],
        dtype=float,
    )
    stream = build_animation_stream(
        points=points,
        point_names=["p1", "p2"],
        draw_points=["p1"],
        lines=[("p1", "p2")],
        view={"azim": 45.0, "elev": 30.0, "proj": "ortho"},
        pixel_coords=False,
        frame_ids=np.array([10, 11]),
    )

    frame = stream.get_frame(0)
    assert stream.frame_count == 2
    assert stream.frame_ids.tolist() == [10, 11]
    assert frame.shape[2] == 3


def test_tracking_animation_stream_feature_text_overlay_draws():
    t = _tracking_xy()
    stream = t.animation_stream(
        points=[],
        features={"nose_x": "nose.x"},
        pixel_coords=True,
        canvas_size=(128, 64),
        style={
            "text": {
                "origin": (8, 20),
                "format": ".1f",
                "default": {"color": (255, 255, 255), "font_scale": 0.6},
            }
        },
    )
    frame = stream.get_frame(0)
    assert np.any(frame != 0)


def test_features_animation_stream_feature_text_overlay_draws():
    t = _tracking_xy()
    f = Features(t)
    f.data["speed"] = pd.Series([0.1, 0.2, 0.3], index=t.data.index)
    stream = f.animation_stream(
        points=[],
        features={"spd": "speed"},
        pixel_coords=True,
        canvas_size=(128, 64),
    )
    frame = stream.get_frame(0)
    assert np.any(frame != 0)


def test_features_animation_stream_dynamic_style_uses_features_data_only():
    t = _tracking_xy()
    f = Features(t)
    with pytest.raises(ValueError, match="not found in Features\\.data"):
        f.animation_stream(
            points=["nose"],
            pixel_coords=True,
            style={"points": {"nose": {"radius": {"from": "nose.x", "map": {0: 1, 1: 3}}}}},
        )


def test_overlay_value_formatting_bool_and_scientific():
    assert _format_overlay_value(True, ".3f") == "True"
    assert _format_overlay_value(np.bool_(False), ".3f") == "False"
    assert _format_overlay_value(0.0001234, ".2e") == "1.23e-04"
    assert _format_overlay_value(1.0, ".3f", as_bool=True) == "True"
    assert _format_overlay_value(0.0, ".3f", as_bool=True) == "False"


def test_tracking_animation_stream_feature_spacer_entries_allowed():
    t = _tracking_xy()
    stream = t.animation_stream(
        points=[],
        features=["nose.x", None, "tail.x", ""],
        pixel_coords=True,
        canvas_size=(128, 64),
    )
    frame = stream.get_frame(0)
    assert frame.shape == (64, 128, 3)


def test_text_colormap_precompute_runs_with_matplotlib():
    t = _tracking_xy()
    stream = t.animation_stream(
        points=[],
        features={"nose_x": "nose.x"},
        pixel_coords=True,
        style={
            "text": {
                "default": {"color": (255, 255, 255)},
                "nose_x": {"cmap": "viridis", "vmin": 0.0, "vmax": 20.0},
            }
        },
    )
    frame = stream.get_frame(0)
    assert frame.ndim == 3


def test_text_colormap_handles_pd_na_values():
    t = _tracking_xy()
    t.data["nullable_metric"] = pd.Series([1.0, pd.NA, 3.0], index=t.data.index, dtype="Float64")
    stream = t.animation_stream(
        points=[],
        features={"metric": "nullable_metric"},
        pixel_coords=True,
        style={
            "text": {
                "default": {"color": (255, 255, 255)},
                "metric": {"cmap": "viridis", "vmin": 0.0, "vmax": 5.0, "nan_color": (10, 20, 30)},
            }
        },
    )
    frame = stream.get_frame(1)
    assert frame.ndim == 3


def test_text_colormap_uses_dynamic_vmin_per_frame():
    t = _tracking_xy()
    t.data["metric"] = pd.Series([10.0, 10.0, 10.0], index=t.data.index)
    t.data["flag"] = pd.Series([0, 1, 0], index=t.data.index)
    stream = t.animation_stream(
        points=[],
        features={"metric": "metric"},
        pixel_coords=True,
        style={
            "text": {
                "format": ".1f",
                "default": {"color": (255, 255, 255)},
                "metric": {
                    "cmap": "viridis",
                    "vmin": {"from": "flag", "map": {0: 0.0, 1: 10.0}},
                    "vmax": 20.0,
                },
            }
        },
    )
    frame0 = stream.get_frame(0)
    frame1 = stream.get_frame(1)
    assert not np.array_equal(frame0, frame1)


def test_text_outline_and_panel_render():
    t = _tracking_xy()
    stream = t.animation_stream(
        points=[],
        features={"speed": "nose.x"},
        pixel_coords=True,
        canvas_size=(128, 64),
        style={
            "text": {
                "origin": (8, 20),
                "default": {
                    "color": (255, 255, 255),
                    "outline_color": (0, 0, 0),
                    "outline_thickness": 2,
                },
                "panel": {"enabled": True, "alpha": 0.5, "padding": 6, "color": (0, 0, 0)},
            }
        },
    )
    frame = stream.get_frame(0)
    assert frame.shape == (64, 128, 3)


def test_dynamic_boundary_style_from_feature_source():
    points = np.empty((3, 0, 2), dtype=float)
    poly = np.array([[10.0, 10.0], [50.0, 10.0], [50.0, 40.0], [10.0, 40.0]], dtype=float)
    boundary_arrays = [("zone", np.repeat(poly[None, :, :], 3, axis=0))]
    stream = build_animation_stream(
        points=points,
        point_names=[],
        frame_ids=np.array([0, 1, 2]),
        boundary_arrays=boundary_arrays,
        canvas_size=(64, 48),
        pixel_coords=True,
        style={
            "boundaries": {
                "zone": {
                    "fill_color": (255, 255, 255),
                    "fill_alpha": {"from": "flag", "map": {0: 0.0, 1: 1.0}},
                    "edge_width": 0,
                }
            }
        },
        style_sources={"flag": np.array([0, 1, 0], dtype=float)},
    )
    frame0 = stream.get_frame(0)
    frame1 = stream.get_frame(1)
    assert frame1.sum() > frame0.sum()


@pytest.mark.parametrize(
    "fill_alpha_map,expect_raise",
    [
        ({False: 0.1, True: 0.8, "default": 0.5}, False),
        ({False: 0.1, True: 0.8, None: 0.5}, False),
        ({False: 0.1, True: 0.8}, True),
    ],
)
@pytest.mark.parametrize("include_nan_color", [True, False])
def test_features_dynamic_boundary_style_na_behavior_matrix(
    fill_alpha_map,
    expect_raise,
    include_nan_color,
):
    df = pd.DataFrame(
        {
            "tl.x": [10.0, 10.0, 10.0],
            "tl.y": [10.0, 10.0, 10.0],
            "tr.x": [50.0, 50.0, 50.0],
            "tr.y": [10.0, 10.0, 10.0],
            "br.x": [50.0, 50.0, 50.0],
            "br.y": [40.0, 40.0, 40.0],
            "bl.x": [10.0, 10.0, 10.0],
            "bl.y": [40.0, 40.0, 40.0],
            "bodycentre.x": [30.0, 30.0, 30.0],
            "bodycentre.y": [25.0, 25.0, 25.0],
        },
        index=pd.Index([0, 1, 2], name="frame"),
    )
    tracking = Tracking(df, meta={"fps": 30.0}, handle="demo")
    features = Features(tracking)
    _ = features.define_static_boundary(["tl", "tr", "br", "bl"], name="arena")
    features.data["nullable_bool"] = pd.Series(
        [True, pd.NA, False],
        index=tracking.data.index,
        dtype="boolean",
    )
    features.data["nullable_float"] = pd.Series(
        [0.0, np.nan, 1.0],
        index=tracking.data.index,
        dtype="Float64",
    )
    fill_color_spec = {
        "from": "nullable_float",
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 1.0,
    }
    if include_nan_color:
        fill_color_spec["nan_color"] = (10, 20, 30)

    if expect_raise:
        with pytest.raises(
            ValueError,
            match=(
                "Dynamic style boundaries\\.arena\\.fill_alpha from source nullable_bool, "
                "resolved to value NA/None not specified in map"
            ),
        ):
            features.animation_stream(
                points=["bodycentre"],
                boundaries=["arena"],
                pixel_coords=True,
                canvas_size=(64, 48),
                style={
                    "boundaries": {
                        "arena": {
                            "fill_color": fill_color_spec,
                            "fill_alpha": {"from": "nullable_bool", "map": fill_alpha_map},
                            "edge_width": 0,
                        }
                    }
                },
            )
        return

    stream = features.animation_stream(
        points=["bodycentre"],
        boundaries=["arena"],
        pixel_coords=True,
        canvas_size=(64, 48),
        style={
            "boundaries": {
                "arena": {
                    "fill_color": fill_color_spec,
                    "fill_alpha": {"from": "nullable_bool", "map": fill_alpha_map},
                    "edge_width": 0,
                }
            }
        },
    )
    frame0 = stream.get_frame(0)
    assert frame0.shape == (48, 64, 3)
    frame1 = stream.get_frame(1)
    assert frame1.shape == (48, 64, 3)


# ---------------------------------------------------------------------------
# _clip_axis_to_canvas unit tests
# ---------------------------------------------------------------------------


class TestClipAxisToCanvas:
    """Unit tests for the canvas-edge clipping of an infinite axis."""

    W, H = 64, 48

    def test_horizontal_axis_clips_to_left_and_right_edges(self):
        # Line through y=24, reference points outside canvas on both sides
        p1, p2 = np.array([-10.0, 24.0]), np.array([90.0, 24.0])
        result = _clip_axis_to_canvas(p1, p2, self.W, self.H)
        assert result is not None
        cp1, cp2 = result
        assert cp1[1] == 24 and cp2[1] == 24
        assert min(cp1[0], cp2[0]) == 0
        assert max(cp1[0], cp2[0]) == self.W - 1

    def test_vertical_axis_clips_to_top_and_bottom_edges(self):
        p1, p2 = np.array([32.0, -10.0]), np.array([32.0, 58.0])
        result = _clip_axis_to_canvas(p1, p2, self.W, self.H)
        assert result is not None
        cp1, cp2 = result
        assert cp1[0] == 32 and cp2[0] == 32
        assert min(cp1[1], cp2[1]) == 0
        assert max(cp1[1], cp2[1]) == self.H - 1

    def test_diagonal_spanning_canvas_corners(self):
        p1, p2 = np.array([0.0, 0.0]), np.array([float(self.W - 1), float(self.H - 1)])
        result = _clip_axis_to_canvas(p1, p2, self.W, self.H)
        assert result is not None
        pts = sorted([tuple(result[0]), tuple(result[1])])
        assert pts[0] == (0, 0)
        assert pts[1] == (self.W - 1, self.H - 1)

    def test_axis_entirely_off_canvas_returns_none(self):
        # Line from (-10, 60) to (20, 70): stays above (y > H−1) the canvas
        p1, p2 = np.array([-10.0, 60.0]), np.array([20.0, 70.0])
        result = _clip_axis_to_canvas(p1, p2, self.W, self.H)
        assert result is None

    def test_coincident_points_returns_none(self):
        p = np.array([32.0, 24.0])
        result = _clip_axis_to_canvas(p, p.copy(), self.W, self.H)
        assert result is None

    def test_reference_points_inside_canvas_still_works(self):
        # Both reference points inside canvas: result should still be two canvas-edge points
        p1, p2 = np.array([10.0, 24.0]), np.array([50.0, 24.0])
        result = _clip_axis_to_canvas(p1, p2, self.W, self.H)
        assert result is not None
        cp1, cp2 = result
        # Clipped endpoints should be on left and right edges for a horizontal axis
        assert min(cp1[0], cp2[0]) == 0
        assert max(cp1[0], cp2[0]) == self.W - 1


# ---------------------------------------------------------------------------
# _data_to_pixel_float unit tests
# ---------------------------------------------------------------------------


class TestDataToPixelFloat:
    def test_min_corner_maps_to_pixel_origin(self):
        bounds = (0.0, 10.0, 0.0, 10.0)
        pts = np.array([[0.0, 0.0]])
        out = _data_to_pixel_float(pts, width=11, height=11, bounds=bounds, pixel_coords=False)
        # x=0 → px=0; y=0 → py=height-1=10 (y-axis is flipped)
        assert out[0, 0] == pytest.approx(0.0)
        assert out[0, 1] == pytest.approx(10.0)

    def test_max_corner_maps_to_pixel_max(self):
        bounds = (0.0, 10.0, 0.0, 10.0)
        pts = np.array([[10.0, 10.0]])
        out = _data_to_pixel_float(pts, width=11, height=11, bounds=bounds, pixel_coords=False)
        assert out[0, 0] == pytest.approx(10.0)
        assert out[0, 1] == pytest.approx(0.0)

    def test_pixel_coords_passthrough(self):
        bounds = (0.0, 1.0, 0.0, 1.0)
        pts = np.array([[15.7, 30.2]])
        out = _data_to_pixel_float(pts, width=64, height=48, bounds=bounds, pixel_coords=True)
        assert np.allclose(out, pts)

    def test_out_of_bounds_values_preserved(self):
        # Reference points well outside canvas range should keep their (large) pixel values
        bounds = (0.0, 10.0, 0.0, 10.0)
        pts = np.array([[-5.0, 15.0]])  # outside on both axes
        out = _data_to_pixel_float(pts, width=64, height=48, bounds=bounds, pixel_coords=False)
        # Should not be clipped; x should be negative, y should be negative
        assert out[0, 0] < 0
        assert out[0, 1] < 0


# ---------------------------------------------------------------------------
# Axis rendering via build_animation_stream
# ---------------------------------------------------------------------------


def _empty_points(n_frames: int) -> np.ndarray:
    return np.empty((n_frames, 0, 2), dtype=float)


def test_axis_array_renders_green_pixels_on_canvas():
    # Horizontal axis through y=24 of a 64×48 canvas (pixel coords)
    n = 3
    axis_arr = np.tile(np.array([[[0.0, 24.0], [63.0, 24.0]]]), (n, 1, 1))
    stream = build_animation_stream(
        points=_empty_points(n),
        point_names=[],
        frame_ids=np.arange(n),
        axis_arrays=[("midline", axis_arr)],
        canvas_size=(64, 48),
        pixel_coords=True,
    )
    frame = stream.get_frame(0)
    assert frame.shape == (48, 64, 3)
    # Default axis style is green (B=0, G=255, R=0 in BGR); row 24 should have green pixels
    assert np.any(frame[24, :, 1] == 255)


def test_axis_array_nan_reference_frame_renders_blank():
    # Frame 0 has a NaN reference point; frame 1 is valid
    axis_arr = np.array(
        [
            [[np.nan, 24.0], [63.0, 24.0]],  # invalid: NaN
            [[0.0, 24.0], [63.0, 24.0]],  # valid
        ],
        dtype=float,
    )
    stream = build_animation_stream(
        points=_empty_points(2),
        point_names=[],
        frame_ids=np.arange(2),
        axis_arrays=[("midline", axis_arr)],
        canvas_size=(64, 48),
        pixel_coords=True,
    )
    frame0 = stream.get_frame(0)
    frame1 = stream.get_frame(1)
    assert np.all(frame0 == 0), "NaN frame should produce a blank canvas"
    assert np.any(frame1 != 0), "Valid frame should have rendered pixels"


def test_axis_reference_points_outside_canvas_still_renders():
    # Vertical axis with reference points far above and below the canvas
    n = 1
    axis_arr = np.array([[[32.0, -1000.0], [32.0, 1000.0]]], dtype=float)
    stream = build_animation_stream(
        points=_empty_points(n),
        point_names=[],
        frame_ids=np.arange(n),
        axis_arrays=[("vert", axis_arr)],
        canvas_size=(64, 48),
        pixel_coords=True,
    )
    frame = stream.get_frame(0)
    # Column 32 should contain green pixels along its full height
    assert np.any(frame[:, 32, 1] == 255)


def test_axis_custom_style_applied():
    n = 2
    axis_arr = np.tile(np.array([[[0.0, 24.0], [63.0, 24.0]]]), (n, 1, 1))
    stream = build_animation_stream(
        points=_empty_points(n),
        point_names=[],
        frame_ids=np.arange(n),
        axis_arrays=[("midline", axis_arr)],
        canvas_size=(64, 48),
        pixel_coords=True,
        style={"axes": {"midline": {"edge_color": (255, 0, 0), "edge_width": 2}}},
    )
    frame = stream.get_frame(0)
    # OpenCV uses BGR; (255, 0, 0) → blue (channel 0 = 255, channel 1 = 0)
    assert np.any(frame[24, :, 0] == 255)
    assert np.all(frame[24, frame[24, :, 0] == 255, 1] == 0)


def test_axis_edge_width_zero_produces_blank_canvas():
    n = 1
    axis_arr = np.array([[[0.0, 24.0], [63.0, 24.0]]], dtype=float)
    stream = build_animation_stream(
        points=_empty_points(n),
        point_names=[],
        frame_ids=np.arange(n),
        axis_arrays=[("midline", axis_arr)],
        canvas_size=(64, 48),
        pixel_coords=True,
        style={"axes": {"midline": {"edge_width": 0}}},
    )
    frame = stream.get_frame(0)
    assert np.all(frame == 0)


def test_multiple_axis_arrays_all_rendered():
    n = 1
    h_axis = np.array([[[0.0, 16.0], [63.0, 16.0]]], dtype=float)
    v_axis = np.array([[[32.0, 0.0], [32.0, 47.0]]], dtype=float)
    stream = build_animation_stream(
        points=_empty_points(n),
        point_names=[],
        frame_ids=np.arange(n),
        axis_arrays=[("h", h_axis), ("v", v_axis)],
        canvas_size=(64, 48),
        pixel_coords=True,
    )
    frame = stream.get_frame(0)
    # Row 16 should have green (horizontal axis)
    assert np.any(frame[16, :, 1] == 255)
    # Column 32 should have green (vertical axis)
    assert np.any(frame[:, 32, 1] == 255)


def test_features_animation_stream_axes_param():
    """Features.animation_stream accepts axes= and renders the axis."""
    df = pd.DataFrame(
        {
            "a.x": [0.0, 0.0, 0.0],
            "a.y": [24.0, 24.0, 24.0],
            "b.x": [63.0, 63.0, 63.0],
            "b.y": [24.0, 24.0, 24.0],
            "q.x": [32.0, 32.0, 32.0],
            "q.y": [30.0, 30.0, 30.0],
        },
        index=pd.RangeIndex(3, name="frame"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = Tracking(df, {"fps": 30.0}, handle="demo")
    f = Features(t)
    f.define_static_axis("a", "b", name="midline")
    stream = f.animation_stream(
        points=[],
        axes=["midline"],
        canvas_size=(64, 48),
        pixel_coords=True,
    )
    frame = stream.get_frame(0)
    assert frame.shape == (48, 64, 3)
    assert np.any(frame != 0)
