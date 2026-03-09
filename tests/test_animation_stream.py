import types

import numpy as np
import pandas as pd

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
