import types

import numpy as np
import pandas as pd

from py3r.behaviour.animation.geometry_stream import (
    build_geometry_stream,
    build_geometry_stream_from_points,
    undo_meta_scaling_for_geometry,
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
    stream = build_geometry_stream_from_points(
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
    stream = build_geometry_stream_from_points(
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


def test_undo_meta_scaling_for_geometry_inverts_meta_scaling():
    df = pd.DataFrame(
        {
            "nose.x": [20.0],
            "nose.y": [10.0],
            "nose.z": [9.0],
        }
    )
    meta = {"aspectratio_correction": 2.0, "rescale_factor": {"x": 4.0, "y": 2.0}}
    out = undo_meta_scaling_for_geometry(df, meta, dims=("x", "y"))

    assert float(out["nose.x"].iloc[0]) == 2.5
    assert float(out["nose.y"].iloc[0]) == 5.0
    assert float(out["nose.z"].iloc[0]) == 9.0


def test_build_geometry_stream_wrapper_supports_3d_points():
    df = pd.DataFrame(
        {
            "p1.x": [0.0, 1.0],
            "p1.y": [0.0, 0.0],
            "p1.z": [0.0, 0.5],
            "p2.x": [1.0, 2.0],
            "p2.y": [0.0, 0.0],
            "p2.z": [0.0, 0.5],
        },
        index=pd.Index([10, 11], name="frame"),
    )
    stream = build_geometry_stream(
        df,
        point_names=["p1"],
        lines=[("p1", "p2")],
        dims=("x", "y", "z"),
        view={"azim": 45.0, "elev": 30.0, "proj": "ortho"},
        pixel_coords=False,
        frame_ids=df.index.to_numpy(copy=True),
    )

    frame = stream.get_frame(0)
    assert stream.frame_count == 2
    assert stream.frame_ids.tolist() == [10, 11]
    assert frame.shape[2] == 3
