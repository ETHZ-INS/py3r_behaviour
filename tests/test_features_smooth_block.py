from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from py3r.behaviour.features.features import Features
from py3r.behaviour.tracking.tracking import Tracking
from py3r.behaviour.util import series_utils


def _make_features(handle: str = "A", n_frames: int = 8) -> Features:
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


def test_smooth_block_warns_and_uses_new_pipeline():
    f = _make_features()
    s = pd.Series(["A", "A", np.nan, "A", "B", np.nan, "B", "B"], index=f.tracking.data.index)
    f.store(s, "state")

    expected = series_utils.block_fill(
        series_utils.block_filter(s, min_block=2),
        max_gap=2,
        direction="both",
        require_same_label=True,
    )

    with pytest.warns(UserWarning, match="Legacy block behavior"):
        out = f.smooth("state", method="block", window=2, inplace=False)

    assert out.tolist() == expected.tolist()
    # inplace=False should still leave source feature unchanged.
    assert f.data["state"].tolist() == s.tolist()


def test_smooth_block_accepts_fill_kwargs():
    f = _make_features()
    s = pd.Series(["A", np.nan, "B"], index=f.tracking.data.index[:3])
    f.store(s, "state2")

    with pytest.warns(UserWarning, match="Legacy block behavior"):
        out = f.smooth(
            "state2",
            method="block",
            window=1,
            inplace=False,
            fill_direction="backward",
            fill_require_same_label=True,
        )

    # Backward fill should select right label for the single-frame gap.
    assert out.tolist() == ["A", "B", "B"]
