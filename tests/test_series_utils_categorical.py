from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from py3r.behaviour.util.series_utils import block_fill, block_filter


def test_block_filter_marks_short_observed_blocks_only():
    s = pd.Series(["A", "A", np.nan, np.nan, "A"], dtype="object")
    out = block_filter(s, min_block=2)
    expected = pd.Series(["A", "A", np.nan, np.nan, np.nan], dtype="object")
    assert_series_equal(out, expected)


def test_block_filter_treats_nan_as_block_separator():
    s = pd.Series(["A", "A", np.nan, "A", "A"], dtype="object")
    out = block_filter(s, min_block=3)
    expected = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], dtype="object")
    assert_series_equal(out, expected)


def test_block_fill_bridges_only_short_bracketed_gaps():
    s_short = pd.Series(["A", np.nan, "A"], dtype="object")
    out_short = block_fill(
        s_short,
        max_gap=1,
        direction="both",
        require_same_label=True,
    )
    expected_short = pd.Series(["A", "A", "A"], dtype="object")
    assert_series_equal(out_short, expected_short)

    s_long = pd.Series(["A", np.nan, np.nan, "A"], dtype="object")
    out_long = block_fill(
        s_long,
        max_gap=1,
        direction="both",
        require_same_label=True,
    )
    assert_series_equal(out_long, s_long)


def test_block_fill_respects_direction_and_label_requirement():
    s = pd.Series(["A", np.nan, "B"], dtype="object")

    out_both_strict = block_fill(
        s,
        max_gap=1,
        direction="both",
        require_same_label=True,
    )
    assert_series_equal(out_both_strict, s)

    out_both_relaxed = block_fill(
        s,
        max_gap=1,
        direction="both",
        require_same_label=False,
    )
    expected_both_relaxed = pd.Series(["A", "A", "B"], dtype="object")
    assert_series_equal(out_both_relaxed, expected_both_relaxed)

    out_backward = block_fill(
        s,
        max_gap=1,
        direction="backward",
        require_same_label=True,
    )
    expected_backward = pd.Series(["A", "B", "B"], dtype="object")
    assert_series_equal(out_backward, expected_backward)


def test_block_filter_then_fill_keeps_long_gap_missing():
    s = pd.Series(["A", "A", np.nan, np.nan, "A"], dtype="object")
    smoothed = block_filter(s, min_block=2)
    out = block_fill(
        smoothed,
        max_gap=1,
        direction="both",
        require_same_label=True,
    )
    expected = pd.Series(["A", "A", np.nan, np.nan, np.nan], dtype="object")
    assert_series_equal(out, expected)
