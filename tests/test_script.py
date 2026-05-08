"""Tests for py3r.behaviour.script — Param, Output, inspect, run."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from py3r.behaviour.script import Output, Param, ScriptResults, inspect, run
from py3r.behaviour.script.discovery import discover_outputs, discover_params
from py3r.behaviour.script.runner import _build_iterations

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_script(tmp_path: Path, source: str) -> Path:
    p = tmp_path / "pipeline.py"
    p.write_text(textwrap.dedent(source))
    return p


# ---------------------------------------------------------------------------
# Param / Output transparency
# ---------------------------------------------------------------------------


def test_param_returns_default_int():
    assert Param(5, name="x") == 5
    assert type(Param(5, name="x")) is int


def test_param_returns_default_float():
    assert Param(0.5, name="x") == 0.5
    assert type(Param(0.5, name="x")) is float


def test_param_returns_default_str():
    assert Param("hello", name="x") == "hello"


def test_param_returns_default_bool():
    assert Param(True, name="x") is True


def test_param_rejects_non_scalar():
    with pytest.raises(TypeError, match="scalar"):
        Param([1, 2, 3], name="x")


def test_param_rejects_dataframe():
    with pytest.raises(TypeError, match="scalar"):
        Param(pd.DataFrame(), name="x")


def test_output_returns_value_unchanged():
    df = pd.DataFrame({"a": [1]})
    assert Output(df, name="result") is df


def test_output_accepts_any_type():
    assert Output(42, name="x") == 42
    assert Output("hello", name="x") == "hello"
    assert Output([1, 2], name="x") == [1, 2]


# ---------------------------------------------------------------------------
# AST discovery
# ---------------------------------------------------------------------------


def test_discover_params_basic(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param
        window = Param(5, name="window")
        cat = 7
        dog = [1,2,3]
        fps = Param(30, name="fps")
        mouse = cat + dog
    """,
    )
    params = discover_params(p)
    assert params == {"window": 5, "fps": 30}


def test_discover_params_missing_name_raises(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param
        window = Param(5)
    """,
    )
    with pytest.raises(ValueError, match="missing the 'name' keyword"):
        discover_params(p)


def test_discover_outputs_basic(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Output
        Output(42, name="result")
    """,
    )
    assert discover_outputs(p) == ["result"]


def test_discover_outputs_missing_name_raises(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Output
        Output(42)
    """,
    )
    with pytest.raises(ValueError, match="missing the 'name' keyword"):
        discover_outputs(p)


def test_discover_outputs_preserves_order(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Output
        Output(1, name="first")
        Output(2, name="second")
        Output(3, name="third")
    """,
    )
    assert discover_outputs(p) == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# _build_iterations
# ---------------------------------------------------------------------------


def test_build_iterations_independent():
    params = {"a": [1, 2], "b": [10, 20]}
    iters = _build_iterations(params, "independent")
    # nominal is a=1, b=10; vary a then b
    assert {"a": 1, "b": 10} in iters
    assert {"a": 2, "b": 10} in iters
    assert {"a": 1, "b": 20} in iters
    # full grid combination should NOT be present
    assert {"a": 2, "b": 20} not in iters


def test_build_iterations_independent_no_duplicates():
    params = {"a": [1, 1, 2]}
    iters = _build_iterations(params, "independent")
    keys = [tuple(sorted(d.items())) for d in iters]
    assert len(keys) == len(set(keys))


def test_build_iterations_grid():
    params = {"a": [1, 2], "b": [10, 20]}
    iters = _build_iterations(params, "grid")
    assert len(iters) == 4
    assert {"a": 2, "b": 20} in iters


# ---------------------------------------------------------------------------
# run() — end-to-end
# ---------------------------------------------------------------------------


def test_run_basic(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        window = Param(5, name="window")
        result = window * 10
        Output(result, name="product")
    """,
    )
    sr = run(p, params={"window": [3, 5, 7]})
    assert isinstance(sr, ScriptResults)
    assert sr[{"window": 3}]["product"] == 30
    assert sr[{"window": 5}]["product"] == 50
    assert sr[{"window": 7}]["product"] == 70


def test_run_to_dataframe(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(1, name="x")
        Output(x * 2, name="doubled")
    """,
    )
    sr = run(p, params={"x": [1, 2, 3]})
    df = sr.to_dataframe()
    assert list(df.columns) == ["x", "doubled"]
    assert sorted(df["doubled"].tolist()) == [2, 4, 6]


def test_run_outputs_filter(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(1, name="x")
        Output(x, name="early")
        Output(x * 100, name="late")
    """,
    )
    sr = run(p, params={"x": [2]}, outputs=["early"])
    result = sr[{"x": 2}]
    assert "early" in result
    assert "late" not in result


def test_run_stop_after_outputs(tmp_path):
    # The script has a deliberate error after the output — with stop_after_outputs=True
    # it should succeed; without it, the error would cause a failure.
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(1, name="x")
        Output(x, name="result")
        raise RuntimeError("should not reach here")
    """,
    )
    sr = run(p, params={"x": [5]}, outputs=["result"], stop_after_outputs=True)
    assert sr[{"x": 5}]["result"] == 5
    assert len(sr.errors) == 0


def test_run_stop_after_outputs_false_captures_error(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(1, name="x")
        Output(x, name="result")
        raise RuntimeError("boom")
    """,
    )
    sr = run(p, params={"x": [5]}, stop_after_outputs=False)
    assert len(sr.errors) == 1


def test_run_unknown_param_raises(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param
        x = Param(1, name="x")
    """,
    )
    with pytest.raises(ValueError, match="not found in script"):
        run(p, params={"y": [1, 2]})


def test_run_unknown_output_raises(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(1, name="x")
        Output(x, name="result")
    """,
    )
    with pytest.raises(ValueError, match="not found in script"):
        run(p, params={"x": [1]}, outputs=["nonexistent"])


def test_run_error_iteration_recorded(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(1, name="x")
        if x == 2:
            raise ValueError("bad value")
        Output(x, name="result")
    """,
    )
    sr = run(p, params={"x": [1, 2, 3]})
    assert len(sr.errors) == 1
    assert sr[{"x": 1}]["result"] == 1
    assert sr[{"x": 3}]["result"] == 3


def test_run_grid_mode(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        a = Param(1, name="a")
        b = Param(10, name="b")
        Output(a + b, name="sum")
    """,
    )
    sr = run(p, params={"a": [1, 2], "b": [10, 20]}, mode="grid")
    assert sr[{"a": 1, "b": 10}]["sum"] == 11
    assert sr[{"a": 2, "b": 20}]["sum"] == 22


# ---------------------------------------------------------------------------
# inspect()
# ---------------------------------------------------------------------------


def test_inspect_runs_without_error(tmp_path, capsys):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        window = Param(5, name="window")
        Output(window, name="result")
    """,
    )
    inspect(p)
    out = capsys.readouterr().out
    assert "window" in out
    assert "result" in out
    assert "5" in out


def test_inspect_shows_type(tmp_path, capsys):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param
        x = Param(3.14, name="x")
    """,
    )
    inspect(p)
    out = capsys.readouterr().out
    assert "float" in out


# ---------------------------------------------------------------------------
# ScriptResults
# ---------------------------------------------------------------------------


def test_scriptresults_missing_key_error():
    sr = ScriptResults(param_names=["x"], output_names=["y"])
    sr._add({"x": 1}, {"y": 42})
    with pytest.raises(KeyError, match="No result"):
        sr[{"x": 99}]


def test_scriptresults_to_dataframe_non_scalar_raises():
    sr = ScriptResults(param_names=["x"], output_names=["y"])
    sr._add({"x": 1}, {"y": pd.DataFrame()})
    with pytest.raises(TypeError, match="not scalar"):
        sr.to_dataframe()


def test_scriptresults_repr():
    sr = ScriptResults(param_names=["x"], output_names=["y"])
    sr._add({"x": 1}, {"y": 42})
    r = repr(sr)
    assert "ScriptResults" in r
    assert "x" in r
    assert "y" in r
