"""Tests for py3r.behaviour.script — Param, Output, inspect, run, sensitivity."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from py3r.behaviour.script import Output, Param, ScriptResults, inspect, run, sensitivity
from py3r.behaviour.script.discovery import discover_outputs, discover_params, validate_names
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


def test_param_no_default_raises():
    with pytest.raises(RuntimeError, match="no default"):
        Param(name="x")


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
        fps = Param(30, name="fps")
    """,
    )
    assert discover_params(p) == {"window": 5, "fps": 30}


def test_discover_params_required(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param
        path = Param(name="path")
    """,
    )
    assert discover_params(p) == {"path": None}


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
# Name collision validation
# ---------------------------------------------------------------------------


def test_validate_names_duplicate_param_raises(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param
        x = Param(1, name="x")
        y = Param(2, name="x")
    """,
    )
    with pytest.raises(ValueError, match="Duplicate Param name"):
        validate_names(p)


def test_validate_names_duplicate_output_raises(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Output
        Output(1, name="result")
        Output(2, name="result")
    """,
    )
    with pytest.raises(ValueError, match="Duplicate Output name"):
        validate_names(p)


def test_validate_names_param_output_collision_raises(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(1, name="x")
        Output(x, name="x")
    """,
    )
    with pytest.raises(ValueError, match="collision"):
        validate_names(p)


def test_validate_names_clean_script_passes(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(1, name="x")
        Output(x * 2, name="result")
    """,
    )
    validate_names(p)  # should not raise


def test_run_raises_on_name_collision(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(1, name="x")
        Output(x, name="x")
    """,
    )
    with pytest.raises(ValueError, match="collision"):
        run(p, {"x": 2})


def test_inspect_raises_on_name_collision(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(1, name="x")
        Output(x, name="x")
    """,
    )
    with pytest.raises(ValueError, match="collision"):
        inspect(p)


# ---------------------------------------------------------------------------
# _build_iterations
# ---------------------------------------------------------------------------


def test_build_iterations_independent():
    params = {"a": [1, 2, 5], "b": [10, 20, 50]}
    nominal = {"a": 5, "b": 50}
    iters = _build_iterations(params, nominal, "independent")
    # nominal a=5, b=50; vary a then b, others held at nominal
    assert {"a": 1, "b": 50} in iters
    assert {"a": 2, "b": 50} in iters
    assert {"a": 5, "b": 10} in iters
    assert {"a": 5, "b": 20} in iters
    # full grid combination should NOT be present
    assert {"a": 1, "b": 10} not in iters


def test_build_iterations_independent_no_duplicates():
    params = {"a": [1, 1, 2]}
    nominal = {"a": 1}
    iters = _build_iterations(params, nominal, "independent")
    keys = [tuple(sorted(d.items())) for d in iters]
    assert len(keys) == len(set(keys))


def test_build_iterations_grid():
    params = {"a": [1, 2], "b": [10, 20]}
    nominal = {"a": 1, "b": 10}
    iters = _build_iterations(params, nominal, "grid")
    assert len(iters) == 4
    assert {"a": 2, "b": 20} in iters


# ---------------------------------------------------------------------------
# run() — single execution
# ---------------------------------------------------------------------------


def test_run_single_with_param(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        window = Param(5, name="window")
        Output(window * 10, name="product")
    """,
    )
    sr = run(p, {"window": 3})
    assert isinstance(sr, ScriptResults)
    assert sr[{"window": 3}]["product"] == 30


def test_run_single_uses_default(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        window = Param(5, name="window")
        Output(window, name="w")
    """,
    )
    sr = run(p)
    assert sr[{}]["w"] == 5


def test_run_required_param_provided(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        path = Param(name="path")
        Output(path, name="result")
    """,
    )
    sr = run(p, {"path": "/my/data"})
    assert sr[{"path": "/my/data"}]["result"] == "/my/data"


def test_run_required_param_missing_raises(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param
        path = Param(name="path")
    """,
    )
    with pytest.raises(ValueError, match="Required parameters not provided"):
        run(p)


def test_run_unknown_param_raises(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param
        x = Param(1, name="x")
    """,
    )
    with pytest.raises(ValueError, match="not found in script"):
        run(p, {"y": 1})


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
        run(p, outputs=["nonexistent"])


def test_run_stop_after_outputs(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(1, name="x")
        Output(x, name="result")
        raise RuntimeError("should not reach here")
    """,
    )
    sr = run(p, {"x": 5}, outputs=["result"], stop_after_outputs=True)
    assert sr[{"x": 5}]["result"] == 5
    assert len(sr.errors) == 0


def test_run_error_recorded(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(1, name="x")
        Output(x, name="result")
        raise RuntimeError("boom")
    """,
    )
    sr = run(p, {"x": 5})
    assert len(sr.errors) == 1


# ---------------------------------------------------------------------------
# sensitivity() — multi-value sweep
# ---------------------------------------------------------------------------


def test_sensitivity_basic(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        window = Param(5, name="window")
        Output(window * 10, name="product")
    """,
    )
    sr = sensitivity(p, {"window": [3, 7]})
    # default 5 is auto-appended as nominal
    assert sr[{"window": 3}]["product"] == 30
    assert sr[{"window": 5}]["product"] == 50
    assert sr[{"window": 7}]["product"] == 70


def test_sensitivity_default_appended(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(5, name="x")
        Output(x, name="result")
    """,
    )
    sr = sensitivity(p, {"x": [3, 7]})
    # default 5 should appear even though not in [3, 7]
    assert sr[{"x": 5}]["result"] == 5


def test_sensitivity_default_not_duplicated(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(5, name="x")
        Output(x, name="result")
    """,
    )
    sr = sensitivity(p, {"x": [3, 5, 7]})
    # 5 is already in list — should not run twice
    keys = [k for k in sr._results if True]
    assert len(keys) == 3


def test_sensitivity_nominal_uses_default(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        a = Param(10, name="a")
        b = Param(100, name="b")
        Output(a + b, name="sum")
    """,
    )
    sr = sensitivity(p, {"a": [1, 2]})
    # b held at its default (100) as nominal
    assert sr[{"a": 1}]["sum"] == 101
    assert sr[{"a": 2}]["sum"] == 102


def test_sensitivity_nominal_override(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        a = Param(10, name="a")
        b = Param(100, name="b")
        Output(a + b, name="sum")
    """,
    )
    sr = sensitivity(p, {"a": [1, 2]}, nominal={"b": 999})
    # b held at 999 as nominal, not its default 100
    assert sr[{"a": 1}]["sum"] == 1000
    assert sr[{"a": 2}]["sum"] == 1001


def test_sensitivity_required_param_needs_nominal(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        path = Param(name="path")
        x = Param(5, name="x")
        Output(x, name="result")
    """,
    )
    with pytest.raises(ValueError, match="no value in nominal"):
        sensitivity(p, {"x": [1, 2]}, nominal={})


def test_sensitivity_required_param_with_nominal(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        path = Param(name="path")
        x = Param(5, name="x")
        Output(x, name="result")
    """,
    )
    sr = sensitivity(p, {"x": [1, 2]}, nominal={"path": "/data"})
    assert sr[{"x": 1}]["result"] == 1


def test_sensitivity_stop_after_outputs(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(1, name="x")
        Output(x, name="result")
        raise RuntimeError("should not reach here")
    """,
    )
    sr = sensitivity(p, {"x": [2, 3]}, outputs=["result"], stop_after_outputs=True)
    assert len(sr.errors) == 0


def test_sensitivity_error_iteration_recorded(tmp_path):
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
    sr = sensitivity(p, {"x": [1, 2, 3]})
    assert len(sr.errors) == 1
    assert sr[{"x": 1}]["result"] == 1
    assert sr[{"x": 3}]["result"] == 3


def test_sensitivity_grid_mode(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        a = Param(1, name="a")
        b = Param(10, name="b")
        Output(a + b, name="sum")
    """,
    )
    sr = sensitivity(p, {"a": [1, 2], "b": [10, 20]}, mode="grid")
    assert sr[{"a": 1, "b": 10}]["sum"] == 11
    assert sr[{"a": 2, "b": 20}]["sum"] == 22


def test_sensitivity_to_dataframe(tmp_path):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param, Output
        x = Param(1, name="x")
        Output(x * 2, name="doubled")
    """,
    )
    sr = sensitivity(p, {"x": [1, 2, 3]})
    df = sr.to_dataframe()
    assert "x" in df.columns
    assert "doubled" in df.columns
    assert sorted(df["doubled"].tolist()) == [2, 4, 6]


# ---------------------------------------------------------------------------
# inspect()
# ---------------------------------------------------------------------------


def test_inspect_shows_param_and_output(tmp_path, capsys):
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
    assert "float" in capsys.readouterr().out


def test_inspect_shows_required(tmp_path, capsys):
    p = _write_script(
        tmp_path,
        """
        from py3r.behaviour.script import Param
        path = Param(name="path")
    """,
    )
    inspect(p)
    assert "required" in capsys.readouterr().out


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
