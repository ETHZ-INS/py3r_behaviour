from __future__ import annotations

import itertools
import json
import pickle
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from py3r.behaviour.script.discovery import discover_outputs, discover_params, validate_names
from py3r.behaviour.script.results import ScriptResults

_WRAPPER = Path(__file__).parent / "_subprocess_wrapper.py"
_COMBINATORIAL_WARN = 100

# Sentinel returned by discover_params for required (no-default) params.
_REQUIRED = None


def inspect(script_path: str | Path) -> None:
    """
    Print a summary of the :func:`Param` and :func:`Output` calls in a script.

    Useful for quickly checking what parameters a script exposes, which have
    defaults, and what outputs it produces — without actually running it.

    Parameters
    ----------
    script_path : str | Path
        Path to a Python script containing :func:`Param` and/or :func:`Output` calls.

    Examples
    --------
    ```python
    from py3r.behaviour.script import inspect
    inspect("pipeline.py")
    # Script: pipeline.py
    #
    # Parameters:
    #   data_path            default='/my/data'  type=str
    #   smooth_window        default=5           type=int
    #   threshold            required
    #
    # Outputs:
    #   tracking
    #   summary
    ```
    """
    script_path = Path(script_path)
    validate_names(script_path)
    params = discover_params(script_path)
    outputs = discover_outputs(script_path)

    print(f"Script: {script_path}")
    print()
    if params:
        print("Parameters:")
        for name, default in params.items():
            if default is _REQUIRED:
                print(f"  {name:<20} required")
            else:
                print(f"  {name:<20} default={default!r:<12} type={type(default).__name__}")
    else:
        print("Parameters: none")
    print()
    if outputs:
        print("Outputs:")
        for name in outputs:
            print(f"  {name}")
    else:
        print("Outputs: none")


def _build_iterations(
    params: dict[str, list],
    nominal: dict[str, Any],
    mode: Literal["independent", "grid"],
) -> list[dict]:
    if mode == "grid":
        keys = list(params.keys())
        return [
            dict(zip(keys, combo, strict=True)) for combo in itertools.product(*params.values())
        ]

    # independent: vary one param at a time, others held at nominal
    seen: set[tuple] = set()
    iterations: list[dict] = []
    for name, values in params.items():
        for v in values:
            candidate = {**nominal, name: v}
            key = tuple(sorted(candidate.items()))
            if key not in seen:
                seen.add(key)
                iterations.append(candidate)
    return iterations


def _resolve_outputs(
    requested: list[str] | None,
    script_outputs: list[str],
) -> list[str]:
    if requested is None:
        return script_outputs
    unknown = set(requested) - set(script_outputs)
    if unknown:
        raise ValueError(
            f"Outputs not found in script: {sorted(unknown)}. Script exposes: {script_outputs}."
        )
    return requested


def _stop_after_name(
    requested_outputs: list[str],
    script_outputs: list[str],
    stop_after_outputs: bool,
) -> str | None:
    if not stop_after_outputs or not requested_outputs:
        return None
    return max(requested_outputs, key=lambda n: script_outputs.index(n))


def _run_one(
    script_path: Path,
    param_values: dict,
    stop_after: str | None,
) -> tuple[dict | None, str | None]:
    """Run one iteration in a subprocess. Returns (outputs, error_message)."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        output_path = Path(f.name)

    cmd = [
        sys.executable,
        str(_WRAPPER),
        str(script_path),
        json.dumps(param_values),
        str(output_path),
    ]
    if stop_after is not None:
        cmd.append(stop_after)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if not output_path.exists() or output_path.stat().st_size == 0:
            return None, proc.stderr.strip() or "subprocess produced no output"

        data: dict = pickle.loads(output_path.read_bytes())

        if "__error__" in data:
            return None, f"{data['__type__']}: {data['__error__']}"

        return data, None

    finally:
        output_path.unlink(missing_ok=True)


def _make_results_recorder(
    results: ScriptResults,
    requested_outputs: list[str],
):
    def _record(param_values: dict, raw_outputs: dict | None, error: str | None) -> None:
        if error:
            results._add_error(param_values, error)
        else:
            filtered = {k: v for k, v in raw_outputs.items() if k in requested_outputs}
            results._add(param_values, filtered)

    return _record


def run(
    script_path: str | Path,
    params: dict[str, Any] | None = None,
    *,
    outputs: list[str] | None = None,
    stop_after_outputs: bool = False,
) -> ScriptResults:
    """
    Run a parameterised script exactly once with the given parameter values.

    Unspecified parameters use their script default. Parameters with no default
    must be provided.

    Parameters
    ----------
    script_path : str | Path
        Path to a Python script containing :func:`Param` and/or :func:`Output` calls.
    params : dict[str, scalar] | None
        Parameter values to inject, keyed by name. Unspecified params use their
        script default.
    outputs : list[str] | None
        Names of :func:`Output` values to capture. Defaults to all outputs.
    stop_after_outputs : bool
        If True, the subprocess is terminated immediately after the last requested
        output is captured.

    Returns
    -------
    ScriptResults
        Single-entry results container. Access outputs via
        ``sr[params]["output_name"]``.

    Examples
    --------
    ```python
    from py3r.behaviour.script import run

    sr = run("pipeline.py", {"data_path": "/data/session1", "threshold": 0.8})
    summary = sr[{"data_path": "/data/session1", "threshold": 0.8}]["summary"]
    ```

    Capture only an early output and stop the subprocess before downstream
    stages run:

    ```python
    sr = run("pipeline.py", {"threshold": 0.8}, outputs=["tracking"], stop_after_outputs=True)
    ```
    """
    script_path = Path(script_path)
    params = params or {}
    validate_names(script_path)

    script_params = discover_params(script_path)
    script_outputs = discover_outputs(script_path)

    unknown = set(params) - set(script_params)
    if unknown:
        raise ValueError(
            f"Parameters not found in script: {sorted(unknown)}. "
            f"Script exposes: {sorted(script_params)}."
        )

    missing_required = [
        name
        for name, default in script_params.items()
        if default is _REQUIRED and name not in params
    ]
    if missing_required:
        raise ValueError(
            f"Required parameters not provided: {missing_required}. "
            "These parameters have no default and must be supplied via params=."
        )

    requested_outputs = _resolve_outputs(outputs, script_outputs)
    stop_after = _stop_after_name(requested_outputs, script_outputs, stop_after_outputs)

    raw_outputs, error = _run_one(script_path, params, stop_after)

    results = ScriptResults(param_names=list(params.keys()), output_names=requested_outputs)
    record = _make_results_recorder(results, requested_outputs)
    record(params, raw_outputs, error)

    if error:
        warnings.warn(f"Script run failed: {error}", UserWarning, stacklevel=2)

    return results


def sensitivity(
    script_path: str | Path,
    params: dict[str, list],
    *,
    nominal: dict[str, Any] | None = None,
    outputs: list[str] | None = None,
    stop_after_outputs: bool = False,
    mode: Literal["independent", "grid"] = "independent",
) -> ScriptResults:
    """
    Run a parameterised script repeatedly, sweeping flagged :func:`Param` values.

    Each iteration runs in a subprocess. :func:`Output` values are collected and
    returned in a :class:`ScriptResults` container. Failed iterations are recorded
    but do not stop the run.

    For each swept parameter, its script default is automatically included in the
    sweep (deduplicated silently) and used as the nominal value for independent-mode
    sweeps. Use ``nominal`` to override this.

    Parameters
    ----------
    script_path : str | Path
        Path to a Python script containing :func:`Param` and :func:`Output` calls.
    params : dict[str, list]
        Mapping of parameter name to list of values to sweep.
    nominal : dict[str, scalar] | None
        Nominal (baseline) values for independent-mode sweeps. Overrides script
        defaults. Required for any swept parameter that has no script default.
    outputs : list[str] | None
        Names of :func:`Output` values to capture. Defaults to all outputs.
    stop_after_outputs : bool
        If True, each subprocess is terminated after the last requested output
        is captured. Useful when outputs appear early in a long pipeline.
    mode : "independent" | "grid"
        ``"independent"`` varies one parameter at a time, holding the others at
        their nominal value. ``"grid"`` tests every combination.

    Returns
    -------
    ScriptResults
        Results container keyed by parameter combination. Access individual
        outputs via ``sr[{"param": value}]["output_name"]``, or flatten
        scalar outputs with ``sr.to_dataframe()``.

    Examples
    --------
    Independent sweep (default mode) — one parameter varies at a time:

    ```python
    from py3r.behaviour.script import sensitivity

    sr = sensitivity("pipeline.py", {"smooth_window": [3, 5, 7], "threshold": [0.5, 0.8]})
    df = sr.to_dataframe()
    ```

    Grid sweep — every combination of parameter values:

    ```python
    sr = sensitivity(
        "pipeline.py",
        {"smooth_window": [3, 5, 7], "threshold": [0.5, 0.8]},
        mode="grid",
    )
    ```

    Required parameter not being swept must be supplied via ``nominal``:

    ```python
    sr = sensitivity(
        "pipeline.py",
        {"smooth_window": [3, 5, 7]},
        nominal={"threshold": 0.8},
    )
    ```

    Early termination — stop each subprocess after the first output is
    captured, skipping downstream pipeline stages:

    ```python
    sr = sensitivity(
        "pipeline.py",
        {"smooth_window": [3, 5, 7]},
        outputs=["tracking"],
        stop_after_outputs=True,
    )
    ```
    """
    script_path = Path(script_path)
    nominal = nominal or {}
    validate_names(script_path)

    script_params = discover_params(script_path)
    script_outputs = discover_outputs(script_path)

    unknown_params = set(params) - set(script_params)
    if unknown_params:
        raise ValueError(
            f"Parameters not found in script: {sorted(unknown_params)}. "
            f"Script exposes: {sorted(script_params)}."
        )

    unknown_nominal = set(nominal) - set(script_params)
    if unknown_nominal:
        raise ValueError(
            f"nominal keys not found in script: {sorted(unknown_nominal)}. "
            f"Script exposes: {sorted(script_params)}."
        )

    # Build resolved nominal and sweep lists for swept params.
    resolved_nominal: dict[str, Any] = {}
    resolved_params: dict[str, list] = {}

    for name, values in params.items():
        script_default = script_params[name]

        # Nominal: explicit override > script default > error.
        if name in nominal:
            nom_value = nominal[name]
        elif script_default is not _REQUIRED:
            nom_value = script_default
        else:
            raise ValueError(
                f"Parameter {name!r} has no script default and no nominal value. "
                "Provide a nominal value via nominal={name!r: <value>}."
            )
        resolved_nominal[name] = nom_value

        # Append nominal to sweep list if not already present (deduplicate).
        deduped = list(dict.fromkeys(values))  # preserve order, remove dups within list
        if nom_value not in deduped:
            deduped.append(nom_value)
        resolved_params[name] = deduped

    # Non-swept params: validate required ones are covered, build injection base.
    non_swept_injected: dict[str, Any] = {}
    for name, script_default in script_params.items():
        if name in params:
            continue  # swept — handled above
        if name in nominal:
            non_swept_injected[name] = nominal[name]
        elif script_default is _REQUIRED:
            raise ValueError(
                f"Required parameter {name!r} is not being swept and has no value in nominal. "
                "Provide a value via nominal=."
            )
        # else: has a script default — subprocess handles it, no injection needed

    # Every iteration gets non-swept injected values merged in.
    full_nominal = {**resolved_nominal, **non_swept_injected}

    requested_outputs = _resolve_outputs(outputs, script_outputs)
    stop_after = _stop_after_name(requested_outputs, script_outputs, stop_after_outputs)

    iterations = _build_iterations(resolved_params, full_nominal, mode)
    # Merge non-swept injections into every iteration dict.
    if non_swept_injected:
        iterations = [{**non_swept_injected, **it} for it in iterations]
    n = len(iterations)

    if n > _COMBINATORIAL_WARN:
        warnings.warn(
            f"sensitivity() will execute {n} iterations. "
            "Use mode='independent' or reduce value lists to lower this.",
            UserWarning,
            stacklevel=2,
        )

    print("Timing first iteration...", end=" ", flush=True)
    t0 = time.monotonic()
    first_outputs, first_error = _run_one(script_path, iterations[0], stop_after)
    first_elapsed = time.monotonic() - t0
    print(f"{first_elapsed:.1f}s  →  estimated total: {first_elapsed * n:.0f}s for {n} iterations")

    results = ScriptResults(param_names=list(params.keys()), output_names=requested_outputs)
    record = _make_results_recorder(results, requested_outputs)
    record(iterations[0], first_outputs, first_error)

    with tqdm(total=n, initial=1, desc="sensitivity") as bar:
        for param_values in iterations[1:]:
            raw_outputs, error = _run_one(script_path, param_values, stop_after)
            record(param_values, raw_outputs, error)
            bar.set_postfix(params=str(param_values), error=bool(error))
            bar.update()

    n_err = len(results.errors)
    if n_err:
        warnings.warn(
            f"{n_err} of {n} iterations failed. Check results.errors for details.",
            UserWarning,
            stacklevel=2,
        )

    return results
