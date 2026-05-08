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
from typing import Literal

from tqdm import tqdm

from py3r.behaviour.script.discovery import discover_outputs, discover_params
from py3r.behaviour.script.results import ScriptResults

_WRAPPER = Path(__file__).parent / "_subprocess_wrapper.py"
_COMBINATORIAL_WARN = 100


def inspect(script_path: str | Path) -> None:
    """
    Print a summary of the :func:`Param` and :func:`Output` calls in a script.

    Parameters
    ----------
    script_path : str | Path
        Path to a Python script containing :func:`Param` and/or :func:`Output` calls.
    """
    script_path = Path(script_path)
    params = discover_params(script_path)
    outputs = discover_outputs(script_path)

    print(f"Script: {script_path}")
    print()
    if params:
        print("Parameters:")
        for name, default in params.items():
            type_name = type(default).__name__ if default is not None else "unknown"
            print(f"  {name:<20} default={default!r:<12} type={type_name}")
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
    mode: Literal["independent", "grid"],
) -> list[dict]:
    if mode == "grid":
        keys = list(params.keys())
        return [
            dict(zip(keys, combo, strict=True)) for combo in itertools.product(*params.values())
        ]

    # independent: vary one param at a time, others held at their first (nominal) value
    nominal = {k: vs[0] for k, vs in params.items()}
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


def run(
    script_path: str | Path,
    params: dict[str, list],
    *,
    outputs: list[str] | None = None,
    stop_after_outputs: bool = False,
    mode: Literal["independent", "grid"] = "independent",
) -> ScriptResults:
    """
    Run a parameterised script repeatedly, sweeping flagged :func:`Param` values.

    Each iteration runs in a subprocess. :func:`Output` values are collected and
    returned in a :class:`ScriptResults` container. Failed iterations are recorded
    but do not stop the run.

    Parameters
    ----------
    script_path : str | Path
        Path to a Python script containing :func:`Param` and :func:`Output` calls.
    params : dict[str, list]
        Mapping of parameter name to list of values to sweep.
        Names must match the ``name`` argument of :func:`Param` calls in the script.
    outputs : list[str] | None
        Names of :func:`Output` values to capture. Defaults to all outputs in the script.
    stop_after_outputs : bool
        If True, each subprocess is terminated immediately after the last requested
        output is captured. Useful when outputs appear early in a long pipeline.
        Defaults to False.
    mode : "independent" | "grid"
        ``"independent"`` varies one parameter at a time, holding the others at
        their first (nominal) value — the default for most sensitivity analyses.
        ``"grid"`` tests every combination.

    Returns
    -------
    ScriptResults
    """
    script_path = Path(script_path)

    script_params = discover_params(script_path)
    script_outputs = discover_outputs(script_path)

    unknown_params = set(params) - set(script_params)
    if unknown_params:
        raise ValueError(
            f"Parameters not found in script: {sorted(unknown_params)}. "
            f"Script exposes: {sorted(script_params)}."
        )

    if outputs is not None:
        unknown_outputs = set(outputs) - set(script_outputs)
        if unknown_outputs:
            raise ValueError(
                f"Outputs not found in script: {sorted(unknown_outputs)}. "
                f"Script exposes: {script_outputs}."
            )
        requested_outputs = outputs
    else:
        requested_outputs = script_outputs

    # The subprocess stops after the last requested output in script order.
    stop_after: str | None = None
    if stop_after_outputs and requested_outputs:
        # Find which requested output appears last in the script.
        stop_after = max(requested_outputs, key=lambda n: script_outputs.index(n))

    iterations = _build_iterations(params, mode)
    n = len(iterations)

    if n > _COMBINATORIAL_WARN:
        warnings.warn(
            f"run() will execute {n} iterations. "
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

    def _record(param_values: dict, raw_outputs: dict | None, error: str | None) -> None:
        if error:
            results._add_error(param_values, error)
        else:
            filtered = {k: v for k, v in raw_outputs.items() if k in requested_outputs}
            results._add(param_values, filtered)

    _record(iterations[0], first_outputs, first_error)

    with tqdm(total=n, initial=1, desc="running") as bar:
        for param_values in iterations[1:]:
            raw_outputs, error = _run_one(script_path, param_values, stop_after)
            _record(param_values, raw_outputs, error)
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
