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

from py3r.behaviour.sensitivity.discovery import discover_outputs, discover_params
from py3r.behaviour.sensitivity.results import SensitivityResults

_WRAPPER = Path(__file__).parent / "_subprocess_wrapper.py"
_COMBINATORIAL_WARN = 100


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


def _run_one(script_path: Path, param_values: dict) -> tuple[dict | None, str | None]:
    """Run one iteration in a subprocess. Returns (outputs, error_message)."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        output_path = Path(f.name)

    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(_WRAPPER),
                str(script_path),
                json.dumps(param_values),
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )

        if not output_path.exists() or output_path.stat().st_size == 0:
            return None, proc.stderr.strip() or "subprocess produced no output"

        data: dict = pickle.loads(output_path.read_bytes())

        if "__error__" in data:
            return None, f"{data['__type__']}: {data['__error__']}"

        return data, None

    finally:
        output_path.unlink(missing_ok=True)


def run_sensitivity(
    script_path: str | Path,
    params: dict[str, list],
    *,
    mode: Literal["independent", "grid"] = "independent",
) -> SensitivityResults:
    """
    Run a parameterised script repeatedly, sweeping flagged :func:`Param` values.

    Each iteration runs in a subprocess. :func:`Output` values are collected and
    returned in a :class:`SensitivityResults` container. Failed iterations are
    recorded but do not stop the run.

    Parameters
    ----------
    script_path : str | Path
        Path to a Python script containing :func:`Param` and :func:`Output` calls.
    params : dict[str, list]
        Mapping of parameter name to list of values to sweep.
        Names must match the ``name`` argument of :func:`Param` calls in the script.
    mode : "independent" | "grid"
        ``"independent"`` varies one parameter at a time, holding the others at
        their first (nominal) value — the default for most sensitivity analyses.
        ``"grid"`` tests every combination.

    Returns
    -------
    SensitivityResults
    """
    script_path = Path(script_path)

    script_params = discover_params(script_path)
    output_names = discover_outputs(script_path)

    unknown = set(params) - set(script_params)
    if unknown:
        raise ValueError(
            f"Parameters not found in script: {sorted(unknown)}. "
            f"Script exposes: {sorted(script_params)}."
        )

    iterations = _build_iterations(params, mode)
    n = len(iterations)

    if n > _COMBINATORIAL_WARN:
        warnings.warn(
            f"run_sensitivity will execute {n} iterations. "
            "Use mode='independent' or reduce value lists to lower this.",
            UserWarning,
            stacklevel=2,
        )

    # Time the first iteration to give an ETA before committing to the rest
    print("Timing first iteration...", end=" ", flush=True)
    t0 = time.monotonic()
    first_outputs, first_error = _run_one(script_path, iterations[0])
    first_elapsed = time.monotonic() - t0
    eta = first_elapsed * n
    print(f"{first_elapsed:.1f}s  →  estimated total: {eta:.0f}s for {n} iterations")

    results = SensitivityResults(param_names=list(params.keys()), output_names=output_names)

    def _record(param_values: dict, outputs: dict | None, error: str | None) -> None:
        if error:
            results._add_error(param_values, error)
        else:
            results._add(param_values, outputs)

    _record(iterations[0], first_outputs, first_error)

    with tqdm(total=n, initial=1, desc="sensitivity") as bar:
        for param_values in iterations[1:]:
            outputs, error = _run_one(script_path, param_values)
            _record(param_values, outputs, error)
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
