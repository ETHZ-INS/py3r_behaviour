"""
Internal script executed in each sensitivity analysis subprocess.

Called as:
    python _subprocess_wrapper.py <script_path> <param_json> <output_pkl>
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path


def _make_param(param_values: dict):
    def Param(default, *, name: str):
        if name in param_values:
            return type(default)(param_values[name])
        return default

    return Param


def _make_output(store: dict):
    def Output(value, *, name: str):
        store[name] = value
        return value

    return Output


def main() -> None:
    script_path = sys.argv[1]
    param_values: dict = json.loads(sys.argv[2])
    output_path = Path(sys.argv[3])

    outputs: dict = {}
    capturing_param = _make_param(param_values)
    capturing_output = _make_output(outputs)

    # Patch the installed module so that `from py3r.behaviour.sensitivity import Param, Output`
    # in the user's script gets our capturing versions rather than the transparent defaults.
    import py3r.behaviour.sensitivity as _sens

    _sens.Param = capturing_param  # type: ignore[attr-defined]
    _sens.Output = capturing_output  # type: ignore[attr-defined]

    namespace = {
        "__name__": "__main__",
        "__file__": script_path,
        "Param": capturing_param,
        "Output": capturing_output,
    }

    source = Path(script_path).read_text()
    try:
        exec(compile(source, script_path, "exec"), namespace)  # noqa: S102
    except Exception as exc:
        output_path.write_bytes(
            pickle.dumps({"__error__": str(exc), "__type__": type(exc).__name__})
        )
        sys.exit(1)

    output_path.write_bytes(pickle.dumps(outputs))


if __name__ == "__main__":
    main()
