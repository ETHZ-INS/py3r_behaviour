"""
Internal script executed in each script runner subprocess.

Called as:
    python _subprocess_wrapper.py <script_path> <param_json> <output_pkl> [stop_after_output]
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

_MISSING = object()


class _StopAfterOutput(Exception):
    pass


def _make_param(param_values: dict):
    def Param(default=_MISSING, *, name: str):
        if name in param_values:
            v = param_values[name]
            # Cast to match default type when a default is available.
            return type(default)(v) if default is not _MISSING else v
        if default is _MISSING:
            raise ValueError(
                f"Required parameter {name!r} was not provided. Pass it via run() or sensitivity()."
            )
        return default

    return Param


def _make_output(store: dict, stop_after: str | None):
    def Output(value, *, name: str):
        store[name] = value
        if stop_after is not None and name == stop_after:
            raise _StopAfterOutput(name)
        return value

    return Output


def main() -> None:
    script_path = sys.argv[1]
    param_values: dict = json.loads(sys.argv[2])
    output_path = Path(sys.argv[3])
    stop_after: str | None = sys.argv[4] if len(sys.argv) > 4 else None

    outputs: dict = {}
    capturing_param = _make_param(param_values)
    capturing_output = _make_output(outputs, stop_after)

    # Patch the installed module so that `from py3r.behaviour.script import Param, Output`
    # in the user's script gets our capturing versions rather than the transparent defaults.
    import py3r.behaviour.script as _script

    _script.Param = capturing_param  # type: ignore[attr-defined]
    _script.Output = capturing_output  # type: ignore[attr-defined]

    namespace = {
        "__name__": "__main__",
        "__file__": script_path,
        "Param": capturing_param,
        "Output": capturing_output,
    }

    source = Path(script_path).read_text()
    try:
        exec(compile(source, script_path, "exec"), namespace)  # noqa: S102
    except _StopAfterOutput:
        pass  # intentional early exit — outputs already captured
    except Exception as exc:
        output_path.write_bytes(
            pickle.dumps({"__error__": str(exc), "__type__": type(exc).__name__})
        )
        sys.exit(1)

    output_path.write_bytes(pickle.dumps(outputs))


if __name__ == "__main__":
    main()
