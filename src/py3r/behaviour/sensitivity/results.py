from __future__ import annotations

import sys
from typing import Any

import pandas as pd

_SIZE_WARN_BYTES = 1 * 1024**3  # 1 GB


class SensitivityResults:
    """
    Container for sensitivity analysis results.

    Keyed by parameter combination, holds captured :func:`Output` values per iteration.
    Access a result with a param dict::

        sr[{"window": 5, "fps": 30}]["summary"]

    Or flatten scalar outputs to a DataFrame::

        sr.to_dataframe()
    """

    def __init__(self, param_names: list[str], output_names: list[str]) -> None:
        self._param_names = param_names
        self._output_names = output_names
        self._results: dict[tuple, dict[str, Any]] = {}
        self._errors: dict[tuple, str] = {}
        self._size_warned = False

    def _key(self, param_dict: dict) -> tuple:
        return tuple(param_dict[k] for k in self._param_names)

    def _add(self, param_dict: dict, outputs: dict[str, Any]) -> None:
        self._results[self._key(param_dict)] = outputs
        self._check_size()

    def _add_error(self, param_dict: dict, message: str) -> None:
        self._errors[self._key(param_dict)] = message

    def __getitem__(self, param_dict: dict) -> dict[str, Any]:
        try:
            return self._results[self._key(param_dict)]
        except KeyError:
            raise KeyError(f"No result for parameters: {param_dict}") from None

    def _check_size(self) -> None:
        if self._size_warned:
            return
        total = sum(
            sys.getsizeof(v) for outputs in self._results.values() for v in outputs.values()
        )
        if total > _SIZE_WARN_BYTES:
            import warnings

            warnings.warn(
                f"SensitivityResults is holding approximately {total / 1e9:.1f} GB in memory. "
                "Consider saving outputs to disk instead.",
                ResourceWarning,
                stacklevel=3,
            )
            self._size_warned = True

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten scalar outputs into a DataFrame (one row per iteration).

        Raises ``TypeError`` for any non-scalar output — access those directly
        via indexing instead.
        """
        records = []
        for key, outputs in self._results.items():
            row = dict(zip(self._param_names, key, strict=True))
            for name, value in outputs.items():
                if not isinstance(value, (bool, int, float, str)):
                    raise TypeError(
                        f"Output '{name}' is {type(value).__name__}, not scalar. "
                        "Access it directly: results[param_dict]['name']."
                    )
                row[name] = value
            records.append(row)
        return pd.DataFrame(records)

    @property
    def errors(self) -> dict[dict, str]:
        """Failed iterations as ``{param_dict: error_message}``."""
        return {dict(zip(self._param_names, k, strict=True)): v for k, v in self._errors.items()}

    def __repr__(self) -> str:
        n_ok = len(self._results)
        n_err = len(self._errors)
        lines = [
            "SensitivityResults",
            f"  parameters : {self._param_names}",
            f"  outputs    : {self._output_names}",
            f"  iterations : {n_ok} completed, {n_err} failed",
        ]
        if self._results:
            first_outputs = next(iter(self._results.values()))
            for out_name, val in first_outputs.items():
                lines.append(f"  {out_name!r:<20}: {type(val).__name__}")
        return "\n".join(lines)
