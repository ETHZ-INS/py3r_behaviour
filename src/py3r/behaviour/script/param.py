from __future__ import annotations

_ALLOWED_TYPES = (bool, int, float, str)  # bool before int: bool subclasses int

# Sentinel for a Param with no default (required param).
_MISSING = object()


def Param(
    default: bool | int | float | str | None = _MISSING,
    *,
    name: str,
) -> bool | int | float | str:
    """
    Mark a script variable as a runner parameter.

    In normal script execution, returns ``default``. If no default is given,
    raises ``RuntimeError`` to signal that the script must be run via
    :func:`run` or :func:`sensitivity` with a value supplied for this parameter.

    When run via :func:`run` or :func:`sensitivity`, returns the injected value.

    Args:
        default: Value used during normal execution. Omit to mark the parameter
            as required (no default — must always be supplied by the runner).
        name: Parameter name, matched against keys passed to ``run`` or
            ``sensitivity``.

    Examples
    --------
    ```pycon
    >>> window = Param(5, name="window")
    >>> window
    5
    >>> type(window)
    <class 'int'>

    ```
    """
    if default is _MISSING:
        raise RuntimeError(
            f"Parameter {name!r} has no default and was not provided by the runner. "
            "Run this script via run() or sensitivity(), supplying a value for this parameter."
        )
    if not isinstance(default, _ALLOWED_TYPES):
        raise TypeError(
            f"Param default must be a scalar (bool, int, float, or str); "
            f"got {type(default).__name__}. "
            "For complex inputs, derive them inside your script from parameter values."
        )
    return default


def Output(value: object, *, name: str) -> object:
    """
    Mark a script value as a runner output to capture.

    In normal script execution, returns ``value`` unchanged. When run via
    :func:`run` or :func:`sensitivity`, captures the value under ``name``
    in the results.

    Args:
        value: The value to capture. Any type is accepted.
        name: Output name, used to key results in ``ScriptResults``.

    Examples
    --------
    ```pycon
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 2]})
    >>> result = Output(df, name="summary")
    >>> result is df
    True

    ```
    """
    return value
