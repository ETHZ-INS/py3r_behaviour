from __future__ import annotations

_ALLOWED_TYPES = (bool, int, float, str)  # bool before int: bool subclasses int


def Param(default: bool | int | float | str, *, name: str) -> bool | int | float | str:
    """
    Mark a script variable as a sensitivity analysis parameter.

    In normal script execution, returns ``default`` unchanged. When run via
    :func:`run_sensitivity`, returns the value injected for this parameter.

    Parameters
    ----------
    default : bool | int | float | str
        Value used during normal (non-sensitivity) execution.
    name : str
        Parameter name, matched against keys passed to :func:`run_sensitivity`.

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
    if not isinstance(default, _ALLOWED_TYPES):
        raise TypeError(
            f"Param default must be a scalar (bool, int, float, or str); "
            f"got {type(default).__name__}. "
            "For complex inputs, derive them inside your script from parameter values."
        )
    return default


def Output(value: object, *, name: str) -> object:
    """
    Mark a script value as a sensitivity analysis output to capture.

    In normal script execution, returns ``value`` unchanged. When run via
    :func:`run_sensitivity`, captures the value under ``name`` in the results.

    Parameters
    ----------
    value : object
        The value to capture. Any type is accepted.
    name : str
        Output name, used to key results in :class:`SensitivityResults`.

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
