from __future__ import annotations

import ast
from pathlib import Path


def _name_kwarg(call: ast.Call) -> str | None:
    for kw in call.keywords:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return None


def _is_call_to(node: ast.expr, fn_name: str) -> bool:
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == fn_name
    )


def validate_names(script_path: str | Path) -> None:
    """
    Raise ``ValueError`` if any name collision exists in a script's Param/Output calls.

    Checks for duplicate param names, duplicate output names, and any name
    appearing in both params and outputs.
    """
    params = discover_params(script_path)
    outputs = discover_outputs(script_path)

    # discover_params already raises on duplicate param names (dict overwrite
    # would silently drop them, so we detect via a separate pass).
    tree = ast.parse(Path(script_path).read_text())
    param_names_in_order: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not _is_call_to(node.value, "Param"):
            continue
        name = _name_kwarg(node.value)
        if name is None:
            continue
        param_names_in_order.append(name)

    seen_params: set[str] = set()
    for name in param_names_in_order:
        if name in seen_params:
            raise ValueError(
                f"Duplicate Param name {name!r}. Each parameter must have a unique name. "
                "If you need the value in multiple places, reuse the variable, not Param()."
            )
        seen_params.add(name)

    output_names_in_order: list[str] = []
    for node in ast.walk(tree):
        call_node: ast.Call | None = None
        if isinstance(node, ast.Assign) and _is_call_to(node.value, "Output"):
            call_node = node.value
        elif isinstance(node, ast.Expr) and _is_call_to(node.value, "Output"):
            call_node = node.value
        if call_node is None:
            continue
        name = _name_kwarg(call_node)
        if name is None:
            continue
        output_names_in_order.append(name)

    seen_outputs: set[str] = set()
    for name in output_names_in_order:
        if name in seen_outputs:
            raise ValueError(
                f"Duplicate Output name {name!r}. Each output must have a unique name."
            )
        seen_outputs.add(name)

    collisions = set(params) & set(outputs)
    if collisions:
        raise ValueError(
            f"Name collision between Param and Output: {sorted(collisions)}. "
            "Params and outputs must have distinct names."
        )


def discover_params(script_path: str | Path) -> dict[str, object]:
    """
    Parse a script for ``Param()`` calls without executing it.

    Returns a ``{name: default}`` dict. Raises ``ValueError`` if any
    ``Param`` call is missing the ``name`` keyword.
    """
    tree = ast.parse(Path(script_path).read_text())
    params: dict[str, object] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not _is_call_to(node.value, "Param"):
            continue
        call = node.value

        name = _name_kwarg(call)
        if name is None:
            raise ValueError(
                f"Param() at line {node.lineno} is missing the 'name' keyword. "
                "Add name=... explicitly so the runner can identify it."
            )

        default = None
        if call.args and isinstance(call.args[0], ast.Constant):
            default = call.args[0].value

        params[name] = default

    return params


def discover_outputs(script_path: str | Path) -> list[str]:
    """
    Parse a script for ``Output()`` calls without executing it.

    Returns a list of output names. Raises ``ValueError`` if any
    ``Output`` call is missing the ``name`` keyword.
    """
    tree = ast.parse(Path(script_path).read_text())
    outputs: list[str] = []

    for node in ast.walk(tree):
        call_node: ast.Call | None = None
        if isinstance(node, ast.Assign) and _is_call_to(node.value, "Output"):
            call_node = node.value
        elif isinstance(node, ast.Expr) and _is_call_to(node.value, "Output"):
            call_node = node.value

        if call_node is None:
            continue

        name = _name_kwarg(call_node)
        if name is None:
            raise ValueError(
                f"Output() at line {node.lineno} is missing the 'name' keyword. "
                "Add name=... explicitly so the runner can capture it."
            )

        outputs.append(name)

    return outputs
