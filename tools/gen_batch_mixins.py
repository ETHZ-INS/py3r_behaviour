# AUTO-GENERATED FILE. DO NOT EDIT.
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, Tuple, List, Optional

"""
Generate grouped-aware batch mixins for collection classes.

Usage:
  python -m tools.gen_batch_mixins

Outputs:
  - src/py3r/behaviour/tracking/tracking_collection_batch_mixin.py
  - src/py3r/behaviour/features/features_collection_batch_mixin.py
  - src/py3r/behaviour/summary/summary_collection_batch_mixin.py

Each mixin provides concrete wrappers for all public instance methods of the
respective leaf class (Tracking, Features, Summary), preserving signatures and
(doc)strings, and delegating to a grouped-aware dispatcher that returns BatchResult.
"""


class MethodInfo:
    def __init__(self, name: str, params_src: str, call_args: str, doc: str):
        self.name = name
        self.params_src = params_src
        self.call_args = call_args
        self.doc = doc


def _expr_src(src: str, node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.get_source_segment(src, node)
    except Exception:
        return None


def _build_params_and_calls(src: str, f: ast.FunctionDef) -> Tuple[str, str]:
    args = f.args
    parts: List[str] = []
    call_parts: List[str] = []

    # Positional-only (py3.8+)
    posonly = getattr(args, "posonlyargs", [])
    pos = args.args[:]  # includes self first
    if pos and pos[0].arg == "self":
        pos = pos[1:]

    # Defaults align to the last N of posonly+pos
    pos_all = posonly + pos
    defaults = args.defaults or []
    num_defaults = len(defaults)
    num_no_default = len(pos_all) - num_defaults

    # Positional-only
    for i, a in enumerate(posonly):
        name = a.arg
        ann = _expr_src(src, a.annotation)
        if i >= num_no_default:
            dsrc = _expr_src(src, defaults[i - num_no_default]) or "None"
            if ann:
                parts.append(f"{name}: {ann}={dsrc}")
            else:
                parts.append(f"{name}={dsrc}")
        else:
            parts.append(f"{name}: {ann}" if ann else name)
        call_parts.append(name)
    if posonly:
        parts.append("/")

    # Positional-or-keyword
    start = len(posonly)
    for i, a in enumerate(pos, start=start):
        name = a.arg
        ann = _expr_src(src, a.annotation)
        if i >= num_no_default:
            dsrc = _expr_src(src, defaults[i - num_no_default]) or "None"
            if ann:
                parts.append(f"{name}: {ann}={dsrc}")
            else:
                parts.append(f"{name}={dsrc}")
        else:
            parts.append(f"{name}: {ann}" if ann else name)
        call_parts.append(name)

    # Var positional
    if args.vararg is not None:
        var_ann = _expr_src(src, args.vararg.annotation)
        if var_ann:
            parts.append(f"*{args.vararg.arg}: {var_ann}")
        else:
            parts.append(f"*{args.vararg.arg}")
        call_parts.append(f"*{args.vararg.arg}")
    else:
        if args.kwonlyargs:
            parts.append("*")

    # Keyword-only
    for a, d in zip(args.kwonlyargs, args.kw_defaults or [None] * len(args.kwonlyargs)):
        name = a.arg
        ann = _expr_src(src, a.annotation)
        if d is None:
            parts.append(f"{name}: {ann}" if ann else name)
        else:
            dsrc = _expr_src(src, d) or "None"
            if ann:
                parts.append(f"{name}: {ann}={dsrc}")
            else:
                parts.append(f"{name}={dsrc}")
        call_parts.append(f"{name}={name}")

    # Var keyword
    if args.kwarg is not None:
        kw_ann = _expr_src(src, args.kwarg.annotation)
        if kw_ann:
            parts.append(f"**{args.kwarg.arg}: {kw_ann}")
        else:
            parts.append(f"**{args.kwarg.arg}")
        call_parts.append(f"**{args.kwarg.arg}")

    return ", ".join(parts), ", ".join(call_parts)


def _iter_public_instance_methods_from_ast(
    file_path: Path, leaf_class_name: str
) -> Iterable[MethodInfo]:
    src = file_path.read_text()
    mod = ast.parse(src)
    leaf_cls: Optional[ast.ClassDef] = None
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == leaf_class_name:
            leaf_cls = node
            break
    if leaf_cls is None:
        return []
    results: List[MethodInfo] = []
    for node in leaf_cls.body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
            if name.startswith("_"):
                continue
            # skip properties
            has_property = False
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name) and dec.id in {
                    "property",
                    "cached_property",
                }:
                    has_property = True
                    break
                if isinstance(dec, ast.Attribute) and dec.attr in {
                    "property",
                    "cached_property",
                }:
                    has_property = True
                    break
            if has_property:
                continue
            if not node.args.args or node.args.args[0].arg != "self":
                continue
            params_src, call_args = _build_params_and_calls(src, node)
            doc = ast.get_docstring(node) or ""
            results.append(MethodInfo(name, params_src, call_args, doc))
    return results


def _escape_doc(doc: str) -> str:
    return doc.replace('"""', '"""')


def _extract_summary_paragraph(doc: str) -> str:
    """
    Extract the first descriptive paragraph from a leaf docstring.
    Stops before blank lines, headings, or code examples.
    """
    if not doc:
        return ""
    lines = doc.splitlines()
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            # stop at first blank line after we've collected something
            if out:
                break
            else:
                continue
        # stop on common headings or code/example indicators
        if s.lower().startswith(("examples", "parameters", "returns", "notes")):
            break
        if s.startswith("```") or s.startswith(">>>"):
            break
        out.append(ln)
    return "\n".join(out)


def _transform_leaf_doc_for_mixin(doc: str) -> str:
    """
    Adapt a leaf method docstring for a batch-mixin context:
    - Keep the content for IDE/autocomplete help
    - Convert runnable fenced blocks (```pycon/python/py) into ```text to avoid xdoctest execution
    """
    lines = doc.splitlines()
    out: List[str] = []
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("```pycon"):
            out.append(ln.replace("```pycon", "```text"))
            continue
        if stripped.startswith("```python"):
            out.append(ln.replace("```python", "```text"))
            continue
        if stripped.startswith("```py"):
            out.append(ln.replace("```py", "```text"))
            continue
        out.append(ln)
    return "\n".join(out)


def _extract_type_checking_imports(src: str) -> List[str]:
    """Capture import lines inside an 'if TYPE_CHECKING:' block in the leaf module."""
    mod = ast.parse(src)
    lines: List[str] = []
    for node in mod.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for stmt in node.body:
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    seg = _expr_src(src, stmt)
                    if seg:
                        # indent consistently for emission
                        for line in seg.splitlines():
                            lines.append(f"{line}")
    return lines


def _generate_collection_mixin(leaf_py: Path, leaf_class: str, mixin_class: str) -> str:
    methods = list(_iter_public_instance_methods_from_ast(leaf_py, leaf_class))
    needs_any = any("Any" in m.params_src for m in methods)
    leaf_src = leaf_py.read_text()
    type_checking_imports = _extract_type_checking_imports(leaf_src)
    # Derive fully qualified module path for the leaf class, e.g.
    # py3r.behaviour.features.features
    parts = list(leaf_py.parts)
    module_name = f"py3r.behaviour.{leaf_py.stem}"
    try:
        idx = parts.index("behaviour")
        subparts = parts[idx + 1 : -1] + [leaf_py.stem]
        module_name = "py3r.behaviour." + ".".join(subparts)
    except ValueError:
        pass
    lines = [
        "# AUTO-GENERATED FILE. DO NOT EDIT.",
        f"# Generated by tools/gen_batch_mixins.py from {leaf_py.as_posix()}",
        "# Regenerate with: PYTHONPATH=src python -m tools.gen_batch_mixins",
        "from __future__ import annotations",
        "",
        "from py3r.behaviour.util.collection_utils import BatchResult",
    ]
    if needs_any:
        lines.append("from typing import Any")
    # Re-emit TYPE_CHECKING imports from leaf for annotation names
    if type_checking_imports:
        lines.append("from typing import TYPE_CHECKING")
        lines.append("if TYPE_CHECKING:")
        lines += [f"    {line}" for line in type_checking_imports]
    lines += [
        "",
        f"class {mixin_class}:",
        "",
    ]
    for mi in methods:
        params = f"self{', ' + mi.params_src if mi.params_src else ''}"
        lines.append(f"    def {mi.name}({params}) -> BatchResult:")
        # Minimal batch-aware docstring for mixin methods (no executable examples)
        summary = _extract_summary_paragraph(mi.doc or "")
        lines.append('        """')
        lines.append(
            f"        Batch-mode wrapper for {leaf_class}.{mi.name} across the collection."
        )
        if summary:
            lines.append("")
            for dl in summary.splitlines():
                lines.append(f"        {dl}")
        lines.append("")
        full_ref = f"{module_name}.{leaf_class}.{mi.name}"
        lines.append(
            f"        See [`{leaf_class}.{mi.name}`][{full_ref}] for examples."
        )
        lines.append('        """')
        suffix = f", {mi.call_args}" if mi.call_args else ""
        # inplace-aware: if caller passes inplace=False, return a new collection via map_leaves
        lines.append("        _inplace = locals().get('inplace', True)")
        lines.append("        if _inplace is False:")
        call_for_map = mi.call_args if mi.call_args else ""
        lines.append(
            f"            return self.map_leaves(lambda _obj: getattr(_obj, '{mi.name}')({call_for_map}))"
        )
        lines.append(f'        return self._invoke_batch("{mi.name}"{suffix})')
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path("src/py3r/behaviour")
    targets = [
        # (leaf_file, leaf_class, output_file, mixin_class)
        (
            root / "tracking" / "tracking.py",
            "Tracking",
            root / "tracking" / "tracking_collection_batch_mixin.py",
            "TrackingCollectionBatchMixin",
        ),
        (
            root / "features" / "features.py",
            "Features",
            root / "features" / "features_collection_batch_mixin.py",
            "FeaturesCollectionBatchMixin",
        ),
        (
            root / "summary" / "summary.py",
            "Summary",
            root / "summary" / "summary_collection_batch_mixin.py",
            "SummaryCollectionBatchMixin",
        ),
    ]
    for leaf_py, leaf_cls, out_py, mixin in targets:
        content = _generate_collection_mixin(leaf_py, leaf_cls, mixin)
        out_py.parent.mkdir(parents=True, exist_ok=True)
        out_py.write_text(content)
        print(f"Wrote {out_py}")


if __name__ == "__main__":
    main()
