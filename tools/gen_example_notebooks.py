#!/usr/bin/env python
"""Generate clean example notebooks from test pipeline scripts.

Strips ``# norender`` (test-only) cells and writes an unexecuted
``.ipynb`` into ``_artifacts/{name}/``.  Place your real dataset in
``_artifacts/{name}/data/`` and then render with::

    python tools/render_notebooks.py --from-artifacts [name]

Usage (from repo root):
    python tools/gen_example_notebooks.py                    # generate all
    python tools/gen_example_notebooks.py oft_pipeline       # generate one
    python tools/gen_example_notebooks.py --list             # show available
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import jupytext
import nbformat

# ---------------------------------------------------------------------------
# Registry — same keys as render_notebooks.py
# ---------------------------------------------------------------------------
PIPELINES: dict[str, Path] = {
    "oft_pipeline": Path("tests/oft_pipeline/oft_pipeline.py"),
    "epm_pipeline": Path("tests/epm_pipeline/epm_pipeline.py"),
}

# ---------------------------------------------------------------------------
# Helpers (intentionally duplicated from render_notebooks to avoid coupling)
# ---------------------------------------------------------------------------


def _has_norender_flag(cell: nbformat.NotebookNode) -> bool:
    """Return True if the cell's source starts with ``# norender``."""
    src = cell.source.lstrip()
    if cell.cell_type == "code" and src.startswith("# norender"):
        return True
    if cell.cell_type in ("markdown", "raw") and src.startswith("norender"):
        return True
    return False


def _strip_norender_cells(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Return a copy of *nb* with ``# norender`` cells removed."""
    nb = copy.deepcopy(nb)
    nb.cells = [c for c in nb.cells if not _has_norender_flag(c)]
    return nb


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def generate(name: str, script_path: Path, out_dir: Path | None = None) -> Path:
    """Generate a clean example notebook from a test pipeline script.

    Parameters
    ----------
    name : str
        Pipeline name (used for the output subdirectory and filename).
    script_path : Path
        Path to the ``# %%``-style Python test script.
    out_dir : Path | None
        Root output directory.  Defaults to ``_artifacts/`` at the repo root.

    Returns
    -------
    Path
        Path to the written ``.ipynb`` file.
    """
    script_path = Path(script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    if out_dir is None:
        repo_root = Path(__file__).resolve().parent.parent
        out_dir = repo_root / "_artifacts"

    artifact_dir = out_dir / name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{name}] Reading {script_path} …")
    nb = jupytext.read(str(script_path))

    # Strip test-only cells
    nb = _strip_norender_cells(nb)

    # Remove empty code cells left over after stripping
    nb.cells = [c for c in nb.cells if not (c.cell_type == "code" and not c.source.strip())]

    # Write unexecuted .ipynb
    ipynb_path = artifact_dir / f"{name}.ipynb"
    with open(ipynb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"[{name}] Written → {ipynb_path}")
    return ipynb_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "names",
        nargs="*",
        help="Pipeline name(s) to generate.  Omit to generate all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available pipelines and exit.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Root output directory (default: _artifacts/).",
    )
    args = parser.parse_args()

    if args.list:
        for k, v in PIPELINES.items():
            print(f"  {k:20s} → {v}")
        return

    # Resolve paths relative to repo root (parent of tools/)
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    targets = args.names if args.names else list(PIPELINES.keys())
    for name in targets:
        if name not in PIPELINES:
            print(f"Unknown pipeline: {name!r}.  Available: {list(PIPELINES.keys())}")
            sys.exit(1)
        generate(name, PIPELINES[name], out_dir=args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
