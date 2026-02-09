#!/usr/bin/env python
"""Render pipeline scripts as executed HTML notebooks.

Usage:
    python tests/render_notebooks.py                    # render all
    python tests/render_notebooks.py oft_pipeline       # render one
    python tests/render_notebooks.py --list             # show available

Reads each ``# %%``-style script via jupytext, strips ``if TEST_MODE`` cells,
executes the notebook in-process, and writes an HTML file that looks like a
Jupyter notebook with inline plots and outputs.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

# ---------------------------------------------------------------------------
# Registry of pipeline scripts
# ---------------------------------------------------------------------------
PIPELINES: dict[str, Path] = {
    "oft_pipeline": Path("tests/oft_pipeline/oft_pipeline.py"),
    "epm_pipeline": Path("tests/epm_pipeline/epm_pipeline.py"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_test_cell(cell: nbformat.NotebookNode) -> bool:
    """Return True if this cell is a TEST_MODE guard or pure-test block."""
    if cell.cell_type != "code":
        return False
    src = cell.source.strip()
    # Cells that start with 'if TEST_MODE'
    if re.match(r"^if\s+TEST_MODE", src):
        return True
    return False


def _strip_test_cells(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Return a copy of *nb* with TEST_MODE cells removed."""
    nb = copy.deepcopy(nb)
    nb.cells = [c for c in nb.cells if not _is_test_cell(c)]
    return nb


def _enable_inline_plots(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Replace ``show=False`` with ``show=True`` so plots render in-cell."""
    nb = copy.deepcopy(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace("show=False", "show=True")
    return nb


def _inject_inline_backend(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Activate the IPython inline backend so figures render in-cell.

    Also suppresses the ``FigureCanvasAgg is non-interactive`` warning that
    ``plt.show()`` emits when called from library code outside IPython's
    display integration.
    """
    nb = copy.deepcopy(nb)
    setup_src = (
        "%matplotlib inline\n"
        "import warnings\n"
        "warnings.filterwarnings(\n"
        '    "ignore", message="FigureCanvasAgg is non-interactive"\n'
        ")\n"
    )
    setup_cell = nbformat.v4.new_code_cell(source=setup_src)
    setup_cell.metadata["tags"] = []
    nb.cells.insert(0, setup_cell)
    return nb


def _tidy_preamble(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Keep TEST_MODE = True (needed for data paths) and skip heavy viz."""
    nb = copy.deepcopy(nb)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source.strip()
        # Cell that defines TEST_MODE
        if re.search(r"^TEST_MODE\s*=\s*True", src, re.MULTILINE):
            # Inject SKIP_HEAVY_VIZ = True if the script uses it
            if "SKIP_HEAVY_VIZ" not in src:
                cell.source += "\nSKIP_HEAVY_VIZ = True\n"
            else:
                cell.source = re.sub(
                    r"SKIP_HEAVY_VIZ\s*=.*",
                    "SKIP_HEAVY_VIZ = True  # skip heavy viz in rendered notebook",
                    cell.source,
                )
            break
    return nb


def render(name: str, script_path: Path, out_dir: Path | None = None) -> Path:
    """Render a single pipeline script to HTML.

    Parameters
    ----------
    name : str
        Human-readable name used for the output file.
    script_path : Path
        Path to the ``# %%``-style Python script.
    out_dir : Path | None
        Directory for the HTML output.  Defaults to ``_rendered/`` next to
        the script.

    Returns
    -------
    Path
        Path to the written HTML file.
    """
    script_path = Path(script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    if out_dir is None:
        out_dir = script_path.parent / "_rendered"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{name}] Reading {script_path} …")
    nb = jupytext.read(str(script_path))

    # Remove test-only cells, tidy preamble, add inline backend
    nb = _strip_test_cells(nb)
    nb = _tidy_preamble(nb)
    nb = _inject_inline_backend(nb)
    nb = _enable_inline_plots(nb)

    # Execute the notebook from the script's own directory
    cwd = str(script_path.parent)
    print(f"[{name}] Executing (cwd={cwd}) …")
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": cwd}},
    )
    client.execute()

    # Convert to HTML
    print(f"[{name}] Converting to HTML …")
    exporter = HTMLExporter()
    exporter.template_name = "lab"  # clean modern look
    body, _resources = exporter.from_notebook_node(nb)

    html_path = out_dir / f"{name}.html"
    html_path.write_text(body, encoding="utf-8")
    print(f"[{name}] Written → {html_path}")
    return html_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names",
        nargs="*",
        help="Pipeline name(s) to render.  Omit to render all.",
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
        help="Output directory for HTML files (default: _rendered/ next to each script).",
    )
    args = parser.parse_args()

    if args.list:
        for k, v in PIPELINES.items():
            print(f"  {k:20s} → {v}")
        return

    # Resolve repo root so relative PIPELINES paths work
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    targets = args.names if args.names else list(PIPELINES.keys())
    for name in targets:
        if name not in PIPELINES:
            print(f"Unknown pipeline: {name!r}.  Available: {list(PIPELINES.keys())}")
            sys.exit(1)
        render(name, PIPELINES[name], out_dir=args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
