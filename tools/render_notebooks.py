#!/usr/bin/env python
"""Render pipeline scripts as executed HTML notebooks.

Usage (from repo root):
    python tools/render_notebooks.py                    # render all
    python tools/render_notebooks.py oft_pipeline       # render one
    python tools/render_notebooks.py --list             # show available

Reads each ``# %%``-style script via jupytext, strips cells marked with
``# norender``, executes the notebook, and writes an HTML file that looks
like a Jupyter notebook with inline plots and outputs.
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
# Registry of pipeline scripts (paths relative to repo root)
# ---------------------------------------------------------------------------
PIPELINES: dict[str, Path] = {
    "oft_pipeline": Path("tests/oft_pipeline/oft_pipeline.py"),
    "epm_pipeline": Path("tests/epm_pipeline/epm_pipeline.py"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_norender_flag(cell: nbformat.NotebookNode) -> bool:
    """Return True if the cell's source starts with ``# norender``."""
    src = cell.source.lstrip()
    # Code cells: first line is literally "# norender"
    if cell.cell_type == "code" and src.startswith("# norender"):
        return True
    # Markdown cells: jupytext strips the leading '# ' so the raw source
    # begins with "norender"
    if cell.cell_type in ("markdown", "raw") and src.startswith("norender"):
        return True
    return False


def _strip_norender_cells(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Return a copy of *nb* with ``# norender`` cells removed."""
    nb = copy.deepcopy(nb)
    nb.cells = [c for c in nb.cells if not _has_norender_flag(c)]
    return nb


def _enable_inline_plots(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Replace ``show=False`` with ``show=True`` so plots render in-cell."""
    nb = copy.deepcopy(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace("show=False", "show=True")
    return nb


def _inject_inline_backend(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Insert a setup cell that configures matplotlib for inline rendering.

    Also suppresses the ``FigureCanvasAgg is non-interactive`` warning that
    ``plt.show()`` emits inside non-interactive kernels.
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
    """Ensure SKIP_HEAVY_VIZ = True so heavy deps are skipped in rendering."""
    nb = copy.deepcopy(nb)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source.strip()
        if re.search(r"^TEST_MODE\s*=\s*True", src, re.MULTILINE):
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


# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------


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

    # Transform the notebook for rendering
    nb = _strip_norender_cells(nb)
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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

    # Always resolve paths relative to the repo root (parent of tools/)
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
