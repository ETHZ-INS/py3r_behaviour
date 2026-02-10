#!/usr/bin/env python
"""Render pipeline scripts as Markdown for MkDocs Material.

Usage (from repo root):
    python tools/render_notebooks.py                    # render all
    python tools/render_notebooks.py oft_pipeline       # render one
    python tools/render_notebooks.py --list             # show available

Reads each ``# %%``-style script via jupytext, strips cells marked with
``# norender``, executes the notebook, and writes a Markdown file with
extracted images suitable for inclusion in a MkDocs Material site.

Cell inputs and outputs are wrapped in distinct HTML containers
(``nb-cell-input`` / ``nb-cell-output``) so MkDocs Material can style
them differently.
"""

from __future__ import annotations

import argparse
import base64
import copy
import html as html_mod
import os
import re
import sys
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

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

# Regex for stripping ANSI escape codes from error tracebacks
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


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


# ---------------------------------------------------------------------------
# Notebook → Markdown converter (replaces nbconvert MarkdownExporter)
# ---------------------------------------------------------------------------


def _notebook_to_markdown(
    nb: nbformat.NotebookNode,
    images_dir_name: str,
) -> tuple[str, dict[str, bytes]]:
    """Convert an executed notebook to structured Markdown.

    Each code cell's source is wrapped in a ``nb-cell-input`` container and
    each output in a ``nb-cell-output`` container so they can be styled
    independently in the MkDocs Material theme.

    Parameters
    ----------
    nb : NotebookNode
        The executed notebook.
    images_dir_name : str
        Subdirectory name for extracted images (e.g. ``"oft_pipeline_files"``).

    Returns
    -------
    body : str
        The Markdown text.
    image_outputs : dict[str, bytes]
        Mapping of ``filename`` → raw image bytes for files that should be
        written into *images_dir_name/*.
    """
    parts: list[str] = []
    image_outputs: dict[str, bytes] = {}

    for cell_idx, cell in enumerate(nb.cells):
        # -- Markdown / raw cells ------------------------------------------
        if cell.cell_type in ("markdown", "raw"):
            parts.append(cell.source)
            parts.append("")
            continue

        if cell.cell_type != "code":
            continue

        # -- Code cell input -----------------------------------------------
        source = cell.source.strip()
        if source:
            parts.append('<div class="nb-cell-input" markdown>')
            parts.append("")
            parts.append("```python")
            parts.append(source)
            parts.append("```")
            parts.append("")
            parts.append("</div>")
            parts.append("")

        # -- Code cell outputs ---------------------------------------------
        for out_idx, output in enumerate(cell.get("outputs", [])):
            otype = output.output_type

            # --- stdout / stderr stream -----------------------------------
            if otype == "stream":
                text = output.text.rstrip("\n")
                if text:
                    escaped = html_mod.escape(text)
                    parts.append('<div class="nb-cell-output">')
                    parts.append(f"<pre><code>{escaped}</code></pre>")
                    parts.append("</div>")
                    parts.append("")

            # --- rich output (execute_result / display_data) --------------
            elif otype in ("execute_result", "display_data"):
                data = output.get("data", {})

                # Prefer image/png > image/svg+xml > text/html > text/plain
                if "image/png" in data:
                    img_data = data["image/png"]
                    if isinstance(img_data, str):
                        img_data = base64.b64decode(img_data)
                    fname = f"output_{cell_idx}_{out_idx}.png"
                    image_outputs[fname] = img_data
                    parts.append('<div class="nb-cell-output nb-output-figure" markdown>')
                    parts.append("")
                    parts.append(f"![output]({images_dir_name}/{fname})")
                    parts.append("")
                    parts.append("</div>")
                    parts.append("")

                elif "image/svg+xml" in data:
                    svg = data["image/svg+xml"]
                    parts.append('<div class="nb-cell-output nb-output-figure">')
                    parts.append(svg)
                    parts.append("</div>")
                    parts.append("")

                elif "text/html" in data:
                    html_content = data["text/html"]
                    parts.append('<div class="nb-cell-output nb-output-table">')
                    parts.append(html_content)
                    parts.append("</div>")
                    parts.append("")

                elif "text/plain" in data:
                    text = data["text/plain"].rstrip("\n")
                    if text:
                        escaped = html_mod.escape(text)
                        parts.append('<div class="nb-cell-output">')
                        parts.append(f"<pre><code>{escaped}</code></pre>")
                        parts.append("</div>")
                        parts.append("")

            # --- error traceback ------------------------------------------
            elif otype == "error":
                tb = "\n".join(output.get("traceback", []))
                if tb:
                    tb = _ANSI_RE.sub("", tb)
                    escaped = html_mod.escape(tb)
                    parts.append('<div class="nb-cell-output nb-cell-error">')
                    parts.append(f"<pre><code>{escaped}</code></pre>")
                    parts.append("</div>")
                    parts.append("")

    return "\n".join(parts), image_outputs


# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------


def render(name: str, script_path: Path, out_dir: Path | None = None) -> Path:
    """Render a single pipeline script to Markdown with extracted images.

    Parameters
    ----------
    name : str
        Human-readable name used for the output file.
    script_path : Path
        Path to the ``# %%``-style Python script.
    out_dir : Path | None
        Directory for the Markdown output.  Defaults to ``docs/examples/``
        relative to the repo root.

    Returns
    -------
    Path
        Path to the written Markdown file.
    """
    script_path = Path(script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    if out_dir is None:
        repo_root = Path(__file__).resolve().parent.parent
        out_dir = repo_root / "docs" / "examples"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{name}] Reading {script_path} …")
    nb = jupytext.read(str(script_path))

    # Transform the notebook for rendering
    nb = _strip_norender_cells(nb)

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

    # Convert to Markdown + images
    images_dir_name = f"{name}_files"
    print(f"[{name}] Converting to Markdown …")
    body, image_outputs = _notebook_to_markdown(nb, images_dir_name)

    # Prepend auto-generated comment
    header = "<!-- AUTO-GENERATED by tools/render_notebooks.py — do not edit manually -->\n\n"
    body = header + body

    # Write the Markdown file
    md_path = out_dir / f"{name}.md"
    md_path.write_text(body, encoding="utf-8")

    # Write image assets into {name}_files/
    if image_outputs:
        images_dir = out_dir / images_dir_name
        images_dir.mkdir(parents=True, exist_ok=True)
        for fname, data in image_outputs.items():
            (images_dir / fname).write_bytes(data)
        print(f"[{name}] Wrote {len(image_outputs)} image(s) to {images_dir}/")

    print(f"[{name}] Written → {md_path}")
    return md_path


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
        help="Output directory for Markdown files (default: docs/examples/).",
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
