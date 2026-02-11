#!/usr/bin/env bash
set -euo pipefail

echo "Rendering notebooks..."
python tools/render_notebooks.py

echo "Building MkDocs site..."
mkdocs build --clean

echo "MkDocs site built in ./site"
