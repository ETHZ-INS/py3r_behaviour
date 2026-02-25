#!/usr/bin/env bash
set -euo pipefail

RENDER_ARGS=()

# Pass --from-artifacts through to render_notebooks.py
if [[ "${1:-}" == "--from-artifacts" ]]; then
    RENDER_ARGS+=(--from-artifacts)
    shift
fi

echo "Rendering notebooks ${RENDER_ARGS[*]:+(${RENDER_ARGS[*]})}..."
python tools/render_notebooks.py "${RENDER_ARGS[@]}"

echo "Building MkDocs site..."
mkdocs build --clean

echo "MkDocs site built in ./site"
