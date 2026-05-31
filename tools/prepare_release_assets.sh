#!/usr/bin/env bash
# Prepare release assets for oft_pipeline:
#   1. Generate clean example notebook (strips test-only cells)
#   2. Execute it against the full dataset in _artifacts/oft_pipeline/data/
#      and render docs/examples/oft_pipeline.md
#   3. Zip _artifacts/oft_pipeline/ → _artifacts/oft_pipeline.zip
#      (excluding macOS hidden files)
#
# Usage (from repo root):
#   bash tools/prepare_release_assets.sh
#
# Before running, place the full dataset at:
#   _artifacts/oft_pipeline/data/tracking/   ← DLC CSVs
#   _artifacts/oft_pipeline/data/tags.csv

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="_artifacts/oft_pipeline/data"

# --- Preflight check --------------------------------------------------------
if [[ ! -d "$DATA_DIR/tracking" ]] || [[ -z "$(ls "$DATA_DIR/tracking"/*.csv 2>/dev/null)" ]]; then
    echo "ERROR: No tracking CSVs found in $DATA_DIR/tracking/"
    echo "Place the full dataset there before running this script."
    exit 1
fi

if [[ ! -f "$DATA_DIR/tags.csv" ]]; then
    echo "ERROR: Missing $DATA_DIR/tags.csv"
    exit 1
fi

# --- Step 1: generate clean notebook ----------------------------------------
echo "==> Generating example notebook..."
python tools/gen_example_notebooks.py oft_pipeline

# --- Step 2: execute against full dataset and render docs -------------------
echo "==> Rendering notebook (--from-artifacts)..."
python tools/render_notebooks.py --from-artifacts oft_pipeline

# --- Step 3: zip the artifact folder ----------------------------------------
echo "==> Packaging _artifacts/oft_pipeline/ → _artifacts/oft_pipeline.zip"
cd "_artifacts"
rm -f oft_pipeline.zip
zip -r oft_pipeline.zip oft_pipeline \
    -x "*.DS_Store" \
    -x "*/__MACOSX/*" \
    -x "*/._*"
cd "$REPO_ROOT"

echo ""
echo "Done. Attach _artifacts/oft_pipeline.zip to the draft GitHub release,"
echo "then publish to trigger docs deployment."
