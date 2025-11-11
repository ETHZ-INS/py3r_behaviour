#!/usr/bin/env bash
set -euo pipefail

out_dir="docs"
mkdir -p "$out_dir"

# locate repo-root-relative css (this script is in tools/)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
css_src="$script_dir/docs/style.css"

modules=(
  py3r.behaviour
  py3r.behaviour.tracking.tracking
  py3r.behaviour.tracking.tracking_collection
  py3r.behaviour.features.features
  py3r.behaviour.features.features_collection
  py3r.behaviour.summary.summary
  py3r.behaviour.summary.summary_collection
)

for m in "${modules[@]}"; do
  python -m pydoc -w "$m"
done

mv ./*.html "$out_dir"/

# copy CSS and inject link into each HTML head
if [[ -f "$css_src" ]]; then
  cp "$css_src" "$out_dir/style.css"
  for f in "$out_dir"/*.html; do
    python - "$f" <<'PY'
import io, sys, pathlib
path = pathlib.Path(sys.argv[1])
html = path.read_text(encoding="utf-8")
link = '<link rel="stylesheet" href="style.css">'
if "</head>" in html and link not in html:
    html = html.replace("</head>", f"  {link}\n</head>")
path.write_text(html, encoding="utf-8")
PY
  done
fi

echo "Pydoc HTML generated in $out_dir/"


