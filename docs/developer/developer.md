**For `py3r.behaviour` package contributors/developers only**

Copy-paste the following commands to set up a local dev environment, run tests (including doctests), and build the docs locally.

### 1) Clone and create a virtual environment

```bash
# Clone your fork (or the upstream repo)
git clone https://github.com/<you>/py3r_behaviour.git
cd py3r_behaviour

# Create and activate a virtual environment (macOS/Linux)
python -m venv .venv
source .venv/bin/activate

# On Windows (Powershell)
# python -m venv .venv
# .\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies (package + docs + tests)

```bash
python -m pip install -U pip
pip install -e ".[dev]" "mkdocstrings[python]" mkdocs-material mike mkdocs-autorefs xdoctest pytest packaging
```

### 3) Generate batch mixins (required before committing/building)

```bash
# Ensure src is on the import path when generating
PYTHONPATH=src python -m tools.gen_batch_mixins

# Optional: verify that generation didn't leave uncommitted changes
git diff --exit-code || echo "Reminder: commit updated generated mixin files"
```

### 4) Run tests (including doctests inside docstrings)

```bash
pytest -q --xdoctest --xdoctest-modules
```

### 5) Build docs locally

```bash
# Live-reloading dev server at http://127.0.0.1:8000
mkdocs serve

# Or build a static site into ./site (in .gitignore)
mkdocs build --strict
```

### 6) Docs publishing (handled by CI)

- Do not publish docs. GitHub Actions handles versioned docs deploys on pushes to `main`.
- On PRs, CI builds docs and uploads a “site” artifact you can download for preview.
- On pushes to `main`, the Release workflow reads the version from `pyproject.toml`, tags `vX.Y.Z`, and deploys docs with `mike` to `gh-pages` with alias `latest`.

Notes:
- CI (GitHub Actions) enforces:
  - mixin generation (clean git diff)
  - tests (including xdoctest)
  - docs build
  - version gate against the latest tag on PRs to main and pushes to main
- For parquet usage in examples, prefer CSV in doctests unless parquet engines are installed (`pyarrow` or `fastparquet`).


