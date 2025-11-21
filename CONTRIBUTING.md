# Contributing to py3r-behaviour

Thanks for your interest in contributing! This document explains the workflow and expectations for external contributors.

## TL;DR
- Fork the repo and create a branch off `dev` named, e.g. `feature/<yourfeature>` or `bugfix/<yourbugfix>`.
- Set up the dev environment (tests + docs), run the test suite locally, and ensure generated files are committed.
- Open a Pull Request against `dev` and fill in the PR template (check at least one type with `[x]`).
- Keep PRs atomic and focused; include doctests for new public APIs.

## Getting started

### Fork and clone

First, fork the repository on GitHub:
- Visit https://github.com/ETHZ-INS/py3r_behaviour and click “Fork”.
- This creates your own copy under your account (with the full contents).

Then clone your fork locally and add the upstream remote:
```bash
git clone https://github.com/<you>/py3r_behaviour.git
cd py3r_behaviour
git remote add upstream https://github.com/ETHZ-INS/py3r_behaviour.git
git fetch upstream
```

### Create an appropriately named branch off `dev`
```bash
git checkout -b <branchtype>/<short-name> upstream/dev
```

### Requirements and environment
- Python ≥ 3.12
- pip ≥ 21.3

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```
Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Editable install (package + docs + tests)
```bash
python -m pip install -U pip
pip install -e ".[dev]" "mkdocstrings[python]" mkdocs-material mike mkdocs-autorefs xdoctest pytest packaging
```

## Before you push
### Generate batch mixins and make sure no uncommitted changes remain after generation:
```bash
PYTHONPATH=src python -m tools.gen_batch_mixins
git diff --exit-code || (echo "Please commit updated generated files" && exit 1)
```

### Run tests (including doctests in docstrings):
```bash
pytest -q --xdoctest --xdoctest-modules
```

### Optionally preview the docs locally:
```bash
mkdocs build --strict
```

## Pull Requests
- Target branch: `dev` (not `main`).
- Fill in the PR template:
  - Check at least one “Type of change” with `[x]` (strictly lowercase x). This auto-applies labels via a bot.
  - Provide a user-facing change bullet for the release notes.
  - Describe testing.
- Keep PRs small and focused (one logical change). Prefer multiple small PRs over one large one.
- Add doctests for new public methods/classes, using fenced `pycon` code blocks as shown in existing docstrings.
- If adding data to doctests, prefer using `importlib.resources` via the `data_path` helper.
- Do not attempt to publish docs yourself; CI handles that. Do not bump versions in PRs to `dev`.

## Versioning and releases
- Version bumps happen only when merging to `main`. CI enforces version sequencing in `pyproject.toml`.
- The docs site is deployed by CI on `main` pushes (versioned with `mike`).

## Coding style and docstrings
- Python 3.12+ with type hints on public APIs.
- Keep control-flow shallow; avoid deep nesting; prefer early returns; avoid broad try/except.
- Docstrings:
  - Use `pycon` fenced blocks for examples with a blank line before the closing fence.
  - Examples should be lightweight and runnable under CI (avoid heavy dependencies).
  - Convert NumPy booleans to Python bools in doctest asserts, e.g., `bool(...)`.
  - For collections/folder examples, it’s fine to use illustrative paths in docs pages; for doctests, isolate with temp dirs.

## Commit messages and labels
- Commit messages: use clear, descriptive subject lines. Conventional Commits style is welcome (`feat:`, `fix:`, `docs:`, `chore:`), but not required.
- Labels are applied automatically from the PR template checkboxes; ensure you use `[x]` on the correct line(s).

## Reporting bugs and proposing features
- Open an issue with a concise description, expected vs actual behavior, and minimal reproduction if possible.
- For features, outline the use case and proposed API; small design discussions in an issue before opening a PR are encouraged.

## Code of Conduct
- Be respectful and constructive. Collaborators and maintainers reserve the right to moderate discussions and contributions to keep the project healthy.


