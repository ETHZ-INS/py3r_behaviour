# Script — AI context page

*This page is written for AI assistants. See [Use with AI](index.md) for the gateway prompt.*

*For full signatures and parameter details, see the [script API reference](../api/script.md).*

---

## What the script module is for

`py3r.behaviour.script` is for **operationalising a finished pipeline**. Once
the user has built and validated an analysis pipeline interactively, they
formalise it as a plain Python script and use the script runner to:

- run it repeatedly with different parameter values, and
- collect the outputs into a structured results container.

This is the right tool when the user wants to run the same pipeline across
many parameter combinations (sensitivity analysis) or re-run it with different
data paths.

---

## Writing a script

A script is an ordinary Python file. Two special calls mark the things the
runner needs to know about:

- `Param(default, name=...)` — declares a parameter the runner can inject.
  Returns `default` during normal execution; returns the injected value when
  run via `run()` or `sensitivity()`.
- `Output(value, name=...)` — marks a value for the runner to capture. Returns
  `value` unchanged in all cases.

```python
# pipeline.py
from py3r.behaviour.script import Param, Output
from py3r.behaviour import TrackingCollection

data_path = Param('/my/data', name='data_path')
threshold  = Param(0.6,       name='threshold')

tc = TrackingCollection.from_dlc_folder(data_path, fps=30)
tc.each.filter_likelihood(threshold)
# ... rest of pipeline ...

sc = tc.to_features().to_summary()
sc.each.time_true('in_arena').store(name='time_in_arena')

Output(sc, name='summary')
```

The script runs as-is with defaults — no runner required during development.

---

## Inspecting a script

Before running, use `inspect()` to see what parameters and outputs a script
exposes:

```python
from py3r.behaviour.script import inspect
inspect('pipeline.py')
```

---

## Running a script once

```python
from py3r.behaviour.script import run

sr = run('pipeline.py', {'data_path': '/data/cohort2', 'threshold': 0.7})
summary = sr[{'data_path': '/data/cohort2', 'threshold': 0.7}]['summary']
```

`run()` returns a `ScriptResults` container. Each run is keyed by its parameter
dict; outputs are accessed by name within that key.

---

## Sensitivity analysis — sweeping parameters

```python
from py3r.behaviour.script import sensitivity

sr = sensitivity(
    'pipeline.py',
    params={'threshold': [0.5, 0.6, 0.7, 0.8]},
)
```

`sensitivity()` runs the script once per parameter combination, each in its own
subprocess. Failed iterations are recorded but do not stop the sweep.

By default, `mode='independent'` varies one parameter at a time while holding
others at their nominal (script default) values. Use `mode='grid'` for a full
cartesian product. See the API reference for `nominal` and other options.

---

## Key points

- `Param` and `Output` are no-ops during normal script execution — the script
  always runs correctly on its own.
- Each iteration runs in a **subprocess**, so the script must be importable and
  self-contained.
- `ScriptResults` is a structured dict — see the API reference for indexing and
  export methods.
- The script module is for **finished pipelines only**. Do not use it during
  exploratory development.
