# py3r.behaviour.script

Run and sweep parameterised Python scripts with clean subprocess isolation.

## Annotate your script

```python
from py3r.behaviour.script import Param, Output

data_path = Param("/my/data", name="data_path")   # optional default
smooth_window = Param(5, name="smooth_window")
threshold = Param(name="threshold")               # required — no default

# ... Tracking stage ...

tracking = Output(tc, name="tracking")            # output mid-pipeline

# ... Features and Summary stages ...

Output(sc.stored_info(), name="summary")          # output at the end
```

`Param` and `Output` are transparent in normal execution — the script runs exactly as written. `Param` with no default raises if the script is run standalone without a value injected.

Outputs are ordered by where they appear in the script. This position matters for early stopping — see below.

## Inspect a script

```python
from py3r.behaviour.script import inspect
inspect("my_pipeline.py")
```

```
Script: my_pipeline.py

Parameters:
  data_path            default='/my/data'  type=str
  smooth_window        default=5           type=int
  threshold            required

Outputs:
  tracking
  summary
```

## Run once

```python
from py3r.behaviour.script import run
sr = run("my_pipeline.py", {"data_path": "/their/data", "threshold": 0.8})
sr[{"data_path": "/their/data", "threshold": 0.8}]["summary"]
```

Unspecified params use their script default. Required params must be provided.

Capture only a subset of outputs with `outputs=`:

```python
sr = run("my_pipeline.py", {"threshold": 0.8}, outputs=["tracking"])
```

## Sweep parameters

```python
from py3r.behaviour.script import sensitivity
sr = sensitivity("my_pipeline.py", {"smooth_window": [3, 7]})
```

- The script default (`5`) is automatically included in the sweep and used as the nominal baseline for independent sweeps.
- Results keyed by the swept param values: `sr[{"smooth_window": 3}]["summary"]`
- Flatten scalar outputs: `sr.to_dataframe()`
- Check failures: `sr.errors`

### Non-swept required params

```python
sr = sensitivity(
    "my_pipeline.py",
    {"smooth_window": [3, 7]},
    nominal={"threshold": 0.8},   # required param not being swept
)
```

### Override the nominal baseline

```python
sr = sensitivity(
    "my_pipeline.py",
    {"smooth_window": [3, 7]},
    nominal={"smooth_window": 7},  # use 7 as baseline, not the script default
)
```

### Early termination

If your script has outputs at different stages, use `outputs=` to request only the ones you need, and `stop_after_outputs=True` to kill the subprocess as soon as the last requested output is captured.

```python
# Script has: Output(..., name="tracking") early on, Output(..., name="summary") at the end.
# Request only "tracking" — subprocess stops after Tracking stage, skipping Features + Summary.
sr = sensitivity(
    "my_pipeline.py",
    {"smooth_window": [3, 7]},
    outputs=["tracking"],
    stop_after_outputs=True,
)
```

`stop_after_outputs=True` on its own (without `outputs=`) stops after the very last output in the script, which gives no speedup. The combination of `outputs=` and `stop_after_outputs=True` is what cuts the pipeline short.

The same pattern works with `run()`.

### Grid sweep

```python
sr = sensitivity("my_pipeline.py", {"smooth_window": [3, 5, 7], "threshold": [0.5, 0.8]}, mode="grid")
```

## Rules

- `Param` values must be scalar: `bool`, `int`, `float`, or `str`.
- `name=` is required on every `Param` and `Output` call.
- No name may appear in both a `Param` and an `Output`. Duplicate names within either also raise.
- Each iteration runs in a fresh subprocess — process state does not leak between iterations.
