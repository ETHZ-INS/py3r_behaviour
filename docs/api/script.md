`py3r.behaviour.script` lets you run and sweep parameterised Python pipeline scripts without modifying them.

Annotate any scalar script variable with [`Param`][py3r.behaviour.script.param.Param] and any intermediate result with [`Output`][py3r.behaviour.script.param.Output]. Both are transparent during normal execution — the script runs unchanged on its own. When invoked via [`run`][py3r.behaviour.script.runner.run] or [`sensitivity`][py3r.behaviour.script.runner.sensitivity], parameter values are injected into each subprocess and outputs are captured and returned in a [`ScriptResults`][py3r.behaviour.script.results.ScriptResults] container.

Each iteration runs in a fresh subprocess, so process state never leaks between runs. `stop_after_outputs=True` combined with `outputs=` terminates the subprocess as soon as the last requested output is captured, allowing you to short-circuit long pipelines when only early-stage results are needed.

## Annotation

::: py3r.behaviour.script.param.Param

::: py3r.behaviour.script.param.Output

## Running and sweeping

::: py3r.behaviour.script.runner.run

::: py3r.behaviour.script.runner.sensitivity

::: py3r.behaviour.script.runner.inspect

## Results

::: py3r.behaviour.script.results.ScriptResults
    options:
      filters:
        - "!^__"
        - "!^_"
