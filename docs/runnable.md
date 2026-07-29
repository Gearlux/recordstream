# Runnables and entry points

A **runnable** is any object exposing a no-arg `run()` — a trainer, an evaluator, a dataset
processor, a workflow. It is the unit `recordstream run` executes:

```yaml
# config.yaml — the ONE runner shape for every kind of run
runnable: !class:mypkg.Classifier
  task: fit                # ← the one knob: fit / evaluate / test / predict
  train_set: !ref:my_split.train
```

```bash
python -m recordstream.cli run config.yaml     # builds `runnable:`, calls .run()
```

### Top-level keys reach the runnable

The runnable is built **against the whole document**, so a *flat* config works: a top-level
key broadcasts into the same-named constructor parameter, with no nesting and no `!ref:`.

```yaml
runnable: !class:mypkg.Classifier
  model: !lazy:mypkg.Backbone { name: resnet18 }

train_set: !class:recordstream.sources.HuggingFaceSource { path: mnist, split: train }
max_epochs: 3          # -> Classifier(max_epochs=3)
batch_size: 32         # -> Classifier(batch_size=32)
```

If you write your own runner CLI, build the bound node with
`recordstream.cli.materialize_runnable(node)` rather than a bare `flow()`. A bare flow builds
the node in isolation, so every top-level key above is dropped — and dropped *silently*:
`train_set` becomes `None` and `max_epochs` quietly falls back to its default, leaving a run
that looks configured and is not.

```python
from recordstream.cli import materialize_runnable

@app.script_command(flow_mode="manual")      # "auto" is the bare flow this replaces
def run(runnable: Any) -> None:
    materialize_runnable(runnable).run()
```

## The problem entry points solve

A merged train+eval class exposes SEVERAL capabilities from one class, dispatched off its
`task` knob. Without extra information, a discovery consumer (a config generator, a visual
editor) would have to *assume* one class per capability — it cannot know that `Classifier`
both trains and evaluates, nor which `task` value means "evaluate". The `@entrypoint` marker
declares exactly that, per method.

## A straightforward example

```python
from recordstream import ProgressReporting, TorchRunner, entrypoint, run_entrypoint

class Classifier(TorchRunner, ProgressReporting):
    """One class, four capabilities — run() dispatches off the ``task`` knob."""

    def __init__(self, task: str = "fit"):
        self.task = task

    def run(self) -> None:
        run_entrypoint(self, self.task)      # the markers below ARE the dispatch table

    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None: ...                      # gradient training

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None: ...                 # metrics over the VALIDATION split

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None: ...                     # metrics over the held-out TEST split

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None: ...                  # stream predictions
```

Each marker states three things: the **`task` value** that reaches this method through
`run()`, the **`role`** capability label (conventionally `"trainer"` / `"evaluator"` /
`"predictor"`; free-form for new capabilities), and — when several methods share a role —
which one is the **`primary`** (here `test` is the default evaluator; `evaluate` is the
secondary, validation-split variant).

## What the introspectors return

Real output for the class above (these are executed facts, not sketches):

```python
>>> from recordstream import runnable_entrypoints, entrypoint_tasks
>>> runnable_entrypoints(Classifier)
{'fit':      {'task': 'fit',      'role': 'trainer',   'primary': True},
 'evaluate': {'task': 'evaluate', 'role': 'evaluator', 'primary': False},
 'test':     {'task': 'test',     'role': 'evaluator', 'primary': True},
 'predict':  {'task': 'predict',  'role': 'predictor', 'primary': True}}

>>> entrypoint_tasks(Classifier, "trainer")
['fit']
>>> entrypoint_tasks(Classifier, "evaluator")
['test', 'evaluate']            # PRIMARY FIRST — "run this as an evaluator" means task: test
>>> entrypoint_tasks(Classifier, "exporter")
[]                              # unknown role: empty, never an error
```

`runnable_entrypoints` walks the MRO (an inherited entry point is found; a subclass override
wins) and reads the marker off the raw function object, so property getters never fire.

## Dispatching: `run_entrypoint`

`run()` above dispatches *through* the markers rather than restating them:

```python
>>> Classifier(task="test").run()      # calls Classifier.test()
>>> Classifier(task="export").run()
ValueError: Unknown task 'export'; expected one of ['fit', 'evaluate', 'test', 'predict'].
```

The declared tasks are listed in **declaration order**, so the error reads as the class's
capability list. The return value of the entry-point method is passed through.

Write the `run()` body this way rather than as a hand-written `{task: method}` dict. The dict
states the same mapping a second time, and the copies drift in one direction that bites: a
config generator pins `task:` from `entrypoint_tasks` — the markers — so a capability added to
the markers and forgotten in the dict produces a *generated* config that dies at dispatch with
"unknown task" while discovery advertises it as supported. With `run_entrypoint` there is one
table, and adding a fifth `@entrypoint` method is all that adding a fifth capability takes.

## How a consumer uses this

A config generator asked for "an evaluator config for `Classifier`" calls
`entrypoint_tasks(Classifier, "evaluator")[0]` → `"test"` and pins `task: test` in the YAML
it emits — one `recordstream run` then dispatches correctly with no human editing. The same
walk over every discovered class tells a visual editor which classes to offer in a
"trainer" picker versus an "evaluator" picker, even when both answers are the same class.

## The two marker mixins

Orthogonal to entry points, a runnable may inherit two stateless mixins:

- **`TorchRunner`** — declares "my `run()` needs autograd" (`__needs_autograd__ = True`,
  duck-typed). A GUI executor that evaluates nodes under `torch.inference_mode()` re-enables
  autograd for the duration of `run()`. Inference-only runnables deliberately do NOT inherit
  it. A merged class can even make it dynamic — a property returning `self.task == "fit"`,
  so the same class trains under autograd and predicts under inference mode. The class and
  the flag are named for different things on purpose: the *class* for the framework whose
  execution mode is at stake (autograd is a torch concept), the *flag* for what it decides —
  which is what makes the dynamic property above read correctly.
- **`ProgressReporting`** — a framework-free progress sink: the executor injects
  `(value, total, desc) -> None` via `set_progress_callback()`, the runnable drains it via
  `self._report_progress(step, total, "epoch 3")` from its loop. With no sink injected
  (a plain CLI run) every call is a silent no-op.

Pins: `tests/test_runnable.py` / `tests/test_entrypoint.py`.
