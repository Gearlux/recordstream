# Runnables and entry points

A **runnable** is any object exposing a no-arg `run()` — a trainer, an evaluator, a dataset
processor, a workflow. It is the unit `recordstream run` executes:

```yaml
# config.yaml — the ONE runner shape for every kind of run
runnable: !class:mypkg.Classifier
  task: fit                # ← the one knob: fit / evaluate / test / predict
  train_set: !ref:train_split          # a DatasetSplit with `split: train`
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

train_set: !class:recordstream.sources.huggingface.HuggingFaceSource { path: mnist, split: train }
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

## A dataset run on several processes: `DatasetProcessor(workers=N)`

`DatasetProcessor` — the source → ops → sink runnable — builds its records one at a time, in the
running process. With `workers: N` it builds them in N spawn worker processes: record `i` is
`source[i]` run through the stream's ops in one of them, and the sink, which stays in the running
process, receives the records in index order. So the sink writes the files a sequential run
writes, name for name and byte for byte.

```yaml
runnable: !class:recordstream.processing.DatasetProcessor
  workers: 8
  stream: !class:recordstream.core.stream.Stream
    source: !class:my_package.TrafficSource {count: 1000, seed: 7}   # record i drawn from (seed, i)
    ops:
      - !class:my_package.Render {}
  sink: !class:recordstream.storage.hdf5.HDF5Sink {path: ./runs/traffic.h5}
```

Measured with a signal generator whose record `i` is drawn from `(seed, i)` (32 records of about
5.8 s each, a 16-core machine): 184.5 s one at a time, 32.4 s on 8 workers, 26.4 s on 16 —
every record the same, byte for byte.

What a run with workers needs, and what it refuses before the first record:

- **A source that can be indexed** — `len()` and `[i]`, with `source[i]` the record iteration
  gives at position `i`. An iterable-only source is refused:
  `DatasetProcessor: workers=3 builds each record in a worker from its index, and the source
  (OnlyIterable) cannot be indexed — give it len() and [i], or set workers: 1`.
- **Ops that pickle**, as every spawn route asks; the source is pickled once into each worker.
- **No stream-level op**: a `Parallel` in the list starts workers of its own, and a worker cannot
  start workers — refused, naming the op's position.
- **No `chunk_size`** on the stream (a stream that hands out lists of records) — refused.

A fixed N is chosen by a record's peak memory, not only by the core count: every worker holds the
record it builds, and at most `2 × N` records are being built or waiting for the sink at once (the
next index is handed out only when there is room). Each worker imports the packages afresh when it
starts — a few seconds — so a run of a handful of records gains nothing. A record that fails stops
the run with its own error; the indices not started are cancelled.

### Letting the run choose: `workers: auto`

```yaml
runnable: !class:recordstream.processing.DatasetProcessor
  workers: auto
```

The run builds record 0 in one worker and measures it there: the worker's peak memory above what it
held before the record (on a machine whose GPU shares the memory, what the GPU took is added), the
cores the record kept busy, the CUDA memory it used. From then on, before each record starts, it
reads the machine's free memory and starts the record only when all of it fits:

- every building record's planned peak — the largest peak measured, times 1.25 — and the
  footprint of every worker not started yet, within the free memory less the reserve the run
  leaves the machine: 2 GB, or a tenth of the machine's memory when that is more (what the started
  workers hold is already missing from the free memory, and counted once);
- the cores: a record that keeps 4 cores busy counts as 4;
- when the records use CUDA, 90 % of the GPU's free memory.

The count follows the free memory while the run goes, and the log says what decided it:

```text
Building 10 record(s), workers: auto — 1 worker builds record 0 to measure it.
workers: auto → 2 (memory: 4.2 GB free − 2.0 GB reserve, 1.0 GB a record + 0.3 GB a worker; cores: 12 at 1.0 each)
```

A record larger than the free memory still builds, alone, with a warning ending `— not even one
record fits the free memory; building one at a time`. The planner is `recordstream.resources`
(`WorkerGovernor`); its plan is a `WorkerPlan` of plain numbers.

### The memory guard

Every run on workers — a fixed count or `auto` — keeps the reserve: a fixed count is a ceiling that
the measured records may lower (`workers: 8 → 2 (memory: …)` in the log), and while free memory is
under the reserve no record starts:

```text
Memory: 1.7 GB free, under the 2.0 GB reserve — holding record 57 until a worker finishes (3 building).
```

Under the reserve with at most one record building, waiting frees nothing, so the run stops — its
worker ended — with `OutOfMemory` (a `MemoryError`):

```text
DatasetProcessor: the machine has 1.5 GB free, under the 2.0 GB reserve, while record 1 builds alone (records
measured so far peak at 4.0 GB) — stopped before the machine swaps; free memory, or build smaller records
```

`workers: 1` builds in the running process, without the guard.

`Stream.parallel(n)` and the `Parallel` op differ: they read the source in the running process and
send each record to a worker for the ops — right when the ops are the work. `workers` sends the
INDEX, so a source whose records are the expensive part (a generator, a decoder) runs in the
workers too. Rationale: `docs/architecture.md` §28.

Pins: `tests/test_processing_workers.py`, `tests/test_resources.py`.

### Resuming a run: `start`

A run stopped part way resumes where its sink stopped with `start: N`: it builds only `source[N:]`, the records a
whole run gives at those indices, one at a time or on workers, and never builds the first N again. The sink must
number its output from N too:

```yaml
runnable: !class:recordstream.processing.DatasetProcessor
  start: 7248                         # samples 0-7247 are written
  workers: 2
  stream: !class:recordstream.core.stream.Stream
    source: !class:my_package.TrafficSource {count: 7812, seed: 7}   # record i drawn from (seed, i)
  sink: !class:my_package.NumberedSink {path: ./runs/traffic, start: 7248}
```

The source must be indexable; otherwise the run is refused before the first record:
`DatasetProcessor: start=2 builds the records from source[2] on, and the source (OnlyIterable) cannot be indexed —
give it len() and [i], or set start: 0`. A `start` past the end builds nothing. Pins: `tests/test_processing_start.py`.
