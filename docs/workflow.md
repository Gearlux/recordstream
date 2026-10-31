# Workflows — composing runnables

The workflow combinators (`recordstream.workflow`) are the RUNNABLE-level analogue of the
composing ops: they HOLD other runnables and orchestrate them, so a multi-stage pipeline
(prepare → train → evaluate) is ONE Confluid document executed by the same
`recordstream run workflow.yaml` as any single runnable.

| Combinator | Runs |
|---|---|
| `Sequence(steps=[...], only=[...], name, title)` | every step in order — the workflow itself (`None` steps skip), each checked before the first runs; a named inner `Sequence` is ONE step an outer `only` picks (an option made of several runs) |
| `Conditional(condition, if_true, if_false)` | one of two branches, by a condition (`None` branch = do nothing, the sequence continues) |
| `Switch(select, cases, default)` | one of several branches, keyed by the select's value (coerced to `str`) |

Conditions are `@configurable` predicates — a no-arg `__call__() -> bool`: `PathExists(path)`
(the canonical cache check), `Not(condition)`, `AllOf(conditions)`, `AnyOf(conditions)` — or any
zero-arg callable, a plain `bool`, or a deferred marker resolving to one.

## The compelling case: a resume-safe pipeline

Re-run the SAME document after a crash (or just again tomorrow) and it skips the work whose
artifact already exists — *memoise and continue*:

```yaml
runnable: !class:recordstream.workflow.Sequence
  steps:
    # Train ONLY when the checkpoint is missing. On a cache hit the !lazy: branch is
    # not just skipped — it is never even BUILT (no model / dataset materialised).
    - !class:recordstream.workflow.Conditional
      condition: !class:recordstream.workflow.PathExists
        path: $MODEL_ROOT/run1/model.ckpt
      if_true: null                     # cache hit -> skip, Sequence continues
      if_false: !lazy:TrainModel        # @configurable classes resolve by registered NAME
        ckpt: $MODEL_ROOT/run1/model.ckpt

    # Always runs; the report FORMAT is a Switch on a plain config value — one key a
    # CLI override can flip (--select text) without touching the workflow shape.
    - !class:recordstream.workflow.Switch
      select: json
      cases:
        json: !lazy:Evaluate { report: $MODEL_ROOT/run1/report.json, fmt: json }
        text: !lazy:Evaluate { report: $MODEL_ROOT/run1/report.txt,  fmt: text }
```

**The guarantee:** only the selected branch's `run()` is ever called, and a branch wired
`!lazy:` is only *constructed* when selected. [`examples/workflow_pipeline.py`](../examples/workflow_pipeline.py)
runs this exact shape twice and asserts both: pass 2 re-evaluates without retraining, and the
trainer class records exactly ONE construction across both passes.

## The steps around the runs: `Require`, `ReplaceFolder`, `RunConfig`

What a shell script does around its runs — refuse early, empty an output folder once, run configs in turn — are
steps of the document too:

| Step | Does |
|---|---|
| `Require(condition, message)` | refuses unless `condition` holds; `message` says what to do |
| `ReplaceFolder(path, marker, parents)` | a folder without `marker` is deleted and made again with the marker in it; a folder holding it is kept, so a re-run after an interruption resumes into it. A filesystem root, the home folder and a folder whose parent is missing are refused — `parents: true` makes the missing parents instead |
| `RunConfig(config, scopes, settings, name, hint, checked_first, environment)` | runs `recordstream run <config> --scope <s> key=value` in its own process; a failed run stops the workflow with its exit status. `environment: {NAME: value}` sets a variable for that process where the machine leaves it unset (a machine's own value wins); the dry run names what it sets |

The conditions say what they check: `EnvSet(name)` (an environment variable is set), `FreeSpace(path, gigabytes)`
(measured on the nearest folder that exists, so `path` may be the folder a run will make), `HasFiles(path,
pattern)`, `PathExists(path)`, and `Not` / `AllOf` / `AnyOf` over them.

```yaml
# make_store.yaml
store:
  root: ${env:DATA_ROOT}/Store/2.0
  min_free_gb: 170
  sets: []                          # empty = every named step

quick: !scope:quick
  store:
    root: ${env:DATA_ROOT}/Store/quick
    min_free_gb: 5

runnable: !class:recordstream.workflow.Sequence
  only: !ref:store.sets
  steps:
    - !class:recordstream.workflow.Require
      condition: !class:recordstream.workflow.EnvSet { name: DATA_ROOT }
      message: source project.bashrc
    - !class:recordstream.workflow.Require
      condition: !class:recordstream.workflow.FreeSpace { path: "${store.root}", gigabytes: "${store.min_free_gb}" }
    - !class:recordstream.workflow.ReplaceFolder { path: "${store.root}", marker: .store_v2 }
    - !class:recordstream.workflow.RunConfig
      name: images
      config: convert_images.yaml
      settings: { output: "${store.root}/images" }
      hint: recordstream run download_images.yaml
    - !class:recordstream.workflow.RunConfig
      name: labels
      config: convert_labels.yaml
      settings: { output: "${store.root}/labels" }
```

```bash
recordstream run make_store.yaml                          # the whole store
recordstream run make_store.yaml --scope quick            # the quick store
recordstream run make_store.yaml store.sets=[labels]      # the checks, then the labels step only
recordstream run make_store.yaml store.root=/elsewhere --dry_run
```

- **A config key read inside a string is dotted:** `"${store.root}/images"` reads the config's `store.root`; a
  plain `${root}` reads the ENVIRONMENT variable `root`. A value given on the command line (`store.root=/elsewhere`)
  reaches every `${store.root}`, and beats a scope's.
- **`config:` is handed to `recordstream run` as written:** a relative path is the run folder's;
  `pkg://<distribution>/<path>` is the file a package ships.
- **`only`** runs the steps it names and every step without a name; a name no step has is refused, listing the
  names there are.

### Every step is checked before the first runs

A `Sequence` plans every step before it runs one, and a failed check stops it there — a long first step never runs
ahead of a refusal a later step would make:

```text
Error: [5/5] labels: FilesSource: pattern '*.json' matched no files under 'labels' (…) — run: recordstream run download_labels.yaml
```

A `RunConfig` is planned by its own process's dry run, so its config's own checks count (an empty input folder, a
sink that cannot say where it stopped). A step whose inputs an EARLIER step writes — the check after a
conversion — says `checked_first: false`, and is checked when its turn comes.

### The dry run: `--dry_run`

`recordstream run <config> --dry_run` runs nothing: it prints what each step would check, delete, create and run,
goes on past a step that would stop, and exits 1 when a real run would stop:

```text
DRY RUN: nothing is changed
[1/5] DATA_ROOT is set
      ok: DATA_ROOT is set
[2/5] 170 GB free at /data/Store (331 GB free)
      ok: 170 GB free at /data/Store (331 GB free)
[3/5] replace /data/Store/2.0
      delete /data/Store/2.0 (88.4G)
      create /data/Store/2.0 and its marker .store_v2
[4/5] images
      recordstream run convert_images.yaml output=/data/Store/2.0/images
      DatasetProcessor:
        ok: the source can be read
        records: 1000, start: 0, to_build: 1000, sink: ImageSink
[5/5] labels
      …
DRY RUN: every step would run
```

`--plan_format json` prints the same plan as one JSON line — what a `RunConfig` reads from its process, and what a
tool returns as is (`recordstream.workflow.Plan.from_dict`). A runnable says what it would do through a `plan()`
method returning a `Plan`; one without it is planned as `run <Class>`.

`--plan_format` only applies to a dry run. Without `--dry_run` it is refused and nothing runs, so asking for the plan
never starts a real run; a format other than `text` or `json` is refused too:

```text
$ recordstream run make_store.yaml --plan_format json
Error: --plan_format only applies with --dry_run; add --dry_run to print the plan (--plan_format json alone ran nothing)
$ recordstream run make_store.yaml --dry_run --plan_format xml
Error: recordstream run: --plan_format must be one of text, json; got 'xml'
```

A refusal is one `Error:` line and exit 1, a failed step's own exit status; `--debug` keeps the traceback.
[`examples/workflow_steps.py`](../examples/workflow_steps.py) runs a two-step workflow, its dry run and a refusal.

## Semantics worth knowing

- **Branches are lazy twice over**: unchosen `!lazy:` branches are never built; chosen ones are
  flowed inside `run()` (zero-arg construction of the combinators themselves does no work).
- **`Conditional` with a `None` branch is "skip and continue"** — the enclosing `Sequence`
  proceeds to the next step; nothing blocks.
- **`Switch` select** may be a plain value (`select: json` — overridable from a CLI), a
  predicate, or any zero-arg callable; an unmatched key (or `None`) runs `default` (`None` =
  no-op).
- **GUI cooperation is inherited**: the combinators mix in `TorchRunner` (a wrapped trainer's
  `loss.backward()` survives an inference-mode executor) and `ProgressReporting` (the injected
  progress callback is FORWARDED to whichever branch is running).

Runnables and the `task`/`role` entry-point markers: [runnable.md](runnable.md).
