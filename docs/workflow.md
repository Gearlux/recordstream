# Workflows — composing runnables

The workflow combinators (`recordstream.workflow`) are the RUNNABLE-level analogue of the
composing ops: they HOLD other runnables and orchestrate them, so a multi-stage pipeline
(prepare → train → evaluate) is ONE Confluid document executed by the same
`recordstream run workflow.yaml` as any single runnable.

| Combinator | Runs |
|---|---|
| `Sequence(steps=[...])` | every step in order — the workflow itself (`None` steps skip) |
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
