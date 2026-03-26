# DataFlux Mandates

- **Functional Purity:** Transforms are plain Python callables. Never introduce base classes or complex inheritance for data operations.
- **Sample Triplet:** All data flows through the `Sample(input, target, metadata)` NamedTuple. Never bypass metadata — full traceability is mandatory.
- **Lazy Evaluation:** Pipelines MUST remain lazy iterators until explicitly consumed. Never eagerly materialize entire datasets.
- **Serialization Symmetry:** Every pipeline configuration MUST be serializable via **Confluid** manifests for full reproducibility.
- **Passive Introspection:** Pipeline discovery MUST use the `discovery` module for automatic JSON manifest generation. Never require manual tool definitions.
- **Storage Protocols:** All storage backends MUST implement the `DataSource`/`DataSink` protocols. Never couple the core engine to a specific format.

## Testing & Validation
- **Pipeline Parity:** Test that serialized-then-deserialized pipelines produce identical output to the original.
- **Multiprocess Safety:** Parallel pipelines MUST use the `spawn` context. Verify pickle-safety of all operations.
- **Line Length:** 120 characters (Black, isort, flake8).
