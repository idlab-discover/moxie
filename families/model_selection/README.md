# Model selection in Moxie

Pure Mojo dense model-selection algorithms built on the preserved moxie
`scaffold-v1.1` derivation (`4843da32ff0f`; see `SCAFFOLD_BASE`).

Implemented groups:

- All fifteen splitter classes, `check_cv`, and dense one-/two-array
  `train_test_split`, with Int64, Float32/Float64 and String stratification
  labels and groups (via `SplitLabels` and the shared `moxie._labels` encoder).
- Typed `ParameterGrid` and `ParameterSampler`, including mixed distributions.
- Serial `GridSearchCV` and `RandomizedSearchCV`, with multi-metric results,
  custom refitting, failure handling and conditional estimator delegation.
- Serial `cross_validate`, `cross_val_score` and `cross_val_predict` through
  typed adapters, including numeric/String outputs and aligned class responses.
- Serial `validation_curve` and `learning_curve`, including
  weighted evaluation and optional learning-curve timings.
- Serial `permutation_test_score`, with grouped shuffling and target-dependent CV.

Native SVG learning/validation displays and explicit incremental CV adapters are
also implemented. Thresholds and halving remain queued. See [scope](docs/scope.md);
exported API placeholders are not working algorithms.

From the Moxie root:

```sh
pixi run python tools/model_selection_checks.py all
pixi run python families/model_selection/tools/check.py test
```

Moxie owns all source, child projects, examples and tests here. Live manifests
resolve shared contracts to Moxie and preserve the copied compatible Pixi locks.
The retired sibling is not a build input. `tools/completeness.py` deliberately
fails for the four threshold/halving placeholders.

- [Splitter interfaces and exact-index checks](algorithms/splitters/docs/scope.md)
- [Search estimators and runnable example](algorithms/search/docs/search.md)
- [Candidate enumeration and sampling](algorithms/search/docs/scope.md)
- [CV adapters, results and real-estimator tests](algorithms/validation/docs/scope.md)
- [Family contracts and ownership](docs/plumbing.md)
- [Shared estimator interfaces](docs/interfaces.md)
- [Contribution rules](AGENTS.md)

Runnable examples: `examples/splitters.mojo`,
`algorithms/search/examples/candidates.mojo`,
`algorithms/validation/examples/learning_curve.mojo`, and the inherited
estimator/plumbing examples. `pixi run start` runs the inherited estimator example. Build tasks compile
all examples. Reference evidence and raw logs stay under ignored `docs/private/`;
cloning the repository does not copy them. No Python executes in library code.

[Provenance](PROVENANCE.json) records the original revision, pre-existing edits,
lock hashes and archived source. Unvalidated threshold/halving drafts remain
explicit deferrals in [scope](docs/scope.md).
