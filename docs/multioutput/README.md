# Multioutput estimators

Moxie owns `MultiOutputClassifier`, `MultiOutputRegressor`, `ClassifierChain`,
and `RegressorChain` in `src/moxie/multioutput.mojo`. Layout helpers and capability
contracts live in adjacent private modules. The retired sibling is not a build
input; its Git history preserves the original implementation.

- [Native API](api-surface.md) covers parameters, learned attributes and composition.
- [Scope](scope.md) records supported behavior and deferred features.
- [Plumbing](plumbing.md) specifies layout helpers, CV and validation.
- [Production example](../../examples/multioutput.mojo) uses DummyRegressor.

Tests live in `tests/multioutput/`. Run Moxie's root build, test and parity tasks,
or `pixi run python tools/multioutput_checks.py test` followed by
`pixi run python tools/multioutput_checks.py parity-only` for focused validation.
The parity oracle uses scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Its 36 comparisons pair an exact
test component and production `Ridge`/`RidgeClassifier` components with the
same sklearn estimators; they establish wrapper behavior, not parity for every
production estimator. Labels, one-neighbor probabilities and orders are exact;
Float64 scores and Ridge outputs use 1e-12 and Float32 Ridge outputs 2e-5.

Toolchain compatibility remains Mojo 1.0.0 and MAX 26.5.0 through Moxie's Pixi
lock. Shared numerical and random sources are assembled live.

## Capability contracts

| Contract | Shared capabilities |
| --- | --- |
| `MultiOutputClassifierAPI` | `ClassifierMixin` |
| `MultiOutputRegressorAPI` | `RegressorMixin` |
| `MultiOutputProbability` | `MultiOutputClassifierAPI`, `ProbabilityPredictor` |

Independent probabilities are a list of matrices. Classifier-chain probabilities
and decisions are matrices in original target order. Optional capabilities depend
on the component traits and remain absent for unsupported components.
