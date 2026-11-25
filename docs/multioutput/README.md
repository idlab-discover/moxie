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
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Its 54 comparisons cover an exact
test component and production `Ridge`, `RidgeClassifier`,
`RandomForestClassifier` and `LogisticRegression` components. Labels,
one-neighbor and seeded forest probabilities and orders are exact; Float64
scores and Ridge outputs use 1e-12 and Float32 Ridge outputs 2e-5. Logistic
probabilities use a 2e-6 absolute tolerance for differences in L-BFGS iterates.
Chain CV parity exercises native KFold and integer CV for both chain types,
including stratified classifier responses and a missing class in a forest fold.
Side-by-side notebooks for the gallery examples live in `notebooks/multioutput/`.
The classifier-chain notebook uses dense logistic regression for independent
fits and the ten-chain probability ensemble, with seeded synthetic count data
to make the example runnable offline. Run it with
`pixi run -e notebooks notebooks notebooks/multioutput/plot_classifier_chain_yeast.ipynb`.

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
