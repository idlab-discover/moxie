# Multioutput estimators

Moxie owns `MultiOutputClassifier`, `MultiOutputRegressor`, `ClassifierChain`,
and `RegressorChain` in `src/moxie/multioutput.mojo`. Layout helpers and capability
contracts live in adjacent private modules. The retired sibling is not a build
input; its Git history preserves the original implementation.

- [Native API](api-surface.md) covers parameters, learned attributes and composition.
- [Scope](scope.md) records supported behavior and deferred features.
- [Plumbing](plumbing.md) specifies layout helpers, CV and validation.
- [Production example](../../examples/multioutput.mojo) uses DummyRegressor.

Tests live in `tests/multioutput/`. Run
`pixi run python tools/multioutput_checks.py test` followed by
`pixi run python tools/multioutput_checks.py parity-only` for focused validation.
The parity oracle uses scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. The comparisons cover a test
component and production `Ridge`, `RidgeClassifier`, `RandomForestClassifier`,
`LogisticRegression`, `DecisionTreeClassifier`, `KNeighborsClassifier`,
`GaussianNB`, `LinearDiscriminantAnalysis`, `DummyClassifier`, and `SVC`
components. Labels, orders, and the tested tree, KNN, dummy and seeded forest
probabilities match exactly. The SVC chain uses the SVM family's decision
tolerances of `rtol=3e-5` and `atol=3e-6`.
Float64 scores and Ridge outputs use 1e-12 and Float32 Ridge outputs 2e-5. Logistic
probabilities use a 2e-6 absolute tolerance for differences in L-BFGS iterates.
Weighted GaussianNB uses 1e-12 for differences in accumulation order and
log-probability normalization. The same bound covers incremental GaussianNB
and GaussianNB chain CV with native List labels. Parallel parity covers weighted
Ridge in both dtypes and weighted and incremental GaussianNB with native List
labels, all with `n_jobs=2`. Float64 LDA probabilities use
1e-12 for differences in the SVD and probability normalization.
Chain CV parity exercises native KFold and integer CV for both chain types,
including stratified classifier responses and missing classes in forest and
tree folds. Mixed-dtype parity covers Float32 input with Float64 augmentation
using Ridge and RidgeClassifier, with the same 1e-12 tolerance as the other
Float64 Ridge cases. Fit metadata parity covers standard-name regression
weights and aliased GaussianNB weights with independent fits and chain CV.
Random-order parity checks permutations and subsequent RNG draws exactly
across both chains. It also checks predictions with configured CV and
weighted fit. Named-input contracts cover fitted name ownership, cloning, refit,
mismatch errors, incremental updates, weights and CV. With pandas available in the
oracle's Python environment, run `python tests/multioutput/parity.py --named`
after building the family test binaries. This also checks the four scikit-learn
wrappers with named DataFrame inputs, including rejection of renamed columns.
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
