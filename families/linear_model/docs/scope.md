# Dense family scope

Implemented: existing estimator capability traits plus dense Float32/Float64
shape and weight validation, independent snapshots and coefficient shape metadata.
Declared extension points: preprocessing, prediction and engine dispatch.
Implemented family plumbing: dense weighted centering, sqrt-weight rescaling,
intercept reconstruction, prediction, coefficient/target shape rules and R2
(`moxie.linear_model._linear_base`, see plumbing.md). Estimator validation remains grouped by child.
Tests use both dtypes, vector/matrix targets, empty/mismatched inputs, invalid
weights, deep copy versus view, inherited lifecycle and negative capabilities.

Deferred sparse branches: `_base.py` sparse preprocessing/rescaling and
SparseCoefMixin conversions, `_coordinate_descent.py` sparse coordinate descent,
`_stochastic_gradient.py` sparse datasets and sparse intercept handling, sparse
Ridge/least-squares solver paths. CSR/CSC/COO inputs/targets/outputs and coefficient
sparsification have no typed API. Never densify user sparse inputs implicitly.
Reference: sklearn/linear_model at 4358542a1bd39af2e7e1cbace8b9fed067473c5b.
Sparse inputs are unsupported by this family. Dense factorizations use the live external linear-model linalg worktree
through the private moxie._linear_linalg namespace.

Constructor defaults and supported solver selections are recorded in each
child API. Sparse exclusions do not make unfinished dense branches complete.

Validation commands: `pixi run build`, `pixi run test`. Scaffold acceptance
requires both; the parity task checks the assembled algorithm children. Test includes two isolated
namespace consumers, drift/conflict rejection, unavailable Predictor capability,
and all existing family/base contracts. These probes do not establish estimator
parity or performance. See [plumbing.md](plumbing.md) for child responsibilities.

## Public estimator assembly

`pixi run build` assembles the real `LinearRegression`, `Ridge`,
`RidgeClassifier`, `ridge_regression`, `ElasticNet`, `Lasso`, `BayesianRidge` and their typed
configuration/results from the least_squares, ridge, elastic_net and bayesian children.
Implementations are owned by Moxie under src/moxie/linear_model. `pixi run start` runs `examples/dense_estimators.mojo` through
those public imports; `pixi run parity` runs the child differential suites.

Supported modes and limitations are detailed in the child scope/API documents.
For LinearRegression/Ridge/RidgeClassifier, scalar nonnegative fit weights
expand to same-dtype sample vectors. ElasticNet/Lasso ignore scalar weights
as their average-loss normalization cancels them.
Classifier column-vector targets are ordinary class labels, not multilabel
indicators. Invalid score weights and nonfinite regression score targets raise.
Single-target intercepts remain length-one arrays. Ridge ravels single-column
coefficients and predictions, as does this pinned reference; LinearRegression
preserves the target rank. Integer-valued numeric classifier labels only.

Unfinished dense work: Ridge SAG/SAGA, RidgeCV/RidgeClassifierCV, random/Gram coordinate
solvers, multi-task estimators, paths, SGD/Perceptron, logistic and remaining
family algorithms. The cyclic coordinate engine is implemented and consumed by
ElasticNet/Lasso (see algorithms/elastic_net/docs/scope.md). Smooth and SGD
engine directories still contain prerequisite probes, not functioning solvers. CV validation belongs to model_selection; shared
`our_rand_r` belongs to Moxie. These gaps are not sparse deferrals and the
family is not complete.

Builds consume current Moxie and family sources at declared local paths. Legacy
source hashes do not gate builds; assembly receipts record the consumed bytes.
Keep the Pixi toolchain lock and scikit-learn correctness reference revision.

BayesianRidge supports dense single-target fitting, sample weights, posterior
covariance and predictive standard deviations in both dtypes. See the
[BayesianRidge API](../algorithms/bayesian/docs/api.md) for shape adaptations,
validation evidence and numerical failure behavior. ARDRegression is deferred.
