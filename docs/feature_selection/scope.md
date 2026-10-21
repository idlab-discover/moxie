# Dense feature-selection scope

Supported and tested: existing family capability contracts, Float32/Float64
NuMojo rank-2 support-mask mapping, inverse zero insertion, owned masks/indices,
feature names, reset/recommit/copy behavior, shape rejection, unfitted errors,
score-result shape validation, and two isolated assembled consumer namespaces.
The root also implements dense VarianceThreshold for Float32/Float64, including
NaN-aware population statistics, strict thresholds and the zero-threshold range
guard. See [variance.md](variance.md) for supported methods and native differences.
The root also exports SelectKBest, SelectPercentile, SelectFpr, SelectFdr,
SelectFwe, GenericUnivariateSelect, chi2, f_classif, f_oneway, f_regression and
r_regression from the existing univariate implementations. See the
[univariate scope](univariate/scope.md) for tested settings and
remaining dense limitations. All reference public names have dense implementations;
no full sklearn parity is claimed.

Deferred sparse branches: CSR/CSC/COO support slicing and inverse indptr
reconstruction in sklearn/feature_selection/_base.py; sparse variance, univariate
scores, mutual-information discrete columns, sparse estimator importances and
sparse input paths of model-based/RFE/sequential selection. These require the
explicitly deferred foundation.sparse owner. No sparse backend, coercion or
implicit densification exists. The accepted session-plan/SPARSITY-SCOPE.md
overrides historical sparse requirements. Dense numerical dependencies retain
the operation exclusions in the shared numerics decision.

No scaffold default requires a sparse setting. Dense mutual information
preserves dense `discrete_features='auto'` semantics. Python metadata routing,
pandas output, Python warning objects and dynamic attribute lookup remain binding
work. Native Float32/Float64 matrices and typed tagged support returns are the
explicit API; integer/object matrices are not accepted by this helper. Native
empty-name lists mean omitted names. See plumbing.md for lifecycle and validation
responsibilities.

Validation commands are `pixi run feature-selection-build` and
`pixi run feature-selection-test`.
Test includes negative
compilation for an unavailable sparse selector and missing predictor capability.
Scaffold acceptance requires these commands to pass; algorithm parity is not a
scaffold requirement. Copy/mapping comparisons use exact equality (no floating
arithmetic in selection). No performance claim is made.

Variance checks: `pixi run parity` compares the local reference fit/support bodies;
`pixi run test` covers lifecycle, invalid inputs and both isolated consumers.

The root commands also compile and execute the univariate score, policy,
selector and shared-selection tests. Public API composition covers variance
followed by regression selection in both dtypes, owned outputs, inverse mapping,
feature names and cloning. Both isolated consumers import real variance and
univariate selectors. The mixed-score-dtype negative check remains enforced.
Root parity runs both variance and univariate reference harnesses against the
root assembly; no score or policy implementation is duplicated for integration.

SelectFromModel is implemented for dense Float32/Float64 inputs through typed
SelectionEstimator adapters. Supplied adapters wrap real Lasso/ElasticNet/Ridge, DecisionTreeClassifier and IncrementalPCA models
and support fit/prefit, multi-output importances, norm aggregation, thresholds,
integer/callable feature limits, array sample weights, cloning, feature names,
and shared transforms. See [from_model.md](from_model.md) for defaults, tested
settings, external model restrictions and native differences. Sparse input and
importance branches remain deferred; no default needs sparse output. Other
estimator adapters and broader fit-parameter routing remain unfinished dense
work. Incremental fitting is conditional on IncrementalSelectionEstimator, with
a real IncrementalPCA loading adapter supplied. Python attribute-path/callable reflection is replaced by
typed adapter methods. Root build/test/parity includes this implementation and
both isolated consumers; no full sklearn parity or performance claim is made.

RFE and the public SelectorMixin are implemented for dense Float32/Float64.
RFE supports count/fraction selection and elimination steps, stable squared
importance ranking, final refitting, sample weights, names and conditional
prediction/scoring through typed adapters. Supplied supervised adapters are Lasso,
ElasticNet, Ridge and DecisionTreeClassifier. See [RFE](rfe.md) and [SelectorMixin](selector-mixin.md) for native
adaptations, validation and remaining estimator coverage. RFE itself does not
use cross-validation; sequential selection adopts
the shared CV engine as described below. Sparse branches remain deferred.

Both mutual-information functions are implemented with shared native neighbor
queries and RNG. See [mutual-info.md](mutual-info.md) for tested dense modes,
native dtype/mutability adaptations, serial n_jobs behavior, and oracle limits.
Sparse columns remain deferred. Root build/test/parity includes MI.

SequentialFeatureSelector now supports dense greedy forward/backward selection
through the shared model-selection CV engine, with typed scoring/adapters,
count/fraction/auto/tol modes, default or materialized folds, names, cloning and
explicit fit/scoring metadata. Lasso, ElasticNet and DecisionTreeClassifier are the real-model parity
consumers. See [sequential.md](sequential.md) for classification/unsupervised
extension coverage, serial execution and native API limits. Sparse paths remain
deferred. Root build/test/parity covers SFS and
both isolated consumers; no full sklearn parity or performance claim is made.

RFECV now evaluates RFE paths on shared CV splits, selects the feature count,
and refits a final estimator. It publishes owned scores, fold masks and rankings,
with typed scorers, separate fit/scoring weights and conditional inference
capabilities. See [RFECV](rfecv.md) for its native API and limitations. Root
build/test/parity includes actual sklearn RFECV comparisons with real Lasso,
ElasticNet and DecisionTreeClassifier; both isolated consumers fit the public implementation.

`DecisionTreeClassifierImportance` adds a production classifier adapter for
dense Float32/Float64 features and native string/Int64 targets. See
[classifier selection](classifier-selection.md) for the supported single-output
contract, dependency provenance and parity checks.
