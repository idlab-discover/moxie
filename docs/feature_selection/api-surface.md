# Public API coverage

Public names and callable parameter names/defaults follow the local scikit-learn
checkout, with typed native adaptations documented in each estimator guide.
All reference exports now resolve to dense implementations. This describes
export coverage, not complete numerical or Python API parity.

Moxie's root and isolated-consumer dependency manifests declare all 19 public
exports and the native extension types. All implementation modules live in
`src/moxie/feature_selection`. Builds use current local dependency sources.
See [integration requirements](migration.md) for external acceptance cases.

## Coverage against the local reference

Reference: `../scikit-learn` at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`, using
`sklearn/feature_selection/__init__.py` and its 19 public exports.

| Public surface | Implementation status |
| --- | --- |
| SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect | Implemented in the root and univariate assemblies for Float32/Float64 built-in scoring; incomplete parity |
| chi2, f_classif, f_oneway, f_regression, r_regression | Dense Float32/Float64 inputs in the root and univariate assemblies; native results and dtype differences apply |
| VarianceThreshold | Dense Float32/Float64 implementation in the root assembly; see [usage and limits](variance.md) |
| SelectorMixin | Implemented native trait delegating to SelectionState; see [SelectorMixin](selector-mixin.md) |
| SelectFromModel | Dense wrapper with typed Lasso/ElasticNet/Ridge/DecisionTreeClassifier adapters and conditional partial_fit using IncrementalPCA; see [usage and limits](from_model.md) |
| RFE | Dense recursive elimination with typed estimator adapters; see [RFE](rfe.md) |
| SequentialFeatureSelector | Dense greedy selection through the shared CV engine; see [usage and limits](sequential.md) |
| RFECV | Dense RFE paths with shared CV contracts; see [usage and limits](rfecv.md) |
| mutual_info_classif, mutual_info_regression | Dense Float32/Float64 implementation and typed selector scorer; see [usage and limits](mutual-info.md) |

All 19 names appear in the root source and assembled initializers. All nineteen names
resolve to implementations: VarianceThreshold, six univariate selectors, seven
scoring functions, SelectFromModel, RFE, RFECV, SequentialFeatureSelector and SelectorMixin. This count measures export coverage, not numerical parity.

Scorer input and result dtypes are now separate, enabling
Float32 selectors with Float64 statistics and Float32 transforms.
Univariate built-in scorers accept independent Float32/Float64 target dtypes.
Classification scorers and all six univariate selectors also support native
StringTarget and exact Int64 labels; tests include values above 2**53 and Int64
extremes. Remaining dense gaps include broader custom-scorer and
estimator-wrapper parity, and numerical stress cases.
Univariate stride checks cover stepped/reversed, column-major and offset inputs
with strided targets in both dtypes; this is bounded coverage, not arbitrary-layout
or numerical-stress certification. See the [univariate scope](univariate/scope.md).

`tests/feature_selection/test_fluent.mojo` covers concrete and shared-trait chaining for all six
univariate selectors and VarianceThreshold, in Float32 and Float64. It checks
that returned borrows mutate the original estimator, preserve clone semantics,
and do not undo successful earlier calls when a later fit fails.

Native adaptations include typed scorer configuration and parameter records,
learned-attribute accessors such as `scores_()`, fitting that returns a mutable estimator borrow, `ScoreResult` rather than Python tuples, and `f_oneway(groups)` rather
than Python variadic arrays. Python metadata routing, container output settings,
warning transport and reflection remain unavailable. Sparse support is explicitly
deferred by the accepted project scope.

The existing parity harness compares extracted local sklearn scoring function
bodies and support-mask policies, with validation/label helpers supplied by the
harness. It does not execute sklearn's full estimator validation pipeline or
prove end-to-end wrapper compatibility. Its tolerance is rtol=5e-5, atol=2e-12;
passing bounded fixtures does not establish extreme-tail or cancellation accuracy.

Root `pixi run build` and `pixi run test` compile and exercise the univariate
contracts alongside variance, including a Float32/Float64 pipeline, public score
imports and both isolated consumers. `pixi run parity` runs variance, univariate, model-based, RFE, RFECV, mutual-information and sequential-selection reference
harnesses against the root namespace. Univariate checks run within the full family commands.

`SelectFromModel` adds typed model adapters, stable importance thresholds and
feature limits. Its model algorithms remain in external linear-model/decomposition dependencies;
incremental updates are available through `IncrementalSelectionEstimator`.
Its real-model contracts and sklearn importance/mask parity run from the root.
See [the model selector guide](from_model.md) for native differences and gaps.

A production DecisionTreeClassifier adapter supports all four estimator-based
selectors with independent native string/Int64 targets. See
[classifier selection](classifier-selection.md) for capabilities, dependency
provenance and bounded binary/multiclass parity coverage.
