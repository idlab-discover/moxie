# SelectFromModel

`SelectFromModel` fits a cloned estimator or reads a prefitted one, then selects
features by their importances. It delegates optimization to the wrapped model.
The family supplies adapters for the live dense Lasso, ElasticNet,
DecisionTreeClassifier and IncrementalPCA implementations. Both Float32 and Float64 are supported.

```mojo
from moxie.linear_model import Lasso
from moxie.feature_selection import SelectFromModel, LassoImportance, ModelThreshold, MaxFeatures

var selector = SelectFromModel(
    LassoImportance(Lasso(alpha=0.1)),
    threshold=ModelThreshold(String("1.25*mean")),
    max_features=MaxFeatures(5),
)
selector.fit(X, y.copy())
var selected = selector.transform(X)
```

Run the complete example in `examples/feature_selection/from_model.mojo` against the assembled
`builds/deps` namespace. The consumed linear-model modules are external,
live source dependencies; this family does not modify their solvers.

## Selection and lifecycle

Defaults match sklearn: `threshold=None`, `prefit=False`, `norm_order=1`,
`max_features=None` and `importance_getter="auto"`. Native `MaxFeatures()`
represents None; `MaxFeatures(k)` represents a nonnegative integer. Values above
the input feature count are accepted and capped by the available columns.
Ties for the limit prefer earlier columns, and retained columns preserve their
original order. Features equal to the threshold are retained.

Thresholds accept `ModelThreshold(Float64(...))`, `"mean"`, `"median"`, and
scaled strings such as `"1.25*mean"`, wrapped as above. The default is `1e-5`
for Lasso and ElasticNet whose `l1_ratio` is close to one, otherwise the mean.
Vector importances use absolute values. Matrix coefficients use column norms:
nonzero integer orders and positive/negative infinity are supported. NaNs and
infinities in importances follow the reference aggregation/mask behavior.

Ordinary fit clones constructor configuration, so the supplied model is not
fitted or mutated, and repeated fits discard previous learned state, including
warm-start coefficients. `prefit=True` allows support/transform immediately when
the supplied model is fitted. Calling fit in that mode copies fitted state and
ignores training values/targets/weights. Learned accessors such as `threshold_()`
and `estimator_()` require fit or partial fit even in prefit mode. Copies returned by accessors
own their state. Failed fits preserve previously committed native state.

`set_params(threshold=...)` recalculates support without refitting. `get_params()`
and `set_params(params)` use `SelectFromModelParams`, including typed nested
estimator configuration. `clone(selector)` recursively clones the constructor
estimator and is unfitted; cloning a prefit selector does not retain its fitted
model. Use `fit_named(X, y, names)` for native string feature names.

Array-valued `sample_weight` is forwarded by the fit overload to the real
estimator. No arbitrary Python keyword/metadata routing is exposed. Scalar
weights currently have no selector overload. Transform and inverse-transform
use `SelectionState` and return owned arrays in the input dtype, including
empty-column selections. Negative strides are supported. Inverse input requires
at least one feature and finite values, matching the shared selector contract.

## Typed extensions

`SelectionEstimator` extends the family's `ImportanceProvider` and base estimator
contracts. An adapter supplies cloning, fitting, feature count, raw importances,
and the default-threshold and NaN policies. The supplied `LassoImportance`,
`ElasticNetImportance` and `RidgeImportance` own a real model in `.model`; their
`"auto"` and `"coef_"` getters return its coefficients. Ridge is not
l1-penalized, so its default threshold is the mean, as in sklearn. Other getter strings fail clearly. An adapter
for an estimator exposing both attributes must prefer `coef_` for `"auto"`.
Custom getter logic belongs in a typed adapter's `importances` method; Python
attribute paths and dynamic callable objects are not interpreted.

A data-dependent limit implements `FeatureLimitPolicy.resolve[dtype](X)` and is
passed as `max_features=YourPolicy()`. It is evaluated during fit and stored as
`max_features_()`. Before fit, a callable policy raises, including in prefit mode.
Its type needs a default constructor and independently copied configuration. See `HalfFeatures` in
`tests/feature_selection/test_from_model.mojo` for an example. The policy type is fixed at compile
time; replacing it with a different type requires a new selector.

## Incremental fitting

`partial_fit(X, y=None)` is available only when the adapter implements
`IncrementalSelectionEstimator`. Lasso and ElasticNet lack this capability, so
calling it on their selectors fails at compile time. The first call clones the
constructor configuration; later calls update the learned model. An ordinary
`fit` resets it, and `partial_fit` after `fit` continues that fitted state.
Support and thresholds are recomputed from the current importances when queried.

```mojo
from moxie.decomposition import IncrementalPCA
from moxie.feature_selection import IncrementalPCAImportance, IncrementalPCAUpdateParams

var selector = SelectFromModel(
    IncrementalPCAImportance(IncrementalPCA[DType.float64](1)),
    importance_getter="components_",
)
selector.partial_fit(first_batch)
selector.partial_fit(next_batch, partial_fit_params=IncrementalPCAUpdateParams())
var selected = selector.transform(X)
```

This adapter selects original columns using PCA loading norms; it does not return
PCA projections. `importance_getter="components_"` is required, as it is with
sklearn's IncrementalPCA. See `examples/feature_selection/from_model_incremental.mojo` for a
complete example. The underlying decomposition implementation is a verified,
live external dependency; this family supplies only its adapter.

An incremental adapter defines a copyable `PartialFitParams` record and
`partial_fit_selection(X, y, params: Optional[Self.PartialFitParams])`. The
selector forwards its `partial_fit_params` record on every batch. This permits
adapter-specific classes or sample weights without requiring them from all
models. `IncrementalPCAUpdateParams` contains `check_input=True`; PCA does not
accept sample weights. Native PCA always checks shape and finiteness even when
`check_input=False`. Use its default `copy=True`: the selector does not expose
PCA's separate input-mutating methods required for `copy=False`.

A callable feature limit is evaluated once on the first partial fit, or on each
ordinary fit. With `prefit=True`, the first partial fit copies the already fitted
model; later calls do nothing. This branch ignores feature names, matching the
reference. Otherwise `partial_fit_named(X, y, names, partial_fit_params=...)`
records names on the first call and checks their order on later named batches.
Unnamed later batches preserve existing names; names provided only on a later
batch do not create learned names. Python name warnings are not emitted.

Native updates operate on an owned copy and commit only after success, preserving
previous learned state on failure. Consequently Python estimator object identity
is not preserved; `estimator_()` returns an independent snapshot. Cumulative
updates follow sklearn's default, metadata-routing-disabled path. They do not
reproduce the pinned reference's routing-enabled branch that reclones on every
batch. General Python metadata routing remains unavailable.

Other estimator adapters, arbitrary fit-parameter routing, Python bindings/tags,
container output and warning transport remain gaps. Sparse inputs and importances
remain deferred. The adopted Lasso/ElasticNet restrictions (cyclic fitting, no
Gram precomputation, no multitask group penalty or CV) remain in force.

## Validation

`pixi run build` and `pixi run test` compile all family modules, the example and
two isolated consumers. Tests exercise real Lasso/ElasticNet fitting, multiple
outputs, weights, clone/refit/prefit ownership, threshold updates, names, invalid
inputs, empty selections and callable limits in both dtypes.

`pixi run parity` includes `tests/feature_selection/parity_from_model.py`. It executes threshold,
importance-aggregation and mask bodies from the recorded local sklearn revision,
including positive/negative strides, ties, zero coefficients and nonfinite
importances. Real native Lasso/ElasticNet wrapper cases use orthogonal designs
whose analytic solutions validate fit/weight routing independently of the wrapper.
Masks and transformed values compare exactly. Numerical tolerances are
Float32 rtol `3e-5`/atol `2e-8` and Float64 rtol `3e-8`/atol `2e-12`
for bounded fixtures, accommodating reduction order and generic-power differences; they do not certify ill-conditioned models or extreme norms.
This is not sklearn's complete estimator validation or metadata-routing pipeline.
No performance claim is made.

`tests/feature_selection/test_from_model_incremental.mojo` covers cumulative state, clone/reset,
prefit snapshots, names, failed-update rollback and first-call feature limits.
The incremental example and both isolated consumers use real IncrementalPCA.
`tests/feature_selection/parity_from_model_incremental.py` executes the actual pinned sklearn
SelectFromModel and IncrementalPCA across 24 cases / 72 batches, including
prefit, partial fit after ordinary fit, two component counts, both dtypes and
negative strides. Masks, row counts and transformed values compare exactly;
loading magnitudes, thresholds and running moments use Float32 rtol `2e-4` /
atol `2e-5` and Float64 rtol `3e-8` / atol `3e-9`. These bounded, separated-spectrum
fixtures accommodate native SVD/reduction order and retained Float32 factors;
they do not certify degenerate bases or extreme conditioning.

`DecisionTreeClassifierImportance` adds a production classifier adapter for
dense Float32/Float64 features and native string/Int64 targets. See
[classifier selection](classifier-selection.md) for the supported single-output
contract, dependency provenance and parity checks.
