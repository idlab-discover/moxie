# Random forests and extra trees

`RandomForestClassifier`, `RandomForestRegressor`, `ExtraTreesClassifier`, and
`ExtraTreesRegressor` are native Mojo estimators exported by `moxie.ensemble`.
They use the family forest engine and the live tree dependency. Extra trees
use randomized splits and default to `bootstrap=False`; random forests default
to bootstrapping. All four default to 100 trees.

```mojo
from moxie.ensemble import RandomForestRegressor
from moxie.base import clone

var model = RandomForestRegressor(100, random_state=42, oob_score=True)
_ = model.fit(X, y)
var prediction = model.predict(X_test)
var leaves = model.apply(X_test)
var unfitted = clone(model)
_ = model.set_params(warm_start=True, n_estimators=150).fit(X, y)
```

Inputs are NuMojo Float32 feature matrices. Regression targets are Float64,
with single or multiple outputs. Classification labels are Float64 by default;
select Int64 or Float32 labels with `label_dtype`, or strings with
`string_labels=True` (a `moxie._labels.StringTarget`), as for the trees:
`RandomForestClassifier[label_dtype=DType.int64](...)` or
`ExtraTreesClassifier[string_labels=True](...)`. Numeric labels must be
integral, `classes_` is sorted in the label type, and `predict` returns that
type. Members always train on Float64 class codes, as scikit-learn's forest
members do. Float64 feature matrices are not accepted (scikit-learn casts them
to float32); see the tree family's composition notes. No sparse conversion occurs.

Single-output predictions have shape `(samples,)`; probabilities and log
probabilities have shape `(samples, classes)`. Multiple regression outputs have
shape `(samples, outputs)`. For multiple classification outputs, construct
`RandomForestClassifier[multi_output=True]` or
`ExtraTreesClassifier[multi_output=True]`. Their `classes_`, `n_classes_`,
probabilities and OOB decisions are lists, one element per output. This explicit
compile-time choice replaces Python's runtime-changing return types. The default
classifier rejects a target matrix with more than one column.

Constructor parameters retain the reference names and defaults. Get the complete
owned `ForestParams` record with `get_params()`, edit its fields and pass it to
`set_params(params)`. Named partial updates support every constructor parameter.
Omitted keywords preserve their current values. Use `None` to reset `max_depth`, `max_leaf_nodes`,
`n_jobs`, `random_state` or `monotonic_cst`. Variant-valued updates use explicit
wrappers imported from `moxie.tree._classes` (`SampleCount`, `FeatureCount`,
`ClassWeight`) or `moxie.ensemble._sampling` (`MaxSamples`), for example
`set_params(min_samples_leaf=SampleCount(3), max_samples=MaxSamples(NoneType()))`.
For floating-point keywords, use float literals such as `ccp_alpha=0.0`.
Updates preserve fitted members until the next fit; validation occurs during fit.
Parameters can also be assigned directly before fitting. The common
implementation exposes `class_weight` on regressors but rejects any non-None
value.

`class_weight` accepts None, `"balanced"`, `"balanced_subsample"`, a
dictionary from labels to weights (`Dict[Float64, Float64]` by default, or keyed by
the selected label type), or a list of dictionaries for multiple outputs; the type is
`TypedClassWeight[Label]` from `moxie.tree._classes`. Dictionary keys are original
labels, not internal codes. `sample_weight` is an
optional Float64 vector passed to `fit` or `score`. Member classifiers retain
encoded class labels, as scikit-learn's forest members do.

Integer `random_state` resets the stream on each fit. None consumes Moxie's
process-wide random stream. A mutable RandomState constructor argument is not
implemented here; the underlying engine accepts one explicitly. Warm starts
retain members and replay integer-seeded draws before growing new members.
Ordinary fits replace members; cloning copies constructor configuration only.

Learned attributes include `n_features_in_`, `n_outputs_`,
`feature_importances_`, and OOB results. `estimators_()` and `estimator_()` return
borrows of the fitted members and member template. `estimators_samples_()`
regenerates sampled row indices. These methods replace Python attributes or
properties where returning a mutable borrow or computing an array is explicit.
`model[index]` supports negative indices; `len(model)` is the fitted member count.
The forest itself is not iterable (sklearn's `for tree in forest`); iterate
`model.estimators_()` instead.

`score` uses accuracy or uniform-average R2, with optional weights. Accuracy
rejects multiclass-multioutput targets, matching scikit-learn; prediction still
supports them. Unfitted prediction and inspection raise `NotFittedError`-prefixed
Mojo errors. Learned fields have empty initial values rather than Python's
attribute absence.

`n_jobs` accepts every value scikit-learn accepts except 0, which raises joblib's
`ValueError`; members are always built serially, which yields the same members
as any parallel schedule. scikit-learn's fit warnings (few bootstrap rows, rows
without OOB predictions, warm starts that add no trees) are printed to stderr as
`UserWarning: ...` lines instead of Python warnings.

Current limits: dense inputs only, `verbose=0`, no callable
OOB scorer, no `decision_path`, feature-name metadata, partial dependence, or
Python metadata routing. These are not full scikit-learn parity claims.

Run `pixi run ensemble-test` for lifecycle and contract checks, and `pixi run ensemble-parity` for
source-built scikit-learn comparisons. The public oracle checks 20 configurations
and 24 fits, including labels, multi-output results, class weights, NaNs, OOB and
warm starts. Predictions, probabilities, leaves and sampled rows must match
exactly. Importances and scores use relative tolerance 1e-12; log probabilities
allow absolute tolerance 1e-14 for libm differences. The existing engine oracle
checks additional sampling, tree parameters and shared-stream cases.
The edge oracle (`tests/public_edge_parity.py`) checks NaN routing at prediction,
`max_samples` above the row count, `log_loss`/`friedman_mse`, a single sample,
constant features and negative labels exactly, and requires invalid criteria,
`max_features`, class weights, NaN/infinite inputs, zero or negative weights and
unfitted inference to fail with the same error class as scikit-learn.
`tests/test_composition.mojo` checks clone, `make_pipeline`, `cross_val_score` and
`GridSearchCV` against manual fits.
