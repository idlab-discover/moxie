# Dense dummy estimators

Import `DummyClassifier` and `DummyRegressor` from `moxie.dummy` (also exported
from `moxie`). Both implement the family's baseline and weighted-fit traits.
They ignore feature values and do not enforce prediction feature counts.
See [scope](scope.md) for dense-only exclusions and validation coverage.

## Regression

`DummyRegressor(strategy="mean", constant=None, quantile=None)` implements mean,
median, quantile and constant prediction. `fit(X, y)` replaces learned
`constant_` and `n_outputs_`; weighted fit uses the `sample_weight` keyword.
`n_features_in_` is optional, because one-dimensional X has no feature count.
Targets may be vectors, one-column matrices, or multioutput matrices.

Both estimators accept a compile-time `dtype`, either `DType.float32` or
`DType.float64` (the default). Storage is NuMojo `NDArray[dtype]` throughout;
`Dense` remains a convenience alias for `NDArray[DType.float64]`. Feature arrays,
weights and regression targets/constants must use the selected dtype. Convert
mixed inputs explicitly with `.astype[dtype]()`. Regression learned constants
and predictions preserve that dtype. Classifier priors/probabilities and
regression standard deviations remain Float64, matching the reference.
Integer regression arrays require explicit conversion to a supported float dtype.

```mojo
var regressor = DummyRegressor[dtype=DType.float32]()
var classifier = DummyClassifier[dtype=DType.float32]()
# Use NuMojo NDArray[DType.float32] inputs with these specializations.
```

Inputs are borrowed; constructor parameter snapshots and learned state are
independent of caller mutation. Noncontiguous numerical inputs use stride-aware
NuMojo indexing, including negative strides and offsets. No Python executes
during learning or inference.

`constant` accepts a scalar `Scalar[dtype]` or an optional `NDArray[dtype]` vector with one value
per output. Scalars are normalized into a length-one vector in the typed
configuration; scalar `set_params(constant=...)` updates are also supported.
`constant_` has shape `(1, n_outputs)`. A constant whose length differs from
`n_outputs_` raises during `fit`. For single-output targets the reference instead
stores the mismatched `constant_` and fails at `predict` when broadcasting; Moxie
rejects it one step earlier and leaves the previous fitted state unchanged. `predict(X)` returns shape `(n_samples,)`
for one output, including one-column training targets, or `(n_samples, n_outputs)`
otherwise.

`predict(X, return_std=flag)` returns a fixed `DummyRegressorPrediction[dtype]` record:
`prediction` has the normal shape and `std` is an optional same-shaped Float64 zero
array. With `flag=False`, `std` is absent. This typed record replaces Python's
runtime-dependent tuple return. `score(X, y, sample_weight=None)` computes
uniform-average R2, including finite constant-target handling and NaN for fewer
than two test samples. `score(Optional[NDArray[dtype]](), y)` ignores X.

Unweighted quantiles use NumPy's linear interpolation; the unweighted median
averages the middle pair exactly as `np.median` does. Weighted quantiles use the first
sorted observation whose cumulative weight reaches the requested fraction.
Weighted median thus differs from unweighted median for an even sample count.
Negative finite weights follow the reference arithmetic; they are not interpreted
as probabilities. A zero weight total raises for a mean or weighted score;
all-zero weight arrays are rejected during validation for every strategy.

## Classification

`DummyClassifier(strategy="prior", random_state=None, constant=None)` implements
prior, most_frequent, uniform, stratified and constant. Learned names are
`classes_`, `n_classes_`, `class_prior_`, `n_outputs_`, `n_features_in_`, and
`sparse_output_` (always false after dense fitting). Classes sort increasingly;
tied class counts choose the first sorted class.

Use `Label(Int(...))`, `Label(Float64(...))`, or `Label("text")`; inspect its
`integer`, `number`, or `text` field according to `kind` (0, 1, or 2). Integers
remain exact, including values beyond Float64's exact range. Unicode strings
sort lexicographically. Each target column must use a homogeneous label kind;
perform numeric promotion explicitly. Boolean labels may be represented as
integer 0/1. Arbitrary Python object labels and object identity are binding work.

`fit` accepts `List[Label]` for one output or rectangular `List[List[Label]]`
with samples in rows. `constant` accepts a scalar integer, Float64, string or
`Label`, or an optional `List[Label]` with one value per output. Scalars normalize to a length-one list
in the typed configuration. Named scalar `set_params(constant=...)` updates
use the same conversion. A constant must use its target column's label kind:
`Label(Float64(1.0))` does not match integer label `1`, whereas NumPy's `1.0 == 1`
lets the reference accept it. Convert the constant explicitly.

`DummyClassifier()` is the single-output form: `predict` returns `List[Label]`,
`classes_` is `List[Label]`, `n_classes_` is `Int`, `class_prior_` is a one-dimensional
Float64 array, and probability methods return a Float64 array of shape
`(n_samples, n_classes)`. Both flat and one-column training targets are accepted;
multiple target columns raise a clear error without replacing learned state.

Choose `DummyClassifier[multioutput=True]()` for multiple outputs. Predictions
are sample-major `List[List[Label]]`; classes, counts, priors and probabilities
retain an outer output list. Each probability array has shape
`(n_samples, n_classes_for_output)`. This specialization retains its outer lists
even for a single target column. Selecting the layout at compile time gives
Mojo a stable result type while preserving the ordinary single-output shapes.
Cloning preserves the specialization. Both forms implement the family traits.

```mojo
var single = DummyClassifier(strategy="constant", constant="cat")
var multiple = DummyClassifier[multioutput=True](strategy="prior")
# single.predict(X)[i] is one Label; multiple.predict(X)[i][j] is output j.
```

Existing callers that expect nested results should select `multioutput=True`;
new single-output callers can remove the extra `[0]` output index.
Class priors retain zero-weight classes. Signed weights can yield invalid
probabilities, rejected by the shared multinomial sampler when sampling.

`score` accepts flat labels and rectangular targets, including with weights,
omitted X, or an explicit mutable RNG. It computes weighted
accuracy for single-output discrete labels and subset accuracy for multilabel
indicator matrices. Continuous targets and multiclass-multioutput accuracy
raise descriptive errors, as in sklearn. `score(Optional[NDArray[dtype]](), y)` omits X.

An integer seed constructs a fresh shared `RandomState` for each prediction
call, reproducing sklearn's reset behavior and output-major draw order.
For an advancing state, call `predict(X, rng)`, `predict_proba(X, rng)`,
`predict_log_proba(X, rng)`, or `score(X, y, rng, weights)` with a mutable shared
`RandomState`. This explicit argument selects the supplied stream, independent
of the stored integer seed. For `random_state=None`, operations borrow the process-wide MT19937 stream
from Moxie's `check_random_state`; repeated calls and different estimators share
its advancing state. `seed_global_random_state(seed)` from `moxie._random` resets
that stream reproducibly. It is separate from NumPy's process state. Uniform
probabilities and deterministic strategies consume no draws. The default stream
has an exclusive operation-scoped lease; concurrent or nested access raises
rather than racing. Pass the active mutable state explicitly inside a lease.

## Configuration and errors

Fits and parameter updates mutate in place. `get_params(deep=True)` returns an
independent typed parameter record; `set_params(record)` replaces configuration.
Named partial updates preserve omitted parameters; clearing optional parameters
uses an outer `Optional` or the typed record. `clone(model)` from `moxie.base`
copies constructor configuration into an unfitted estimator.
Changing classifier strategy takes effect at the next fit, matching the
reference's fitted `_strategy`; changing an integer seed affects the next call.

Errors are Mojo `Error` with descriptive prefixes. Dynamic dictionaries, pandas
feature names, metadata routing, Python serialization/reflection, and warning
objects remain binding work under the foundation contract. NaN/Inf numerical
regression targets and malformed dimensions/weights are errors. There is no
incremental-fit, transformation or sparse-target API.

Build and run `examples/dummy.mojo` through `pixi run build`, then
`builds/examples/dummy`. Tests run with `pixi run test` and `pixi run parity`.
Parity executes the pinned local Dummy classes, weighted percentile, class
counting, inherited score and metric bodies; installed sklearn only supplies supporting infrastructure.
Comparisons allow 2e-12 relative/absolute error for Float64 and 3e-6 for Float32
to accommodate rounding and summation order on the tested cases;
class labels and seeded predictions must match exactly, as must unweighted
median constants and predictions. Classifier scores are compared for every
target kind, including multilabel indicators; continuous and
multiclass-multioutput targets must raise on both sides. Log probabilities use
the standard LLVM log primitive to avoid the pinned stdlib approximation error.

## Cost model

Regression means use linear work in samples per output. Median and quantile
currently use insertion-sorted sample indices, with quadratic worst-case fit
time and linear workspace; they are correctness baselines for small problems.
Classifier fitting compares labels against each output's vocabulary; its work
scales with samples times class count. Prediction allocates the requested dense
output. No throughput advantage over sklearn is claimed. Historical benchmark
drivers remain in the standalone `moxie-dummy` repository.

## Experimental native persistence

Dummy owns `DummyRegressorSchema[dtype]` and
`DummyClassifierSchema[multioutput, dtype]` in `moxie.dummy_serialization`.
They use the shared Moxie engine in the same assembled package.
The engine and wire-format specification live in Moxie's
[serialization guide](../serialization.md).

```mojo
from moxie.serialization import save_model, load_model
from moxie.dummy_serialization import DummyRegressorSchema

save_model[DummyRegressorSchema](model, "dummy.moxie")
var restored = load_model[DummyRegressorSchema]("dummy.moxie")
var predictions = restored.predict(X)
```

Use the same compile-time dtype and multioutput parameters when loading.
Mismatches and unknown schema versions raise. Pass `SnappyCodec` from
`moxie.serialization_snappy` as a second parameter to opt into compression.
Files restore fitted state without refitting; unfitted files remain unfitted.
Clone still discards learned state. Arrays are independently owned after loading.
The schemas preserve current constructor settings separately from learned values,
including the classifier's fitted `_strategy` after `set_params`.

State schema 1 stores the regressor's dtype, strategy, optional quantile and
constructor constant, fitted flag, optional feature/output counts, and learned
`constant_`. The classifier stores dtype/multioutput, strategy, optional integer
random_state, optional constructor labels, fitted flag, feature count, fitted
strategy, output count, and each output's labels, class count and Float64 priors.
Dense fitted classifiers restore `sparse_output_=False`. Arrays are rank 1/2;
constructor arrays of other ranks are rejected by this schema. Class labels retain
integer, floating-point or UTF-8 kind and exact values; classes must be sorted,
unique and homogeneous within an output. Unknown or extra arrays are rejected.

Integer random_state retains repeated seeded prediction behavior. `None` retains
its meaning of using the destination process's global RNG; saving does not capture
that global stream. Explicit caller-owned RandomState arguments are also outside
the estimator file. For repeatable random predictions, use an integer seed or
supply the appropriate external RNG after loading.

`tests/dummy/serialization.mojo` covers both dtypes, all strategies, weighted regression,
classification, single/multioutput state, raw/Snappy files, UTF-8/NUL labels,
integers beyond Float64's exact range, parameter changes, unfitted state, clone,
repeat fitting and schema rejection. This experimental schema is not a stable
cross-release promise and does not claim pickle compatibility.
