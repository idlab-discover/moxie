# Native Naive Bayes

`moxie.naive_bayes` implements five dense CPU classifiers in pure Mojo. Their
behavior follows the local scikit-learn `1.10.dev0` reference at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

## Models and parameters

All constructor parameters are keyword-only. Constructors store configuration;
validation occurs during fitting. Each model has a corresponding typed
`<ModelName>Params` record.

| Model | Constructor defaults | Learned statistics |
| --- | --- | --- |
| `GaussianNB` | `priors=None`, `var_smoothing=1e-9` | `theta_`, `var_`, `epsilon_`, `class_prior_` |
| `MultinomialNB` | `alpha=1.0`, `alpha_values=None`, `force_alpha=True`, `fit_prior=True`, `class_prior=None` | `feature_count_`, `feature_log_prob_`, `class_log_prior_` |
| `ComplementNB` | Multinomial parameters plus `norm=False` | Same, plus `feature_all_` |
| `BernoulliNB` | `alpha=1.0`, `alpha_values=None`, `force_alpha=True`, `fit_prior=True`, `class_prior=None`, `binarize=0.0` | `feature_count_`, `feature_log_prob_`, `class_log_prior_` |
| `CategoricalNB` | `alpha=1.0`, `force_alpha=True`, `fit_prior=True`, `class_prior=None`, `min_categories=None` | `category_count_`, `n_categories_`, `feature_log_prob_`, `class_log_prior_` |

All models also expose `classes_`, `class_count_`, and `n_features_in_`.
Classes are sorted ascending, and probability columns follow that order.
Statistics and probabilities use Float64. Before fitting, learned containers are
empty and `n_features_in_` is zero; the fitted-state query returns false.

Gaussian means and population variances are calculated per class using sample
weights. Incremental updates merge the sufficient statistics. Variance
smoothing follows the pinned reference, including its use of the current
batch's unweighted variance when updating `epsilon_`.

Multinomial models normalize smoothed feature counts. Bernoulli models model
both feature presence and absence; values strictly greater than `binarize`
become one. `binarize=None` passes input values through, so supply binary data.
Complement models use counts from the other classes and optionally normalize
their feature weights; like sklearn, their multi-class decision scores do not
add class priors. Categorical models allocate separate category counts for
each feature, growing those buffers during incremental fitting.

## Input and output types

The preserved List interface uses these types; array extensions follow below.

- `X`: rectangular, nonempty `List[List[Float64]]`, samples by features.
- `y` and `classes`: `List[Int]`, one class label per sample or class.
- `sample_weight`: `List[Float64]`, one weight per sample; fit and partial_fit
  also accept a Float64 scalar broadcast to every sample.
- `predict`: `List[Int]`, one predicted class per input row.
- Probability and joint-log-probability methods: `List[List[Float64]]`,
  samples by classes.
- Priors: `Optional[List[Float64]]`, in sorted class order.
- `alpha`: scalar Float64; `min_categories`: optional scalar Int. CategoricalNB
  also accepts `min_categories_values: Optional[List[Int]]` for per-feature minima.

`GaussianNB` also accepts boolean, integer and Float32/Float64 NDArrays, with
numeric NDArray and StringTarget labels through an explicit compile-time domain.
Its explicit `dtype` parameter selects Float32 or Float64 (default) statistics
and arithmetic. See the [Gaussian API](gaussian/api.md) for
signatures, shapes and Float32 examples.

`MultinomialNB` and `ComplementNB` (implemented in `moxie._nb_counts`) also
accept boolean, integer and Float32/Float64 `numojo.NDArray` matrices, including
strided views, for
`fit`, `partial_fit`, prediction and `score`. sklearn's array-valued `alpha` is
the native keyword `alpha_values`, an owned `(n_features,)` Float64 NDArray
that takes precedence over scalar `alpha`; `set_params(alpha=...)` clears it.
Their `class_count_` `(C,)`, `feature_count_` and `feature_log_prob_` `(C,F)`
and ComplementNB's `feature_all_` `(F,)` are Float64 NDArrays, read with
`.item(...)`; `class_log_prior_` and probabilities remain Lists.

`BernoulliNB` (implemented in `moxie._nb_bernoulli`) accepts boolean, integer
and Float32/Float64 NDArrays for all fit, partial_fit, prediction and score methods. Inputs can have
valid strides and offsets. Array prediction probabilities are Float64 NDArrays
of shape `(n_samples, n_classes)`; List inputs retain List probability outputs.
The default target domain uses List[Int]; fit and partial_fit accept a Float64
scalar or List[Float64] weights. Pass `sample_weight=weights.copy()` for a List
on array overloads.

Bernoulli learned `class_count_` `(C,)`, `feature_count_` `(C,F)`, and
`feature_log_prob_` `(C,F)` are now Float64 NDArrays for both input interfaces;
use `.item(c)` or `.item(c, f)` instead of nested List indexing. `class_log_prior_`
remains a List. The constructor and `BernoulliNBParams` also accept
`alpha_values`, an owned rank-one Float64 NDArray of length `n_features`.
It takes precedence over scalar `alpha`; `set_params(alpha=...)` clears it.
Clone, copy and parameter records own independent vector storage. The original
five positional `BernoulliNBParams` arguments remain valid; the optional vector
is appended. Construction can raise on allocation.

Vector smoothing uses `(feature_count_[c,f] + alpha_values[f]) /
(class_count_[c] + 2 * alpha_values[f])`, including when class and feature counts
differ. This intentionally corrects a bug in the pinned scikit-learn revision:
its vector path adds `(n_classes,)` counts to `(n_features,)` alpha before
reshaping, causing broadcast errors or class-indexed denominators. Vector
fixtures therefore compare against separate scalar-alpha scikit-learn fits for
each feature, then scikit-learn's joint likelihood and normalization. Scalar
behavior remains reference-compatible.

Counts widen Float32 input to Float64 after thresholding. Floating threshold
comparisons use the floating input dtype, matching
NumPy's scalar comparison semantics. `binarize=None` preserves input values;
no extra binary-value restriction is imposed. No caller array aliases are kept.

```mojo
import numojo as nm
from numojo import Shape, Item
from moxie.naive_bayes import BernoulliNB

var X = nm.zeros[DType.float32](Shape(2, 2))
X[Item(0, 0)] = 1.0
X[Item(1, 1)] = 1.0
var smoothing = nm.zeros[DType.float64](Shape(2))
smoothing[Item(0)] = 0.25
smoothing[Item(1)] = 2.0
var model = BernoulliNB(alpha_values=smoothing^)
model.fit(X, List[Int]([1, 2]))
var probabilities = model.predict_proba(X)
print(probabilities.item(0, 0))
```

Multinomial and complement training inputs must be nonnegative. Categorical
inputs are integer, boolean or finite Float32/Float64 values. Floating values
truncate toward zero before checking nonnegativity, matching sklearn (so -0.9
becomes category zero). Converted values must fit an Int; each
feature uses category codes starting at zero. A category can be predicted if
its slot was allocated during fitting, including through `min_categories`.
An out-of-range category raises a descriptive error.

`theta_`, `var_`, and discrete feature counts are classes by features.
Categorical `category_count_` and `feature_log_prob_` are `List[NDArray[float64]]`:
one `(n_classes, allocated_categories)` array per feature. Read a count with
`model.category_count_[feature].item(class_index, category_index)`. As in the pinned source, categorical
`n_categories_` describes the latest batch's inferred counts (including the
configured minimum); previously allocated category buffers do not shrink.

## Categorical arrays and typed labels

`CategoricalNB` accepts boolean, integer and Float32/Float64 NDArray features
for fitting, incremental
updates, prediction and scoring, including strided views. List features remain
supported. Its `class_count_` is a Float64 NDArray `(n_classes,)`; category counts
and feature log probabilities use one independently owned Float64 array per
feature. `n_categories_` and `class_log_prior_` remain Lists. Array probability
outputs are Float64 NDArrays `(n_samples, n_classes)`; List features produce List
probabilities. Construction can raise on allocation.

All five classifiers accept an optional compile-time class domain, using the
same shared Moxie label implementation:

| Domain | `y`, `classes`, `classes_`, and `predict` result |
| --- | --- |
| Default (`IntListDomain`) | `List[Int]` |
| `NumericDomain[DType.int64]` | Rank-one Int64 NDArray, preserving large integer identity |
| `NumericDomain[DType.float32]` or `NumericDomain[DType.float64]` | Rank-one NDArray of the selected dtype; finite integral labels only |
| `StringDomain` | Rank-one `moxie._labels.StringTarget` |

```mojo
from moxie.naive_bayes import CategoricalNB, StringDomain
from moxie._labels import StringTarget

var X: List[List[Float64]] = [[0.0, 1.0], [1.0, 0.0]]
var y = StringTarget(["cat", "dog"])
var model = CategoricalNB[StringDomain]()
model.fit(X, y)
print(model.predict(X).get(0))
```

Use `ModelName()` for default List[Int] targets and `ModelName[]` in concrete
type annotations. Label domains are constructor type parameters, not fit-time
dtype inference. This native interface difference is checked by contract tests.
Targets must be single-output and match the feature row count; continuous,
nonfinite and rank-two targets raise. Strings sort lexicographically without
Unicode normalization. First-call classes may contain absent labels; unknown
training labels raise, while unknown score labels count as incorrect.
Inputs and class domains are not retained as mutable caller aliases. Copying a
fitted estimator copies its domain and numerical state; cloning is unfitted.

Feature codes for CategoricalNB are already encoded nonnegative indices, not
raw string columns or Moxie `CategoricalColumn` objects. Fractional floating
feature values truncate toward zero, as sklearn's `dtype="int"` validation does
(see the behavior table below). Missing categories are not accepted.

## Fitting and prediction

All five implement `NaiveBayesClassifier` and `WeightedSupervisedEstimator`:

```mojo
var model = MultinomialNB()
model.fit(X, y)
model.fit(X, y, sample_weight=weights)
var labels = model.predict(X)
var probabilities = model.predict_proba(X)
var log_probabilities = model.predict_log_proba(X)
var joint = model.predict_joint_log_proba(X)
var accuracy = model.score(X, y)
var weighted_accuracy = model.score(X, y, sample_weight=weights.copy())
```

Score is weighted or unweighted classification accuracy. Inference rejects
unfitted state, nonfinite input, and changed feature counts. Probabilities use
log-sum-exp normalization to avoid exponentiating large joint likelihoods.

The first `partial_fit` requires the complete class set, including classes
absent from that batch. Later calls preserve it; supplied class lists must
match after sorting and deduplication. Unknown target labels raise.

```mojo
var model = GaussianNB()
var classes: List[Int] = [0, 1]
model.partial_fit(X_first, y_first, classes=classes^)
model.partial_fit(X_next, y_next, sample_weight=weights)
```

A normal `fit` replaces learned state. Failed fit/partial-fit calls preserve
the previous fitted model. Incremental updates copy existing state before
committing, favoring clear ownership over minimal allocation in this first
implementation.

## Integer/boolean features and scalar weights

All five models accept `NDArray` features with dtype `bool`, signed or unsigned
8-, 16-, 32- or 64-bit integers, `float32`, or `float64`, including valid strided
and offset views. Input storage is preserved; statistics and arithmetic widen
to Float64. Large integer values can therefore round during floating arithmetic.
Float16, BFloat16 and complex features are outside this contract. Targets retain
the separate label-domain restrictions above. Integer/bool Bernoulli features
are promoted to Float64 for comparison with floating `binarize` thresholds.

Both List and array `fit`/`partial_fit` overloads accept a Float64 scalar:

```mojo
model.fit(X, y, sample_weight=2.5)
model.partial_fit(X_next, y_next, sample_weight=0.5)
```

This broadcasts one weight per row through the existing validated weight path.
Scalar weights must be positive and finite, and the batch total must be finite.
Invalid weights leave learned state unchanged. As elsewhere in the native API,
convert numeric variables of other types explicitly with `Float64(weight)`.
`score` still takes optional List weights, matching the reference metric's
vector-weight interface. The generic weighted-estimator trait continues to use
`List[Float64]`; scalar fitting is an additional concrete overload.

## Configuration and ownership

`get_params(deep=True)` returns an independently owned parameter record;
`set_params(params)` replaces constructor configuration. Named partial updates
are also available, for example `model.set_params(alpha=0.5)`. Parameters are
not automatically refitted. Use a new fit after changing a training parameter.

For optional parameters, named updates use nested `Optional`: the outer
`None` leaves the parameter unchanged, while an explicitly present inner
`None` clears it. Editing a typed record is often simpler:

```mojo
var params = model.get_params()
params.class_prior = None
model.set_params(params)
```

`clone(model)` returns an unfitted model with independent configuration;
`model.copy()` copies fitted statistics as well. Optional list arguments
require explicit ownership in Mojo, e.g. `priors=prior.copy()` or
`classes=classes^`. Fits, incremental updates and parameter updates return a mutable reference
to the same estimator, following the [shared native contract](../interfaces.md).

## Composition with other Moxie families

- `clone`, fitted `copy()` and `get_params`/`set_params` follow the shared
  contract above.
- Typed pipelines: each estimator's trait `Input` is `List[List[Float64]]`; the
  `NDArray` methods are extra overloads, not the trait type. A
  `moxie.pipeline` step that outputs `NDArray` (for example `StandardScaler`)
  therefore cannot precede a Naive Bayes classifier:
  `make_pipeline(StandardScaler(), GaussianNB())` fails at compile time (the
  pipeline's `rebind` of the scaler output to `GaussianNB.Input` fails).
  Transform first and pass the array to the classifier's `NDArray` overloads.
- `moxie.model_selection.learning_curve` (like the other CV helpers built on
  the same trait) takes an explicit `CVAdapter` / `TypedScorer`; `GaussianNB` with
  `exploit_incremental_learning=True` drives `partial_fit` through an
  `IncrementalCVAdapter`. See `notebooks/naive_bayes/plot_learning_curve.ipynb`.
- `moxie.calibration.CalibratedClassifierCV` takes a `CalibrationEstimator`
  adapter; no adapter ships for these classifiers yet. A forwarding adapter
  is about 40 lines (see `notebooks/naive_bayes/plot_calibration_curve.ipynb`).

## Supported scope and numerical behavior

This is a dense implementation of all five algorithms, not completion of the
whole-family compatibility mission. Unsupported native inputs are sparse
matrices and multioutput targets. CategoricalNB accepts only scalar `alpha`,
as in the reference (its `_parameter_constraints` reject array-like alpha).
All five classifiers support numeric NDArray and string target domains, and
boolean, integer and Float32/Float64 dense feature arrays.
Feature names, Python adapters and dynamic metadata routing remain outside the
shared native contract.

Weights must be finite, nonnegative, and have positive finite total weight in
each batch. Negative weights raise even where sklearn accepts them. All-zero
rejection matches the pinned reference. Explicit priors must be finite and nonnegative; Gaussian priors must also sum to one. Unknown
incremental labels raise consistently for every model.

`force_alpha=False` floors each scalar or vector alpha entry to `1e-10`; the
floor is applied without emitting a Python-style warning. `force_alpha=True` preserves zero alpha.
Degenerate likelihoods can consequently produce infinities or NaNs, just as
in sklearn; these are not replaced by fabricated uniform probabilities.
Normalization follows `scipy.special.logsumexp`: a row whose maximum is
`+inf` yields NaN at that entry and `-inf` elsewhere.

The differential checks use `atol=1e-7`, `rtol=1e-7` for floating results,
exact class predictions and shapes, and explicit NaN/infinity checks.
Logarithms and exponentials use Numax `core.libm` (fdlibm transcriptions, at
most one ulp from glibc), not Mojo 1.0's `std.math` versions, whose Float64
`log` was measured up to ~2e7 ulp off and flipped a ComplementNB tie. The
tolerance covers remaining one-ulp libm differences and summation order
(sklearn's dot products use OpenBLAS). Bitwise equality with sklearn is not
promised; near-tied predictions at extreme scales may still differ.

## Validation and references

`pixi run test` includes lifecycle, cloning, failed-update, input-validation,
and simple count/probability checks. The separate differential runner compares
learned statistics, predictions, all three probability methods, and weighted
scores against the pinned local sklearn build. It exercises weighted and
incremental fits, priors, absent classes, single-class fits, smoothing,
Bernoulli thresholds, complement normalization, and categorical growth.
See the [README](README.md) for reproducible commands.

Primary local references are `../scikit-learn/sklearn/naive_bayes.py`,
`../scikit-learn/sklearn/tests/test_naive_bayes.py`, and
`../scikit-learn/doc/modules/naive_bayes.rst`. Mathematical references are the
Gaussian weighted variance merge by Chan, Golub and LeVeque; the multinomial
and Bernoulli models in Manning, Raghavan and Schütze; and complement weights
in Rennie et al. (2003), all cited and described by the pinned source/guide.
Upstream attribution is retained in [LICENSES](../../LICENSES/naive-bayes-scikit-learn.txt).

## Extending this family

Shared ownership and NuMojo extension layouts are in
[plumbing.md](plumbing.md). The [dense scope](scope.md) summarizes supported and deferred capabilities. Public imports remain unchanged;
shared helpers live in `_naive_bayes_common`, and preserved implementations
are reached through the `naive_bayes` facade.

## Per-feature categorical minima

Use `CategoricalNB(min_categories_values=List[Int]([3, 4]))` to reserve at least
three categories for the first feature and four for the second. Observed category
indices can grow these limits. Smoothing includes reserved categories, so the
model can predict a category absent from training if its slot was allocated.

The separate `min_categories_values` keyword is the typed Mojo equivalent of
passing an integer vector to scikit-learn's `min_categories`. It accepts one
nonnegative integer per feature or a singleton that broadcasts across all features;
zero infers the observed size.
Negative vector entries are rejected, even though the reference may accept them.
The preserved scalar `min_categories` argument requires a positive integer.

Vectors take precedence when both forms are supplied. A named scalar update
clears the vector; a vector supplied in the same update takes precedence.
`set_params(min_categories_values=Optional[Optional[List[Int]]](Optional[List[Int]](None)))`
clears the vector and restores the stored scalar (or inference when it is None).
`get_params`, `set_params(params)`, clone, and fitted copy own independent vectors.
Construction accepts an owned List; pass `.copy()` to retain your own copy.
Validation happens during fit or partial_fit; failed updates preserve learned state.

Each partial_fit reports the current batch's inferred `n_categories_`, bounded
below by the configured minima. Previously allocated count buffers never shrink
during partial_fit. Ordinary fit starts fresh, so smaller minima can shrink them.
Native lifecycle tests cover these contracts; the differential fixtures compare
weighted counts, smoothing, predictions and incremental growth with the pinned
scikit-learn checkout using the existing 1e-7 absolute/relative likelihood tolerance.
Counts and category dimensions are also checked exactly in the native tests.


## Behavioral compatibility decisions

These classifications refer to sklearn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
A retained deviation is not strict reference parity.

| Behavior | Classification | Regression evidence |
| --- | --- | --- |
| Negative sample weights | Retained native deviation: reject even when sklearn accepts positive class totals; protects nonnegative sufficient statistics | `tests/naive_bayes/check_sklearn.py` checks sklearn acceptance and native fit/update rejection with preserved counts and predictions |
| All-zero-weight batches | Reference-compatible rejection for all five estimators; the pinned `_check_sample_weight` rejects them | Same differential rejection cases; `tests/naive_bayes/test_input_compat.mojo` covers scalar zero and rollback |
| Categorical fractional features | Reference-compatible truncation toward zero in fit, partial_fit and prediction, including values between -1 and 0 | Differential fractional List/Float32/Float64 cases and `test_categorical_arrays.mojo` |
| Singleton `min_categories_values` | Reference-compatible broadcasting | Differential weighted/category-growth cases and native categorical contracts |
| Bernoulli vector alpha | Retained native deviation: smooth each feature with its own alpha; do not reproduce the pinned denominator broadcasting bug | Differential square/rectangular bug probes and independent scalar-sklearn fits, checked after each update |
| Negative discrete `class_prior` entries | Retained native deviation: raise; the pinned reference passes them to `np.log` and stores NaN log priors | `tests/naive_bayes/test_reference_edges.mojo` |
| Unknown labels in a later `partial_fit` (discrete models) | Retained native deviation: raise and keep state; the pinned `label_binarize` silently drops such rows from the counts (GaussianNB raises in both) | `tests/naive_bayes/test_reference_edges.mojo` |
| Gaussian Float32 input with default estimator | Retained native deviation: a fixed compile-time state/output dtype defaults to Float64 across calls; choose `GaussianNB[dtype=DType.float32]` for Float32 reference behavior | `tests/naive_bayes/gaussian/contracts.mojo` distinguishes the two variances on large-offset input; typed differential fixtures check both arithmetic dtypes; default-estimator Float32 fixtures use a Float64 sklearn oracle after preserving input rounding |

Run `pixi run build`, `pixi run test`, then
`builds/naive-bayes-reference-env/bin/python tests/naive_bayes/check_sklearn.py` and
`builds/naive-bayes-reference-env/bin/python tests/naive_bayes/gaussian/parity.py`.
See the setup instructions in the differential driver for installing the exact
local sklearn source as a tooling-only oracle.

The differential fixtures compare learned state after every incremental batch,
class order and exact predicted labels, and all three probability outputs.
NaNs must match NaNs; infinities must match sign; neither is accepted by a finite
tolerance. Float64 checks use `1e-7 * (1 + abs(expected))`, allowing accumulated
reduction/libm error; typed Gaussian Float32 checks use `3e-5` by the same rule
for same-dtype reductions. These are numerical tolerances, never label tolerances.
For default Gaussian estimators with Float32 storage, the family oracle widens
already-rounded input values to Float64, explicitly testing the retained native
arithmetic contract. It does not claim direct Float32 reference parity. Near
singular models can have visibly different probabilities, not just different
last bits; the Gaussian contracts pin a 0.5-versus-1.0 normalization example. The
separate typed Gaussian suite compares each selected arithmetic dtype directly.
Use `tests/naive_bayes/check_sklearn.py --start N --stop M` to reproduce the half-open case
range printed by a run without compiling unrelated cases.
Fixed seeds cover randomized row/feature counts, rare classes arriving late,
singleton and uneven partitions, large weight ratios, extreme feature scales,
and exact/near ties. This bounded campaign does not establish exhaustive parity.
