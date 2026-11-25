# Calibration

The implementation follows `sklearn.calibration` at revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Library code is pure Mojo.
Python, NumPy, SciPy and scikit-learn are used only for validation.

## Using the estimator

```mojo
from moxie.calibration import CalibratedClassifierCV, adapt_classifier
from moxie.naive_bayes import GaussianNB

# X and y are NuMojo arrays.
var model = CalibratedClassifierCV(
    adapt_classifier(GaussianNB()), method="sigmoid", cv=5
)
model.fit(X, y)
var probabilities = model.predict_proba(X_test)
var labels = model.predict(X_test)
```

`X` has shape `(n_samples, n_features)`; `y` and optional weights have shape
`(n_samples,)`. The concrete `fit`, `predict_proba`, `predict` and `score`
methods accept Float32 and Float64 NuMojo feature matrices. Each Float32 call
materializes one Float64 feature matrix before invoking the estimator or its
adapters. Contiguous inputs use NuMojo's `astype`, and strided inputs use
coordinate indexing to preserve their logical order. The associated `Input`
type used by generic estimators and displays is Float64. Targets, weights,
learned classes, predicted labels and probabilities also use Float64 arrays.
The estimator's `fit` and `score` accept strided target views and copy those
views into contiguous arrays in logical sample order.
Inputs must be finite and nonempty. Class labels are integer-valued numbers
with magnitude at most `2**53 - 1`. Weights must be nonnegative with positive
finite sum. Learned arrays do not alias input arrays.

`classes_` is sorted. `predict_proba` returns `(n_samples, n_classes)` with
columns in that order. `predict` breaks ties in favor of the first class.
`score` computes weighted accuracy. Normal fitting replaces learned state;
a failed fit preserves the previous successful fit. `clone` clones the base
estimator configuration and returns an unfitted calibrator. A frozen base
estimator's clone retains its own fitted state.

Constructor parameters are `estimator`, `method="sigmoid"`, `cv=None`,
`n_jobs=None`, and `ensemble="auto"`. Omitting `estimator` or passing
`estimator=None` constructs native `LinearSVC(random_state=0)` through the
built-in adapter. The typed `estimator` field and parameter record expose the
unfitted adapter immediately. Its fitted copies are in `calibrated_classifiers_`.
Pass `adapt_classifier(LinearSVC(...))` to customize the base classifier.
`cv=None` means five stratified folds. Explicit integers of at least two require each
class to have at least `cv` samples, as in sklearn. With `cv=None`, smaller
minority classes are allowed when the largest class has at least five samples.
The reference warns for that case; the native implementation emits no warning.
Pass a native `moxie.model_selection.KFold` or `StratifiedKFold` directly as
`cv`. The model stores the splitter configuration and generates folds from the
current training data at each fit. As in sklearn, each class must have at least
`n_splits` samples, and `LeaveOneOut` raises at fit time. Shuffled splitters
require an integer `random_state` because calibration uses the splitters' native
seeded entry point. The same seed reproduces the folds on repeated fits.

`cv` also accepts `List[CalibrationSplit]`, where each split owns `train` and
`test` index lists. Use this representation for externally generated folds.
Index order and repeated training indices are preserved. Negative and
out-of-range indices, empty folds and empty split lists raise errors.
With `ensemble=False`, test indices must include every sample exactly once.
With `ensemble=True`, repeated holdouts and incomplete test coverage are allowed. `ensemble="auto"` means true except
for a frozen adapter. True fits one classifier/calibrator pair per fold and
averages probabilities. False fits one calibrator on out-of-fold responses
and refits the base classifier on all samples. All computation is serial;
`n_jobs` accepts only None or 1.

`calibrated_classifiers_` stores fitted pairs. The pairs expose their estimator,
classes, method and native sigmoid/isotonic/temperature models. Sigmoid models
have `a_` and `b_`; temperature models have `beta_`. These private model types
are implementation details, not additional public sklearn estimators.

## Classifier adapter

Use `adapt_classifier(estimator)` with the default type specializations of
`GaussianNB`, `MultinomialNB`, `ComplementNB`, `BernoulliNB`, `CategoricalNB`,
`LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis`,
`LogisticRegression` and `SVC`. For `LinearSVC`, use any of `Int`, `Int64`,
`Float32` or `Float64` as the label type. Pass the result to
`CalibratedClassifierCV`.
Adapters with `predict_proba` also support direct use with
`CalibrationDisplay.from_estimator` after fitting. The adapter prefers
`decision_function` for calibration and uses `predict_proba` for displays.
With Naive Bayes it uses probabilities for both. The adapter converts Float64
calibration targets to the classifier's native numeric list or Int64 array and
returns its fitted classes in column order. Integral labels must be exactly
representable in the component's target type. For example,
`adapt_classifier(LinearSVC[Float32]())` rejects 16777217 because Float32
rounds it to 16777216. It forwards explicit weights to
all five Naive Bayes classifiers, LogisticRegression and LinearSVC. The LDA,
QDA and SVC adapters reject explicit weights. Standalone SVC calibration expects
feature input with a built-in kernel. Multiclass SVC requires one-vs-rest
decision scores, the default `decision_function_shape="ovr"`.

For a Float64 pipeline ending in `SVC[]`, pass the entire pipeline to
`adapt_classifier`. Each calibration fold fits its own preprocessing and
SVC. The adapter reads class order from that fold's fitted SVC and uses
decision scores, so the SVC can keep its default probability setting.
For multiclass calibration, use the default one-vs-rest decision shape.
This adapter rejects explicit fit weights because SVC does not implement
the shared weighted-fit trait.

```mojo
from moxie.calibration import CalibratedClassifierCV, adapt_classifier
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler
from moxie.svm import SVC

var calibrated = CalibratedClassifierCV(
    adapt_classifier(make_pipeline(StandardScaler(), SVC())), cv=3,
)
```

The [Pipeline parity test](../../tests/pipeline/calibrated_pipeline_parity.py)
compares held-out probabilities, predictions and class order after refitting
with pinned scikit-learn for both ensemble and pooled sigmoid calibration.
The probability tolerance allows rounding in scaling, SVC and sigmoid fitting.

`ClassifierCalibrationAdapter(estimator, read_classes, name)` also supports
custom classifiers with these input and target types and compatible response
traits. Supply a thin Mojo function that returns the fitted class labels as
a Float64 vector. `estimator()` borrows the fitted classifier for inspection.
Copies share a read-only snapshot, and fitting replaces the snapshot on
success. For classifiers without fitted copying, such as QDA, updating adapter
parameters creates an unfitted snapshot. Ordinary clones are unfitted.
For all five fitted default Naive Bayes specializations,
`LinearDiscriminantAnalysis[]`, `QuadraticDiscriminantAnalysis[]` and
`LogisticRegression[]`, pass the `FrozenEstimator` handle directly to
`adapt_classifier`.

```mojo
var calibrated = CalibratedClassifierCV(adapt_classifier(frozen.copy()))
_ = calibrated.fit(X_calibration, y_calibration)
```

The calibration targets use a Float64 array. Copying the frozen handle shares
the fitted estimator. Cloning the adapter preserves that handle, fitting keeps
the base estimator frozen, and `ensemble="auto"` fits one pooled calibrator.
Explicit sample weights affect only the calibration fit for a frozen model.

`CalibrationEstimator` extends the shared configuration/clone contract. It
requires `fit(X, y, sample_weight=None)`, `class_labels()`, `response(X)`, and
`uses_probabilities()`. Its `is_frozen()` defaults to false.

- `class_labels()` returns sorted numeric labels after fitting. A fold may expose
  a sorted subset of the global classes. Ensemble sigmoid/isotonic calibrators
  map these columns back to `classes_`; unobserved classes receive zero before
  normalization. Pooled out-of-fold probability responses insert zero columns
  for missing classes. Pooled decision responses use the reference's minimum
  finite score and reject incomplete folds with only one or two classes.
  Temperature ensemble folds still require all training classes.
- `response()` returns the preferred decision response when available,
  otherwise probabilities. Binary decisions have shape `(n_samples, 1)`.
  Multiclass decisions and probability responses have one column per class.
- Probability responses must be in `[0, 1]` and row sums must be within `1e-8`
  of one. The wrapper selects the positive-class column for binary sigmoid
  and isotonic fitting.
- Adapters without weighted fitting must reject nonempty weights explicitly.
  The wrapper passes None when the caller did not provide weights.
- A frozen adapter must preserve fitted state in its clone and make fit a
  no-op. Training and calibration datasets should be disjoint, as with sklearn.

See [Centroid](../../examples/calibration_support/calibration_centroid.mojo) for a complete native adapter that
learns weighted class centroids. It is an example classifier, not the default
classifier or a substitute for another family's estimator.

`get_params(deep=True)` returns `CalibratedClassifierParams[E]`, including the
base estimator. Its nested typed parameters remain accessible through
`params.estimator.get_params()`. No flattened `estimator__name` mapping is
provided. Change the record and call `set_params(params)`; refit to use the new
configuration. The keyword overload `set_params(estimator=..., method=...,
cv=..., n_jobs=..., ensemble=...)` updates only the named parameters; resetting
`n_jobs` to None requires the full record. Wrap a splitter as
`CalibrationCV(splitter)` when using the optional `cv` keyword in `set_params`.
`fit` and `set_params` return a mutable borrow of the calibrator, following
Moxie's fluent estimator contract. The family adapter's optional-weight `fit`
method remains a no-result operation.

## Fitting methods

Sigmoid uses Platt's smoothed targets and weighted logistic loss, including
score rescaling at absolute values of 30 or greater. A two-parameter Newton
method with backtracking solves this convex objective. Optimization failures
raise an error. This differs from sklearn's L-BFGS optimizer, so coefficients
need not match bit for bit.

Isotonic uses weighted pool-adjacent-violators fitting after sorting and merging
duplicate scores. Predictions use linear interpolation and clip outside the
training range. Scores separated by less than `1e-15` merge, matching the
reference's Float64 resolution rule. Calibration converts its Float64 arrays to Lists and delegates to Moxie's
`IsotonicRegression[DType.float64](out_of_bounds="clip")`. That owner removes
redundant knots without changing the interpolated function. Calibration retains
its stricter rejection of negative weights and nonfinite weighted results.
The sigmoid and temperature optimizers remain unchanged.

Temperature fits the inverse temperature over log-beta in `[-10, 10]` using
bounded golden-section minimization of weighted multinomial log loss.
Binary decisions become `(-score, score)`. Inputs that look like probabilities
become `log(p + 1e-12)`, following the reference. Stable softmax produces the
probabilities. The optimizer differs from SciPy's bounded Brent search.

Sigmoid and temperature calibration use Numax's Float64 scalar `exp` and
`log`. The sigmoid objective uses `std.math.log1p` to preserve small loss
terms. Numax's array `log1p` calls the same stdlib primitive and would
materialize an array for this scalar operation.

The pinned sklearn temperature wrapper does not encode nonconsecutive numeric
labels before its multinomial loss. This implementation always encodes labels
as consecutive class indices and maps predictions back through `classes_`.
Temperature wrapper parity therefore uses explicitly encoded reference labels
and maps the reference predictions back. This is a deliberate correction of
that reference defect, covered by the nonconsecutive-label cases.

Multiclass sigmoid/isotonic fits one calibrator per class, then normalizes.
If all return zero, the output is uniform. Binary outputs use the positive
probability and its complement. As in the reference, each fold's
probabilities in `(1, 1 + 1e-5]` are set to exactly one before averaging.

## Calibration curves

`calibration_curve(y_true, y_prob, pos_label=None, n_bins=5, strategy="uniform")`
returns a tuple `(prob_true, prob_pred)` of one-dimensional Float64 arrays.
Only occupied bins are returned. Interior boundaries belong to the lower bin.
Uniform edges are `j * (1 / n_bins)` with an exact final edge, as computed by
`numpy.linspace`, so probabilities one ulp around an edge fall in the same bin
as in the reference. Quantile edges reproduce `numpy.percentile`'s linear
method, including its `q * 100 / 100` rounding and its interpolation from the
upper neighbour when the fractional position is at least one half. `n_bins` accepts a positive
integer or `"cube_root"`. Numeric labels outside the usual `{0, 1}` or `{-1, 1}`
sets require `pos_label`. Float32 inputs are widened exactly before binning;
output statistics remain Float64, matching the reference's weighted bin counts.
String labels use `List[String]` and require an explicit string `pos_label`.
Displays and estimator targets use Float64 arrays.

## Calibration displays

`CalibrationDisplay(prob_true, prob_pred, y_prob, estimator_name=None,
pos_label=None)` owns independent copies of its input arrays. `line_`, `ax_`
and `figure_` are optional native records, initially unset. `plot()` populates
them. `from_predictions(...)` computes the curve, retains `y_prob`, validates
binary labels and plots. `from_estimator(...)` selects the requested class's
probability column, defaulting to the second class, then follows the same path.
Names, axis labels and the default square markers follow the reference.

`CalibrationDisplayEstimator` supplies `class_labels()`,
`calibration_probabilities(X)` and `display_name()`. This explicit adapter
replaces Python's reflection on class names and `predict_proba` availability.
`CalibratedClassifierCV` and the example centroid classifier implement it.
The probability accessor is separate from the base classifier's preferred
calibration response, which may be a decision score.

```mojo
from moxie.calibration import CalibrationDisplay, CalibrationAxes

var axes = CalibrationAxes()
var display = CalibrationDisplay.from_estimator(model, X_test, y_test, ax=axes)
display.figure_.value().savefig("builds/reliability.svg")
```

Pass the same mutable `CalibrationAxes` to multiple `plot`, `from_predictions`
or `from_estimator` calls to overlay curves. The reference line appears only
once. Set `ref_line=False` to omit it on new axes. The supported style keywords
are `color`, `alpha`, `marker` (`"s"`, `"o"`, or empty), and `linestyle`
(`"-"`, `"--"`, `":"`, or empty). A `name` passed to plot changes its legend
entry without changing the constructor's `estimator_name`.

`CalibrationLine` stores line coordinates and style. `CalibrationAxes` stores
lines, labels and reference-line state. `CalibrationFigure.savefig` writes
SVG using Mojo only. Display attributes are owned snapshots of the axes at
the time of that plot; later changes to another display do not mutate an older
figure. This ownership model replaces Matplotlib's shared mutable artists.
`plot` requires finite, nonempty coordinate vectors, mutates in place and
returns nothing. Matplotlib axes, arbitrary plot
kwargs, interactive windows and other image formats are not supported.

## Native model files

Choose the schema from `moxie.calibration_serialization` that matches the
calibrator's base estimator. Each schema accepts fitted and unfitted models.

| Base estimator | Schema |
| --- | --- |
| Default LinearSVC or `adapt_classifier(LinearSVC(...))` | `CalibratedClassifierCVSchema` |
| `adapt_classifier(SVC(...))` | `CalibratedSVCClassifierCVSchema` |
| `adapt_classifier(GaussianNB(...))` | `CalibratedGaussianNBClassifierCVSchema` |
| `adapt_classifier(MultinomialNB(...))` | `CalibratedMultinomialNBClassifierCVSchema` |
| `adapt_classifier(ComplementNB(...))` | `CalibratedComplementNBClassifierCVSchema` |
| `adapt_classifier(BernoulliNB(...))` | `CalibratedBernoulliNBClassifierCVSchema` |
| `adapt_classifier(CategoricalNB(...))` | `CalibratedCategoricalNBClassifierCVSchema` |
| `adapt_classifier(LinearDiscriminantAnalysis(...))` | `CalibratedLDAClassifierCVSchema` |
| `adapt_classifier(QuadraticDiscriminantAnalysis(...))` | `CalibratedQDAClassifierCVSchema` |

```mojo
from moxie.calibration import CalibratedClassifierCV
from moxie.calibration_serialization import CalibratedClassifierCVSchema
from moxie.serialization import load_model, save_model

var model = CalibratedClassifierCV(method="isotonic", cv=3)
_ = model.fit(X, y)
save_model[CalibratedClassifierCVSchema](model, "calibrated.moxie")
var restored = load_model[CalibratedClassifierCVSchema]("calibrated.moxie")
var probabilities = restored.predict_proba(X)
```

The experimental schema stores current constructor settings, learned class
order and feature count, and every fitted estimator and calibrator for the
sigmoid, isotonic and temperature methods. Loading reconstructs independent
fitted state without refitting. Parameters changed after fit apply to the next
fit. Each schema requires the matching built-in adapter with its default
feature dtype and label type because custom class readers have no
representation in the file. The LDA and QDA schemas require
`covariance_estimator=None`. For CategoricalNB, use `min_categories` for a
scalar minimum and `min_categories_values=List[Int]([4, 3, 5])` for
scikit-learn's per-feature minimum. Both settings are preserved in the
archive. See [native model files](../serialization.md) for format and load
limits.

## Limits and validation

Remaining gaps include sparse estimators, typed estimator labels, arbitrary fit
metadata, parallel jobs, metadata routing, persistence for other base
estimators, feature names and Matplotlib object interoperability. Temperature
ensemble folds with missing training classes are rejected explicitly.

From the Moxie root run `pixi run build`, `pixi run test`, and `pixi run parity`.
For calibration alone run `pixi run python tools/calibration_checks.py test`
and `pixi run python tools/calibration_checks.py parity`. Run the built example
with `builds/examples/calibration`.
Parity loads calibration and isotonic Python source from the recorded commit
and uses the pinned validation wheels for compiled kernels and dependencies.
It compares native output arrays against that reference, including weighted
and unweighted fitting, constant and extreme scores, duplicate scores, curve
boundaries, binary and multiclass outputs, ensemble modes, frozen adapters,
materialized splits, missing classes, default minority-class splits, and
explicit KFold and seeded shuffled KFold and StratifiedKFold. Production
classifier cases load GaussianNB, LDA, QDA and LogisticRegression from the
same revision and check binary and multiclass class order, predictions and
probabilities. SVC, GaussianNB, MultinomialNB, ComplementNB, BernoulliNB,
CategoricalNB, LDA and QDA cases exercise native round trips for all three
calibration methods in ensemble and pooled modes. The weighted ComplementNB
cases use `norm=True`.
The weighted BernoulliNB cases use a nondefault binarization threshold.
The weighted CategoricalNB cases use per-feature category minima and query
categories reserved by those minima that were absent during training.
Restored probabilities match the original Mojo outputs exactly. Against
scikit-learn, the allowed difference is `5e-6 * (1 + abs(expected))` for solver
rounding. Default LinearSVC cases cover all three calibrators, weighted and
unweighted fitting, and every ensemble setting. Frozen production cases use
disjoint training and calibration data and check that cloning preserves fitted
state and that automatic ensemble selection fits one calibrator.
Weighted LinearSVC cases with Int64 and Float32 component labels check
positive-stride targets against scikit-learn. The native contract also checks
negative-stride targets, weighted scoring and fitted-state retention.
Float32 estimator cases cover all three calibration methods with both response
types, pooled and ensemble fitting, and weighted scores. The reference receives
Float32 features for default LinearSVC and features widened from Float32 for
GaussianNB. scikit-learn's GaussianNB selects its statistics dtype from `X`, so
its Float32 path can produce different calibrated probabilities. The native
public test also checks column-major and negative-stride feature layouts.
Additional checks compare Float32 and string-label curves and native display
statistics, labels and metadata with the reference. SVG structure and native
ownership/overlay behavior have separate contract tests.

Curve and isotonic comparisons allow `2e-12 * (1 + abs(expected))` for Float64
reduction/interpolation differences. Optimized sigmoid and temperature
probabilities allow `5e-6 * (1 + abs(expected))` because the optimizers and
stopping criteria differ. On the three-sample, large-score case, the reference
stops with a gradient near `6.1e-7`; an independent BFGS solve with `gtol=1e-12`
confirmed a probability difference of about `3.2e-6` from that result.
Production composition probabilities use the same `5e-6` bound to account for
differences in the underlying classifiers' solvers and reductions. Labels
compare exactly. Fixtures, tool versions, seed, command and results are
retained under ignored `builds/calibration-parity/`.

## Moxie integration

Moxie owns the implementations under `src/moxie/calibration.mojo` and
`src/moxie/_calibration_*.mojo`. Shared estimator contracts, validation, isotonic
regression and dependency assembly come from their current Moxie owners. The
retired sibling and its historical NuMojo archive are not build inputs. The
compatible Mojo 1.0.0 lock remains in Moxie; the parity environment is isolated
under `builds/calibration-oracle` and installs the exact versions in
`tests/calibration/requirements-parity.txt`.

`FrozenCalibrationEstimator(FrozenEstimator(adapter^))` bridges an existing
`CalibrationEstimator` to Moxie's frozen ownership contract. Its clone retains
the shared fitted handle, its fit checks fitted state without training, and
`is_frozen()` selects pooled calibration for `ensemble="auto"`. It forwards
the adapter's class metadata and preferred response. For a supported production
classifier, use `adapt_classifier(frozen)` directly.
Frozen parameter replacement retains Moxie's shallow parameter semantics.

The isolated consumer uses `adapt_classifier` with Moxie's production
LinearDiscriminantAnalysis and exercises all methods, ordinary and frozen
adapters, fluent generic dispatch, custom splits, failed-fit preservation,
calibration curves and SVG export.
