# Calibration

The implementation follows `sklearn.calibration` at revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Library code is pure Mojo.
Python, NumPy, SciPy and scikit-learn are used only for validation.

## Using the estimator

```mojo
from moxie.calibration import CalibratedClassifierCV

# classifier implements CalibrationEstimator; X and y are NuMojo arrays.
var model = CalibratedClassifierCV(classifier, method="sigmoid", cv=5)
model.fit(X, y)
var probabilities = model.predict_proba(X_test)
var labels = model.predict(X_test)
```

`X` has shape `(n_samples, n_features)`; `y` and optional weights have shape
`(n_samples,)`. All arrays use `numojo.NDArray[DType.float64]`. Inputs must be
finite and nonempty. Class labels are integer-valued numbers with magnitude
at most `2**53 - 1`. Weights must be nonnegative with positive finite sum.
The estimator rejects unsupported dtypes at compile time. Strided inputs use
NuMojo coordinate indexing; learned arrays do not alias input arrays.

`classes_` is sorted. `predict_proba` returns `(n_samples, n_classes)` with
columns in that order. `predict` breaks ties in favor of the first class.
`score` computes weighted accuracy. Normal fitting replaces learned state;
a failed fit preserves the previous successful fit. `clone` clones the base
estimator configuration and returns an unfitted calibrator. A frozen base
estimator's clone retains its own fitted state.

Constructor parameters are `estimator`, `method="sigmoid"`, `cv=None`,
`n_jobs=None`, and `ensemble="auto"`. An explicit estimator is required because
this family does not provide the reference's default LinearSVC. `cv=None`
means five stratified folds. Explicit integers of at least two require each
class to have at least `cv` samples, as in sklearn. With `cv=None`, smaller
minority classes are allowed when the largest class has at least five samples.
The reference warns for that case; the native implementation emits no warning.
`cv` also accepts `List[CalibrationSplit]`, where each split owns `train` and
`test` index lists. This is the native equivalent of sklearn's iterable of
index pairs. Materialize splitter output before passing it to the constructor.
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
configuration. `fit` and `set_params` return a mutable borrow of the calibrator, following
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

The pinned sklearn temperature wrapper does not encode nonconsecutive numeric
labels before its multinomial loss. This implementation always encodes labels
as consecutive class indices and maps predictions back through `classes_`.
Temperature wrapper parity therefore uses explicitly encoded reference labels
and maps the reference predictions back. This is a deliberate correction of
that reference defect, covered by the nonconsecutive-label cases.

Multiclass sigmoid/isotonic fits one calibrator per class, then normalizes.
If all return zero, the output is uniform. Binary outputs use the positive
probability and its complement.

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
These additional input types apply to `calibration_curve`; the estimator and
display interfaces still use Float64 arrays.

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

## Limits and validation

Remaining gaps include the default LinearSVC, sparse estimators, Float32 or
string-label estimator fitting, direct splitter-object dispatch, arbitrary fit
metadata, parallel jobs, metadata routing, persistence, feature names and
Matplotlib object interoperability. Temperature ensemble folds with missing
training classes are rejected explicitly. These are scope limits, not claims
of complete sklearn compatibility.

From the Moxie root run `pixi run build`, `pixi run test`, and `pixi run parity`.
For calibration alone run `pixi run python tools/calibration_checks.py test`
and `pixi run python tools/calibration_checks.py parity`. Run the built example
with `builds/examples/calibration`.
Parity loads calibration and isotonic Python source from the recorded commit
and uses the pinned validation wheels for compiled kernels and dependencies.
It compares native output arrays against that reference, including weighted
and unweighted fitting, constant and extreme scores, duplicate scores, curve
boundaries, binary and multiclass outputs, ensemble modes, frozen adapters,
materialized splits, missing classes and default minority-class splits.
Additional checks compare Float32 and string-label curves and native display
statistics, labels and metadata with the reference. SVG structure and native
ownership/overlay behavior have separate contract tests.

Curve and isotonic comparisons allow `2e-12 * (1 + abs(expected))` for Float64
reduction/interpolation differences. Optimized sigmoid and temperature
probabilities allow `5e-6 * (1 + abs(expected))` because the optimizers and
stopping criteria differ. On the three-sample, large-score case, the reference
stops with a gradient near `6.1e-7`; an independent BFGS solve with `gtol=1e-12`
confirmed a probability difference of about `3.2e-6` from that result.
Labels compare exactly. These tolerances apply to
the tested cases, not arbitrary ill-conditioned data. No performance claim is
made. Fixtures, tool versions, seed, command and results are retained under
ignored `builds/parity/`.

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
the adapter's class metadata and preferred response. A general FrozenEstimator
does not by itself provide those methods. Adapt the classifier first.
Frozen parameter replacement retains Moxie's shallow parameter semantics.

The isolated consumer uses Moxie's production LinearDiscriminantAnalysis with
integer labels. Its test adapter converts numeric targets to `List[Int]`,
reshapes binary decision vectors to one-column matrices, and explicitly
rejects sample weights because LDA has no weighted fitting contract. The
centroid example remains a demonstration classifier. The consumer exercises
all methods, ordinary and frozen adapters, fluent generic dispatch, custom
splits, failed-fit preservation, calibration curves and SVG export.
