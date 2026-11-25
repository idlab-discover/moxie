# Dense estimator API

Import `LabelPropagation`, `LabelSpreading`, `SelfTrainingClassifier`, and
the classifier adapters from `moxie.semi_supervised`. Learning and inference
are pure Mojo. The oracle is scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

## Graph estimators

```mojo
from numojo import zeros, Shape
from moxie.semi_supervised import LabelPropagation

var X = zeros[DType.float64](Shape(3, 1))
X.unsafe_set(1, 1)
X.unsafe_set(2, 0.9)
var labels = List[Int]([0, 1, -1])
var model = LabelPropagation(gamma=2)
model.fit(X, labels)
var predictions = model.predict(X)
var probabilities = model.predict_proba(X)
```

Features are rank-two Float32 or Float64 NuMojo arrays. Use
`LabelPropagation[DType.float32]` or `LabelSpreading[DType.float32]` for Float32.
Distributions and probabilities are Float64, with columns in sorted `classes_`
order. `-1` is the integer unlabeled sentinel; other negative integers are labels.
Use explicitly typed target lists to avoid ambiguity between tagged and integer
fit overloads.

Constructor defaults follow sklearn: `kernel="rbf"`, `gamma=20`,
`n_neighbors=7`, `tol=0.001`, `n_jobs=None`; `max_iter=1000` for propagation and
`max_iter=30`, `alpha=0.2` for spreading. The implementations are specializations
of one native `GraphClassifier`. Its typed parameter record includes `alpha` for
both variants; propagation does not use alpha. RBF execution is serial and
`n_jobs` is retained as configuration, as it has no effect on this reference path.

Both expose `fit`, `predict`, `predict_proba`, weighted accuracy `score`,
`get_params`, `set_params`, `__sklearn_clone__`, and `__sklearn_is_fitted__`.
Updates use typed parameter records or named keyword arguments. `fit` and
`set_params` return a mutable borrow of the estimator for fluent chaining,
following the current Moxie contract. Clone contains configuration
only. Learned attributes are `X_`, `classes_`, `label_distributions_`,
`transduction_`, `n_features_in_`, and `n_iter_`. `X_` is an independent copy,
including when the input is strided.

Weighted `score` requires one finite weight per sample and a nonzero total
weight. Negative weights are accepted, matching sklearn's accuracy calculation.

`converged_` additionally records whether the convergence test succeeded.
Reaching the iteration limit prints a `ConvergenceWarning` diagnostic to stdout;
there is no Python warnings registry. Learned fields have empty initial values;
inference checks fitted state before using them. A failed fit invalidates inference.

For string labels, select the label type explicitly:

```mojo
from moxie.semi_supervised._affinity import RBFAffinity
var model = LabelPropagation[DType.float64, RBFAffinity[DType.float64], String]()
var y: List[Optional[String]] = [String("cat"), None, String("dog")]
model.fit(X, y)
```

`None` marks missing string labels; the string `"-1"` is an ordinary class.

A callable is a `DenseAffinity & Copyable & Deinitable` provider with
`Input = NDArray[dtype]`. Instantiate `LabelSpreading[dtype, Provider]` with
`kernel="callable", affinity=provider`. It returns an owned Float64 matrix with
training rows and query columns. Shape is checked. Asymmetric affinities retain
their orientation; spreading uses column degrees excluding the diagonal. Provider
copy semantics must preserve independently owned constructor configuration.

## Self-training

```mojo
from moxie._nb_gaussian import GaussianNB
from moxie.semi_supervised import SelfTrainingClassifier, GaussianNBAdapter

var base = GaussianNBAdapter(GaussianNB())
var model = SelfTrainingClassifier[type_of(base)](base.copy(), threshold=0.75)
model.fit(X, labels)
var probabilities = model.predict_proba(X)
```

Pass an explicit estimator adapter. The missing-estimator default raises at fit.
Defaults are `threshold=0.75`, `criterion="threshold"`, `k_best=10`,
`max_iter=10`, and `verbose=False`. `max_iter=None` continues until all samples
are labeled or an iteration adds none. Zero iterations still fit the estimator
on the labeled rows. Threshold selection is strictly greater than the threshold. Equal-confidence
k-best candidates are selected in input order. NumPy's unstable partition may
choose different members when a tie crosses the selection boundary.

`estimator_` is an optional, separately cloned adapter. `classes_` preserves its
probability-column order. `transduction_` is `List[Optional[Label]]`, including
for integer labels; `None` denotes samples still unlabeled. `labeled_iter_`,
`n_iter_`, `termination_condition_`, and `n_features_in_` follow sklearn semantics.
Termination strings are `no_change`, `max_iter`, and `all_labeled`, with the
reference's precedence when conditions coincide.

Adapters implement `ProbabilisticEstimatorAdapter` from `_contracts`.
`PredictingEstimatorAdapter`, `LogProbabilityEstimatorAdapter`, and
`DecisionEstimatorAdapter` from `_self_training` enable the corresponding
methods; `ScoringEstimatorAdapter` enables delegated scoring. Missing capabilities
are unavailable at compile time. The wrapper does not replace a base estimator's
prediction rule with probability argmax. Probability rows and columns are checked;
training probabilities must be finite and in `[0,1]` and are never renormalized.

`GaussianNBAdapter[dtype]` selects the feature and GaussianNB state dtype. It
supports prediction, probabilities, log probabilities, and scoring. `GraphClassifierAdapter` is another concrete native adapter.

```mojo
from moxie.linear_model import LogisticRegression
from moxie.semi_supervised import LogisticRegressionAdapter, SelfTrainingClassifier

var base = LogisticRegressionAdapter(LogisticRegression(C=1.7))
var model = SelfTrainingClassifier[type_of(base)](base^)
_ = model.fit(X, labels)
```

`LogisticRegressionAdapter` accepts the native dense L-BFGS classifier with
`int64` labels. It converts `List[Int]` to a rank-one target array and returns
class labels and predictions as owned `List[Int]` values. Binary and multiclass
prediction, probabilities, log probabilities, decision values, and scoring
delegate to logistic regression. Probabilities and decision values use the
self-training Float64 output contract, with the binary decision vector and
multiclass decision matrix following the underlying classifier. For Float32
features, construct `LogisticRegression[DType.float32]()` and specify
`DType.float32` as the second type parameter of `SelfTrainingClassifier`.

`CalibratedClassifierAdapter` accepts a native
`CalibratedClassifierCV` whose base implements `CalibrationEstimator`. It
delegates prediction, probabilities, and scoring, and accepts Float64 features and
`List[Int]` targets, preserving the calibrated classifier's class order.
Labels must be in the closed interval `[-(2^53 - 1), 2^53 - 1]` because
calibration represents labels as Float64. As with other integer adapters,
`-1` in a self-training target list means unlabeled.

SVC uses calibration's explicit fitted-class reader:

```mojo
from numojo import NDArray, Shape
from moxie.calibration import CalibratedClassifierCV, ClassifierCalibrationAdapter
from moxie.semi_supervised import CalibratedClassifierAdapter, SelfTrainingClassifier
from moxie.svm import SVC

def svc_classes(model: SVC[Int]) raises -> NDArray[DType.float64]:
    var classes = NDArray[DType.float64](Shape(len(model.classes_)))
    for i in range(len(model.classes_)):
        classes.unsafe_set(i, Float64(model.classes_[i]))
    return classes^

# In a raising function with Float64 X and List[Int] labels:
var svc = SVC[Int](gamma=0.5, random_state=42)
var calibrated = CalibratedClassifierCV(ClassifierCalibrationAdapter(svc^, svc_classes, "SVC"))
var adapter = CalibratedClassifierAdapter(calibrated^)
var model = SelfTrainingClassifier[type_of(adapter)](adapter^)
_ = model.fit(X, labels)
```

`get_params` recursively clones estimator configuration into a typed record;
modify that record and pass it to `set_params`. Dynamic `estimator__parameter`
routing is not implemented.

## Composition

Graph estimators conform to `SemiSupervisedClassifier`. They can end a
`moxie.pipeline` pipeline (for example `make_pipeline(StandardScaler(),
LabelSpreading())`), be cloned with `moxie.base.clone`, and be evaluated by
`moxie.model_selection.cross_val_score` through a typed `CVAdapter` and scorer.
`SelfTrainingClassifier` conforms only to `BaseEstimator`, because `predict` and
`score` exist only for adapters that provide them. It therefore cannot be a
pipeline step or a generic `SupervisedEstimator`. The shipped adapters are
`GaussianNBAdapter`, `GraphClassifierAdapter`, `LogisticRegressionAdapter`,
and `CalibratedClassifierAdapter`. Other classifiers can implement the
`ProbabilisticEstimatorAdapter` contract.

## Exclusions

Sparse data, KNN graphs, Python callbacks, metadata routing, feature names,
Python reflection, serialization, and Python exception classes are not supported.
Errors use descriptive Mojo `Error` messages. There are no fallback distributions
for disconnected samples: fitting preserves zero rows, while prediction divides
by zero with IEEE NaN results, following the reference.

## Numerical agreement

Oracle checks use absolute tolerances of 2e-9 for Float64 graph/GaussianNB
probabilities, 3e-5 for Float32 graphs, and 5e-5 for Float32 GaussianNB. Float32
intermediate rounding and different matrix reduction orders explain these bounds
on the checked datasets; they are not universal error guarantees. Labels match
exactly away from probability ties. Near-tie label checks permit a maximizer
within 1e-14 for Float64 or 3e-7 for Float32. For example, a sklearn distribution
of `[0.5, 0.5000000000000001]` can be exactly tied in native arithmetic. Native
argmax chooses the first class on an exact tie. Iteration counts are checked
exactly on the oracle fixtures.

The binary and multiclass logistic regression compositions use absolute
tolerances of 2e-5 for probabilities and 1e-4 for log probabilities and decision
values with Float64 features. The corresponding Float32 bounds are 9e-4 and
3e-3, as in the logistic regression owner's Float32 oracle. These bounds
account for optimizer and rounding differences between native L-BFGS and the
pinned sklearn reference.

The calibrated SVC composition checks probabilities to an absolute tolerance
of 3e-5 for the binary and multiclass Float64 fixtures, allowing for
differences in LIBSVM kernel arithmetic and sigmoid optimization. Class order,
predictions, score, pseudo-label iterations, and termination match exactly on
these fixtures. The oracle executes pinned self-training and calibration
sources and requires the installed SVC wrapper and packaged LIBSVM sources to
match the same revision byte for byte.
