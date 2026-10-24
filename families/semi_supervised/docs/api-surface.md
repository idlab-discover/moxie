# Dense estimator API

Import `LabelPropagation`, `LabelSpreading`, `SelfTrainingClassifier`, and
`GaussianNBAdapter` from `moxie.semi_supervised`. Learning and inference are pure
Mojo. The oracle is scikit-learn revision
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
pipeline step or a generic `SupervisedEstimator`. The only shipped self-training
adapters are `GaussianNBAdapter` and `GraphClassifierAdapter`. Other Moxie
classifiers, including `CalibratedClassifierCV` over `SVC` as used by the gallery
examples, need a hand-written `ProbabilisticEstimatorAdapter`.

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
