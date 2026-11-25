# Dense multilayer perceptrons

```mojo
from moxie.neural_network import MLPClassifier, MLPRegressor
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler

var model = MLPClassifier(
    hidden_layer_sizes=[10, 10], solver="lbfgs", alpha=0.1, random_state=1,
)
var pipeline = make_pipeline(StandardScaler(), model^)
_ = pipeline.fit(X, y)
var probabilities = pipeline.predict_proba(X)
```

Both estimators implement ordinary fit with L-BFGS, SGD or Adam. SGD supports
constant, invscaling and adaptive learning rates. The regressor accepts
`loss="squared_error"` or `loss="poisson"`. It requires finite nonnegative targets
for Poisson, whose objective is undefined for negative targets. It returns
a vector for one output and a matrix for multiple outputs. Its score is R²
with uniform output averaging and Float64 reductions. The classifier supports
one or more distinct integer classes in a vector or single-column matrix.
It orders `classes_()` in ascending order, predicts the original label dtype,
and scores weighted or unweighted accuracy. Its binary probability result has
two columns, including the one-class case supported by scikit-learn.
Multiclass probability columns follow the learned class order.

Use `MLPClassifier[dtype, label_dtype]` or `MLPRegressor[dtype]` to specialize
the input dtype. Defaults are Float64 features and Int64 classifier labels.
Features are finite rank-2 NuMojo arrays. Regression targets and sample weights
use the feature dtype. Integer classifier targets must be representable in
Int64 for the shared encoder. All three solvers and prediction accept valid strided
views. L-BFGS compacts strided features before widening for prediction because
NuMojo's `astype` assumes contiguous storage.

The typed predictor returns regressions and probabilities in the compile-time
feature dtype. With Float32 inputs, scikit-learn returns Float64 for L-BFGS
because it retains double-precision coefficients. Moxie also retains
double-precision coefficients and uses them for L-BFGS inference, then rounds
the default result to Float32. Use `predict[DType.float64](X)` for regression
or `predict_proba[DType.float64](X)` and `predict_log_proba[DType.float64](X)`
for probabilities to retain L-BFGS output precision.
Class selection uses the native solver precision
before rounding. SGD and Adam coefficients retain the input dtype.

- `hidden_layer_sizes` is a `List[Int]`. An empty list selects a network with no
  hidden layer. `batch_size=None` means automatic batch sizing.
  `n_iter_no_change=None` corresponds to infinite patience.
- `fit(X, y, sample_weight=weights)` borrows a weight vector. Without an integer
  seed, use `fit(X, y, rng)` or its weighted overload with an explicit mutable
  `RandomState`. A seeded ordinary fit starts a new stream for each call.
- `get_params()` returns an owned `MLPParams` record. Pass an edited record to
  `set_params(params)`, or update individual constructor settings with keywords
  such as `set_params(alpha=0.01, max_iter=50)`. Named updates preserve omitted
  settings and existing learned state. Fit validates the new configuration.
  Fit and set_params return a mutable borrow of the estimator. `clone` copies
  configuration and starts unfitted. A fitted `copy()` owns independent buffers.
- Named updates for `batch_size`, `random_state` and `n_iter_no_change` use a
  nested optional. Pass `Optional[Int](value)` to assign a value, and
  `Optional[Int]()` to clear it. Omitting the keyword keeps the existing value.
- Learned state is read through `n_features_in_()`, `n_iter_()`, `t_()`,
  `n_layers_()`, `n_outputs_()`, `out_activation_()`, `loss_()`,
  `best_loss_()`, `loss_curve_()`, and the classifier's `classes_()`.
  Obtain coefficient and intercept snapshots through `coefs_()` and
  `intercepts_()`. Both default to the feature dtype. To read an L-BFGS model
  with Float32 features, use `coefs_[DType.float64]()` and
  `intercepts_[DType.float64]()`. `parameter_dtype()` reports the fitted dtype.
  A model trained only with L-BFGS returns `None` for both loss-history getters.
- Each successful fit replaces `fit_warnings` with native records, including
  convergence warnings and stochastic batch-size clipping. A failed fit leaves
  the previous fitted state and warnings available. A one-column label matrix
  is accepted without Python's conversion warning. Stochastic verbose output
  prints the iteration and loss using Mojo's numeric formatter.

## Warm-start refitting

With `warm_start=True`, `fit` starts from the fitted coefficients and
intercepts. It requires the same feature width, hidden-layer sizes, output
shape and output activation. The classifier also requires the same class
domain. Set `warm_start=False` to start a fresh fit when changing those
boundaries.

Each stochastic warm fit creates a fresh optimizer with the current settings,
keeps the loss history, best loss and stopping counter, and continues `t_()`.
`n_iter_()` reports iterations from the current call. Integer seeds are
reapplied per call, and an explicit RNG skips initialization draws on warm
fits. A warm L-BFGS refit from L-BFGS state starts from the retained
Float64 parameters and evaluates in Float64. Float32 training arrays are
converted once for that solve.
When switching from L-BFGS to SGD or Adam, coefficients convert to the feature
dtype. Moxie initializes stochastic history if this is the first stochastic
fit, where pinned scikit-learn requires history from a prior stochastic fit.

## Incremental training

`partial_fit` performs one pass over the supplied batch with SGD or Adam.
It retains parameters, optimizer buffers and training history between calls,
including when the first update follows an ordinary stochastic fit. `n_iter_()`
is one after each successful update. `t_()` counts all training rows seen and
`loss_curve_()` gains one entry per update. The optimizer keeps the settings
used to create it. A fresh ordinary fit starts a new optimizer and history.

When updating an unfitted classifier, pass the full class domain as a
nonempty rank-1 array of the label dtype. The method sorts and deduplicates it.
Later batches may contain a subset of those classes. If supplied again, the
class domain must match. When `warm_start=True`, each incremental batch must contain
the full class domain, as in scikit-learn.

```mojo
var classifier = MLPClassifier(solver="sgd", random_state=42)
_ = classifier.partial_fit(X_first, y_first, classes=all_classes)
_ = classifier.partial_fit(X_next, y_next, sample_weight=weights_next)
```

Both estimators support keyword `sample_weight`, an explicit mutable
`RandomState` as the third argument, and fluent mutation through
`MLPClassifierAPI` and `MLPRegressorAPI`. Integer seeds are reapplied on each
call. An explicit RNG advances as training draws from it, including if a
later minibatch raises. Failed updates preserve learned state and fit warnings.

Mojo exposes the method through the incremental traits for every constructor
configuration. Calling it with L-BFGS, L-BFGS fitted state or
`early_stopping=True` raises `ValueError`. The pinned Python regressor can
perform a full L-BFGS solve through its `partial_fit` method. The native
incremental contract limits every successful call to one stochastic epoch.
Continuing a fit requires the same input width, hidden topology, output shape
and output activation. Start a fresh ordinary fit to change these boundaries.

Validation-set early stopping and multilabel classification remain open.
Ordinary supervised, weighted and probability traits remain available for
typed pipelines and dense CV.
