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
- `get_params()` returns an owned `MLPParams` record. Edit it and pass it to
  `set_params(params)`. Fit and set_params return a mutable borrow of the
  estimator. `clone` copies configuration and starts unfitted. A fitted
  `copy()` owns independent buffers.
- Learned state is read through `n_features_in_()`, `n_iter_()`, `t_()`,
  `n_layers_()`, `n_outputs_()`, `out_activation_()`, `loss_()`,
  `best_loss_()`, `loss_curve_()`, and the classifier's `classes_()`.
  Obtain coefficient and intercept snapshots through `coefs_()` and
  `intercepts_()`. Both default to the feature dtype. To read an L-BFGS model
  with Float32 features, use `coefs_[DType.float64]()` and
  `intercepts_[DType.float64]()`. `parameter_dtype()` reports the fitted dtype.
  L-BFGS has no loss history or best stochastic loss. In that case the getters
  return `None`.
- Each successful fit replaces `fit_warnings` with native records, including
  convergence warnings and stochastic batch-size clipping. A failed fit leaves
  the previous fitted state and warnings available. A one-column label matrix
  is accepted without Python's conversion warning. Stochastic verbose output
  prints the iteration and loss using Mojo's numeric formatter.

Incremental fit, warm-start refitting, validation-set early stopping,
multilabel classification and named partial parameter updates remain open.
The ordinary supervised, weighted and probability traits are available for
typed pipelines and dense CV. The incremental MLP capability traits remain
declarations for the incremental implementation.
