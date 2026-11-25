# GaussianProcessRegressor

Pure-Mojo port of `sklearn.gaussian_process.GaussianProcessRegressor`
(reference revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`).

```mojo
from moxie.gaussian_process import GaussianProcessRegressor
from moxie.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

var gpr = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.01),
                                   n_restarts_optimizer=2, random_state=0)
gpr.fit(X, y)                                   # X (n, d), y (n,) or (n, t)
var r = gpr.predict(Xt, return_std=True)        # r.mean, r.std.value()
var cov = gpr.predict(Xt, return_std=False, return_cov=True).covariance.value()
var draws = gpr.sample_y(Xt, 5)                 # random_state=0 like sklearn
var lml = gpr.log_marginal_likelihood(theta, eval_gradient=True)  # .value, .gradient
```

Runnable from Moxie: `examples/gaussian_process.mojo`.

## Types

- `GaussianProcessRegressor` = `GenericGaussianProcessRegressor[DType.float64, FminLBFGSB]`.
  `GenericGaussianProcessRegressor[DType.float32]` accepts Float32 input;
  a second parameter selects a native optimizer.
- Constructor (sklearn names/defaults): `kernel=None` (positional `Kernel` or
  `Optional[Kernel]`), `alpha=1e-10` (Float64 or per-sample `List[Float64]`),
  `optimizer=FminLBFGSB()` (`None` disables), `n_restarts_optimizer=0`,
  `normalize_y=False`, `copy_X_train=True`, `n_targets=None`, `random_state=None`.
- Learned: `kernel_`, `X_train_`, `y_train_`, `L_`, `alpha_`,
  `log_marginal_likelihood_value_`, `n_features_in_`, `convergence_warnings_`.
- Implements the family traits `GaussianProcessRegressorAPI` (fit/predict/score R²),
  `GaussianProcessPosterior` (`sample_y`, `log_marginal_likelihood`),
  `GaussianUncertainty` (`predict(X, *, return_std, return_cov=False)` returning
  the family `PredictionResult`), plus `get_params`/`set_params`/`clone`.

## Structured inputs

Use `StructuredGaussianProcessRegressor[MyKernel]` with an explicit native
`KernelContract` implementation. For example, a sequence kernel can use
`List[String]` as its `Input`. Fit and predict receive the typed batch directly.

```mojo
from moxie.gaussian_process import StructuredGaussianProcessRegressor

# SequenceKernel is defined in the runnable regression contract below.
var gpr = StructuredGaussianProcessRegressor[SequenceKernel](kernel)
_ = gpr.fit(sequences, y)
var posterior = gpr.predict(query_sequences, return_std=True)
```

The batch's `len(X)` gives the sample count, and its `copy()` must preserve an
independent training snapshot. Implement `empty_input()` for the unfitted
estimator and return false from `requires_vector_input()`. The optional second
type parameter selects the target dtype (Float64 by default). `n_features_in_`
has no meaning for structured inputs and stays zero.

Import `KernelContract` and `KernelEvaluation` from `moxie.gaussian_process`
and `Hyperparameter` from `moxie.gaussian_process.kernels`. Return a Float64
covariance matrix of shape `(len(X), len(Y))` (square when `Y=None`), a
`(len(X),)` diagonal, and, when requested, a gradient tensor of shape
`(len(X), len(X), len(theta))`. Gradients and bounds are in log parameter
coordinates. Fixed kernels use empty theta, `(0, 2)` bounds and a zero-width
gradient. The default L-BFGS-B optimizer supports analytic gradients and
seeded restarts. Set `optimizer=None` to use the configured kernel as supplied.

The structured estimator takes a concrete kernel type at compile time and
uses typed constructor parameters. Both values of `copy_X_train` keep a
snapshot. Native persistence and the dense search adapter support the built-in `Kernel`
specialization.

The [sequence kernel contract](../../../tests/gaussian_process/regression/structured.mojo)
compares fixed and optimized fits against the pinned scikit-learn estimator
and gallery kernel through
[`structured_parity.py`](../../../tests/gaussian_process/regression/structured_parity.py).

## Custom vector kernels

Use `DenseGaussianProcessRegressor[MyKernel]` when the native
`KernelContract` has `Input = NDArray[DType.float64]`. The optional second
type parameter selects the input and target dtype. The estimator validates
rank, sample and feature counts, and finiteness before passing contiguous
Float64 matrices to the kernel. It records `n_features_in_` and snapshots
training input in its original dtype.

```mojo
from moxie.gaussian_process import DenseGaussianProcessRegressor
from moxie.kernel_ridge import KernelRidge, TypedGaussianProcessKernel

comptime Adapter = TypedGaussianProcessKernel[MyKernel]
var gp = DenseGaussianProcessRegressor[MyKernel](kernel)
var ridge = KernelRidge[DType.float64, Adapter](kernel=Adapter(kernel.copy()))
_ = gp.fit(X, y)
_ = ridge.fit(X, y)
```

The kernel is required at construction. Covariance, diagonal, and gradient
shapes and log parameter coordinates follow the contract above. GPR calls
`evaluate(X)` for a self-covariance. KernelRidge passes explicit `Y` even
during fit, matching scikit-learn. This matters for kernels with an independent
diagonal noise term. [Native contracts](../../../tests/gaussian_process/regression/vector_kernel.mojo)
and [pinned comparisons](../../../tests/gaussian_process/regression/vector_kernel_parity.py)
cover both estimators with a typed vector kernel.

## Parameter updates

`set_params(params)` replaces the typed constructor configuration.
`set_params(alpha=0.01, normalize_y=True)` updates only the supplied fields.
Both return a mutable borrow for chaining, such as
`gpr.set_params(alpha=0.01).fit(X, y).predict(Xt)`, and preserve learned state
until the next fit. `set_params()` leaves valid configuration unchanged.

All eight constructor parameters have named updates. `alpha` accepts a Float64
or `List[Float64]`. Omitted arguments leave fields unchanged; bare `None` also
leaves the optional update arguments unchanged.
For nullable parameters, wrap the value in its inner `Optional` to distinguish
clearing the field from omitting it:

```mojo
gpr.set_params(kernel=Optional[Kernel](RBF(2.0)), random_state=Optional[Int](0))
gpr.set_params(optimizer=Optional[FminLBFGSB](None), n_targets=Optional[Int](None))
```

Use `Optional[Kernel](None)` or `Optional[Int](None)` to clear `kernel` or
`random_state`. The optimizer wrapper uses the estimator's native optimizer type.
Both update forms validate configuration immediately and restore the previous
configuration on failure. To update a nested kernel directly, edit a kernel
with `set_param` and pass the resulting kernel to `set_params`.

## Cross-validation and search

`GaussianProcessSearchAdapter` supplies five unshuffled folds and the estimator's
R² score. Search candidate values are Float64. The adapter accepts scalar `alpha`
and the numeric parameters of an explicitly supplied kernel, including nested
names such as `kernel__k1__k2__length_scale`.

```mojo
from moxie.gaussian_process import GaussianProcessRegressor, GaussianProcessSearchAdapter
from moxie.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from moxie.model_selection import GridSearchCV

comptime Adapter = GaussianProcessSearchAdapter[]
var model = GaussianProcessRegressor(
    ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.01), optimizer=None,
)
var grid = Dict[String, List[Float64]]()
grid["alpha"] = [0.001, 0.05]
grid["kernel__k1__k2__length_scale"] = [0.2, 0.7, 1.4]
var search = GridSearchCV[Adapter](model, grid, Adapter())
_ = search.fit(X, y)
var prediction = search.predict(Xt)
```

Use `GaussianProcessSearchAdapter[DType.float32]` for Float32 data. An optional
second type parameter selects the estimator's optimizer type. Set
`optimizer=None` when comparing fixed kernel values. With an optimizer enabled,
candidate kernel values initialize the optimizer for each fold. To weight R²,
pass a `List[Float64]` of sample weights as `score_metadata`. See the
[noisy-data notebook](../../../notebooks/gaussian_process/plot_gpr_noisy.ipynb)
for a cross-validation and search example.

## Behaviour (as sklearn)

- Prior prediction before fit (kernel or fixed `1**2 * RBF(1)`), `n_targets`
  replicates prior outputs.
- Shapes: single target mean/std `(n,)`, cov `(n, n)`, samples `(n, s)` (also for
  `(n, 1)` y); t>1 targets `(n, t)`, `(n, n, t)`, `(n, t, s)`.
- For one query and multiple targets, the unfitted mean has shape `(t,)`, as in
  scikit-learn. Standard deviation has shape `(1, t)` and covariance has shape
  `(1, 1, t)`. Native `sample_y` handles the squeezed mean and returns `(1, t, s)`.
  For a single target, Moxie keeps the prior mean as a one-dimensional array,
  including for one query where scikit-learn returns a zero-dimensional array.
- normalize_y uses mean/population std with sklearn's zero-scale guard.
- Optimization: SciPy-faithful L-BFGS-B (mojo-optimize) on the negative LML with
  analytic gradients; restarts draw `rng.uniform(bounds[:,0], bounds[:,1])`; the
  lowest `res.fun` wins; LML -inf on non-SPD K like sklearn.
- Warnings (sklearn `warnings.warn`) are printed to stderr as `UserWarning: ...`;
  fit's ConvergenceWarnings (non-converged runs, hyperparameters at bounds) are
  also kept in `convergence_warnings_`.
- Errors: Mojo `Error` prefixed with the sklearn exception name and message
  (non-SPD kernel `LinAlgError`, alpha length, n_targets, NaN input, feature count).

## Native persistence

```mojo
from moxie.gaussian_process.serialization import GaussianProcessRegressorSchema
from moxie.serialization import load_model, save_model

save_model[GaussianProcessRegressorSchema[]](gpr, "regressor.moxie")
var restored = load_model[GaussianProcessRegressorSchema[]]("regressor.moxie")
```

Use `GaussianProcessRegressorSchema[DType.float32]` for Float32 input. The
experimental schema supports fitted and unfitted models with `FminLBFGSB` or
`optimizer=None`. It restores the configured and learned kernels separately, along
with the training snapshot, Cholesky factor, targets, normalization, dual
coefficients, cached likelihood and fit warnings. Loading preserves pending
parameter updates and restores learned state without fitting. Arrays are written
in logical order, including when the training snapshot is strided. The schema
rejects nonfinite learned arrays and invalid Cholesky factors. Custom optimizer
types and external random streams are outside this schema.
See [native model files](../../serialization.md) for the shared format.

## Language adaptations

- All numerical work is Float64. Float32 X/y are widened exactly; outputs and
  `y_train_` are Float64 (sklearn keeps some float32 intermediates).
- `copy_X_train=False` still stores an independent copy: `fit` borrows X
  immutably, so caller aliasing cannot be retained.
- `random_state=None` leases the shared process-default `RandomState`; `fit(X, y, rng)` / `sample_y(X, n, rng)` borrow and
  advance a caller stream (sklearn's RandomState instance semantics). An explicit
  `sample_y(X, n, None)` also advances the shared default stream; omitted sampling
  state remains integer 0.
- `log_marginal_likelihood` takes a mutable receiver so `clone_kernel=False`
  can update `kernel_`. The default clones the kernel and preserves its theta.
  Neither path refits or replaces the cached posterior factorization.
- Custom optimizer: a struct implementing `GPROptimizer`
  (`minimize(mut LogMarginalLikelihood, initial_theta, bounds) -> OptimizationResult`,
  Defaultable, ImplicitlyCopyable), the family `HyperparameterOptimizer` contract
  specialized to the GPR objective; replaces sklearn's Python callable.
- `sample_y` draws via `RandomState.multivariate_normal` (native SVD factor): the
  standard-normal stream matches NumPy but singular-vector signs may differ, so
  individual draws can differ from NumPy by an orthogonal transform while having
  the same distribution (parity compares whitened norms). As with NumPy's
  default `check_valid="warn"`, a predictive covariance that fails the SVD
  reconstruction check (slightly indefinite posteriors, e.g. eigenvalues near
  -1e-8 on dense query grids) prints
  `RuntimeWarning: covariance is not symmetric positive-semidefinite.` and
  still draws.
- Python reflection, pickling and metadata routing are not provided.

Kernels used: moxie.gaussian_process.kernels (Float64 kernels), mojo-linalg
Cholesky/triangular solves, mojo-optimize L-BFGS-B, moxie._random RandomState.
