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
Unlike sklearn, both update forms validate configuration immediately and restore
the previous configuration on failure. String-based nested updates such as
`kernel__length_scale` remain unsupported; edit a kernel with `set_param` and
pass the resulting kernel to `set_params`.

## Behaviour (as sklearn)

- Prior prediction before fit (kernel or fixed `1**2 * RBF(1)`), `n_targets`
  replicates prior outputs.
- Shapes: single target mean/std `(n,)`, cov `(n, n)`, samples `(n, s)` (also for
  `(n, 1)` y); t>1 targets `(n, t)`, `(n, n, t)`, `(n, t, s)`.
- normalize_y uses mean/population std with sklearn's zero-scale guard.
- Optimization: SciPy-faithful L-BFGS-B (mojo-optimize) on the negative LML with
  analytic gradients; restarts draw `rng.uniform(bounds[:,0], bounds[:,1])`; the
  lowest `res.fun` wins; LML -inf on non-SPD K like sklearn.
- Warnings (sklearn `warnings.warn`) are printed to stderr as `UserWarning: ...`;
  fit's ConvergenceWarnings (non-converged runs, hyperparameters at bounds) are
  also kept in `convergence_warnings_`.
- Errors: Mojo `Error` prefixed with the sklearn exception name and message
  (non-SPD kernel `LinAlgError`, alpha length, n_targets, NaN input, feature count).

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
  the same distribution (parity compares whitened norms).
- Python reflection, pickling and metadata routing are not provided.

Kernels used: moxie.gaussian_process.kernels (Float64 kernels), mojo-linalg
Cholesky/triangular solves, mojo-optimize L-BFGS-B, moxie._random RandomState.
