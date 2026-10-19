# gaussian_process.kernels engine API

Pure-Mojo port of `sklearn.gaussian_process.kernels` (reference revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`) plus the GP algebra and optimizer
adapters shared by the regression and classification posteriors.

```mojo
from moxie.gaussian_process.kernels import ConstantKernel, Kernel, Matern, RBF, WhiteKernel

var kernel: Kernel = ConstantKernel(1.0) * RBF([1.0, 2.0]) + WhiteKernel(0.1)
var result = kernel.evaluate(X, eval_gradient=True)   # covariance (n,n), gradient (n,n,n_dims)
var K_cross = kernel(X, Y)                             # Float32 or Float64 input
kernel.set_param("k1__k2__length_scale", 3.0)
print(kernel)                                          # 1**2 * RBF(length_scale=3) + WhiteKernel(noise_level=0.1)
```

See `examples/gaussian_process_kernels.mojo` (run by `pixi run test`).

## Types

| Name | sklearn | Notes |
| --- | --- | --- |
| `Kernel` | `Kernel` and every concrete kernel | Type-erased composable value; implements the family `KernelContract` (`Input = Dense64`, `Params = KernelParams`). `copy()` is sklearn `clone`. |
| `ConstantKernel`, `WhiteKernel`, `RBF`, `Matern`, `RationalQuadratic`, `ExpSineSquared`, `DotProduct`, `PairwiseKernel` | same | Typed leaf structs with sklearn constructor names/defaults and public fields; convert implicitly to `Kernel` and support `+ * **` with kernels and floats. |
| `Sum`, `Product`, `Exponentiation`, `CompoundKernel` | same | Functions returning `Kernel`. |
| `Bounds` | `(lo, hi)`, per-element array, or `"fixed"` | `Bounds(1e-5, 1e5)`, `Bounds(lows, highs)`, `"fixed"` (implicit). |
| `Hyperparameter` | `Hyperparameter` | Alias of the family `HyperparameterSpec` (name, n_elements, fixed, raw bounds `(n,2)` or `(0,2)`). `value_type` is always "numeric". |
| `KernelParams`, `ParamValue` | `get_params()` dict | Ordered typed name/value pairs, nested `k1__...` names included when deep. |

`Kernel` methods: `evaluate(X, Y=None, eval_gradient=False)`, `__call__(X, Y=None)`,
`diag(X)`, `diag_of(X32)`, `theta()`, `set_theta(theta)`, `bounds()`, `n_dims()`,
`hyperparameters()`, `clone_with_theta(theta)`, `get_params(deep)`,
`set_params(params)`, `set_param(name, value)`, `is_stationary()`,
`requires_vector_input()`, `name()`, `k1()`, `k2()`, `kernel()`, `exponent()`,
`kernels()`, `==`, and `String(kernel)` (sklearn repr).

## Conventions (identical to sklearn)

- theta: natural logs of non-fixed hyperparameters, hyperparameter names sorted
  (RationalQuadratic: alpha, length_scale), operators k1 before k2,
  anisotropic values flattened. bounds: log bounds `(n_dims, 2)`.
- Gradients are with respect to log hyperparameters, shape `(n, n, n_dims)`;
  most leaves require `Y=None`. PairwiseKernel also accepts explicit `Y`,
  producing `(n, m, 1)` for free gamma, or `(n, n, 0)` for fixed gamma, as in
  sklearn. Compositions of pairwise leaves preserve NumPy shape/broadcast rules.
  Fixed hyperparameters contribute no columns.
- Matern nu in {0.5, 1.5, 2.5, inf} uses closed forms and analytic gradients;
  other nu uses `2^(1-nu)/Gamma(nu) (sqrt(2nu) d)^nu K_nu(sqrt(2nu) d)` with the
  native `mojo_linalg.bessel_kv` and, like sklearn, a 1e-10 forward-difference
  gradient. PairwiseKernel gradients are the same forward difference in log gamma.
- WhiteKernel cross-covariance is zero even for identical rows.
- CompoundKernel: covariance `(n, m, k)`, gradient `(n, n, p, k)` for self
  covariance or the matching child gradient shape for explicit `Y`, diag `(n, k)`.

## Language adaptations and documented differences

- All kernel arithmetic is Float64; Float32 input is widened exactly. sklearn keeps
  a few paths in float32 (e.g. `DotProduct`, linear pairwise); results agree to
  ~2e-6 relative. For forward-difference gradients on Float32 data, sklearn's
  difference of float32 matrices is dominated by float32 rounding; native results
  are compared against sklearn on the widened input.
- `exp`, `log` and `pow` use libm through LLVM intrinsics (`std.math.exp/log` in
  Mojo 1.0 are less accurate than NumPy by up to ~1e-11 relative).
- `PairwiseKernel` supports the dense string metrics linear, rbf, laplacian,
  poly/polynomial, sigmoid, cosine, chi2, additive_chi2 and precomputed.
  Callable metrics are rejected explicitly. `pairwise_kernels_kwargs` is a
  `Dict[String, Float64]` (`degree`, `coef0`; others ignored like `filter_params`).
- `PairwiseKernel(metric="precomputed")` returns an owned Float64 copy of `X`.
  Without `Y`, `X` must be square; with `Y`, `X.shape[1]` must equal `Y.shape[0]`,
  while feature counts may differ. Both inputs must be finite and nonempty.
  Gamma does not affect covariance, and its gradient is zero. The reference
  `diag` evaluates each row separately, so it accepts only one-column inputs;
  larger column counts raise. This matches sklearn and is not a general
  precomputed-GP estimator interface.
- `CompoundKernel.set_theta`/`clone_with_theta` split theta per child in order;
  sklearn's setter raises `AttributeError` (it references `self.k1`). Its
  `hyperparameters()` is empty, as in sklearn. CompoundKernel cannot be nested in
  operators (sklearn would fail on broadcasting).
- Kernels are not validated at construction (as in sklearn); invalid bounds text
  raises when bounds are used. Errors are Mojo `Error`s prefixed with the Python
  exception name.

## Private engine modules

- `moxie.gaussian_process._algebra.CholeskyAlgebra` implements
  `PositiveDefiniteAlgebra` over mojo-linalg `cholesky`/`triangular_solve`
  (`LinAlgError` text on non-SPD input, no jitter or fallback), plus `identity(n)`.
- `moxie.gaussian_process._optimization.LBFGSBOptimizer[O]` implements
  `HyperparameterOptimizer` with mojo-optimize's SciPy-faithful L-BFGS-B
  (SciPy default options, `propagate_nonfinite=True`, returned value = SciPy
  `res.fun`, non-convergence message in sklearn's `_check_optimize_result` wording).
  `RestartingOptimizer[Opt]` implements `OptimizerRestarts` (run 0 from theta,
  restarts `rng.uniform(bounds[:,0], bounds[:,1])`, first minimum wins, finite
  bounds required) and `run_all` also returns every run's warnings.

## Dependencies

See [numerical ownership](../README.md) for the live numerical source locations
and assembled namespaces. General numerical implementations remain in their
external libraries.
