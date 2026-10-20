# LinearRegression (dense)

`moxie.linear_model.LinearRegression[dtype=DType.float64]` ports
`sklearn.linear_model.LinearRegression` (reference revision 4358542a1b) for dense
Float32/Float64 NuMojo arrays.

```mojo
from moxie.linear_model import LinearRegression
var model = LinearRegression(fit_intercept=True, tol=1e-6, positive=False)
model.fit(X, y)                              # X (n, p); y (n,) or (n, k)
model.fit(X, y, sample_weight=w)             # w (n,), nonnegative
var y_hat = model.predict(X_new)             # (m,) or (m, k)
var r2 = model.score(X_new, y_true)          # optional sample_weight
```

| Constructor | Default | Notes |
| --- | --- | --- |
| `fit_intercept` | `True` | Weighted centering by the family plumbing |
| `copy_X` | `True` | Accepted; inputs are never modified for either value |
| `tol` | `1e-6` | lstsq cutoff: singular values `<= tol * s_max` are dropped (scipy `cond`) |
| `n_jobs` | `None` | Accepted and validated (0 rejected); targets are solved sequentially |
| `positive` | `False` | Nonnegative coefficients via Lawson-Hanson NNLS per target |

| Learned | Shape / type |
| --- | --- |
| `coef_` | `(p,)` for a rank-1 y, else `(k, p)` |
| `intercept_` | `(k,)`; length one for rank-1 y (sklearn: scalar) |
| `rank_`, `singular_` | `Optional[Int]`, `Optional[NDArray]` (`(min(n,p),)`); `None` when `positive=True` (sklearn leaves them unset) |
| `n_features_in_` | `Int` |

`get_params()` returns `LinearRegressionParams`; `set_params` accepts that record
or keyword updates; `clone(model)` returns an unfitted copy. A failed fit leaves
the previous learned state unchanged.

## Algorithm and kernels

Fit calls the family `preprocess_data` (weighted means, centering, sqrt-weight
row rescaling), then either `mojo_linalg.svd(X).solve(Y, rtol=tol, atol=0)`
(minimum-norm, rank-revealing; one-sided Jacobi SVD in Float64) or
`mojo_optimize.nnls` per target, and finally the family `set_intercept`.
Prediction is the family `linear_decision`; `score` is the family R2.

## Mojo adaptations (contract-tested)

- In-place `fit`; typed params record; compile-time dtype; errors are Mojo
  `Error` with `ValueError:`/`NotFittedError:`/`RuntimeError:`/`LinAlgError:` prefixes.
- `intercept_` is always an array (see above). Without an intercept, sklearn
  stores the scalar `0.0` even for several targets; here it is zeros `(k,)`.
- `positive=True, fit_intercept=False` with Float32: sklearn keeps NNLS's float64
  `coef_`; here `coef_` has the input dtype.
- NNLS iteration exhaustion raises `RuntimeError` like scipy.
- Scalar `sample_weight` is not accepted; pass a `(n,)` array.

## Performance baseline

`pixi run python tools/benchmark.py` writes `benchmarks/linear_regression_fit.json`.
On the Ryzen 9 5950X host (single thread) native OLS fits take 1.3 ms (200x10),
25 ms (1000x20) and 0.31 s (2000x50) versus 0.4-3.5 ms for sklearn (gelsd):
the scalar Jacobi SVD on the full tall matrix is the bottleneck (a QR
preconditioning step is the known next optimization). NNLS fits are within 3-6x.

`fit(..., sample_weight=2.0)` accepts a scalar nonnegative weight and expands it
to a vector in the input dtype. Regression `score` rejects nonfinite targets
and zero-total score weights before computing R2.
