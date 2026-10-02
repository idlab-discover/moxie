# QuantileTransformer, PowerTransformer and helpers

Conventions: [api.md](../api.md). Source: `src/moxie/preprocessing/_quantiles.mojo`,
with family-private numerics in `_special_math.mojo` (Cephes normal CDF/quantile,
Box-Cox, NumPy-order sums, `logsumexp`, linear percentile, `np.interp`) and
`_optimize.mojo` (SciPy `brent`/`fminbound` port). Ported NumPy and SciPy code
keeps its BSD-3-Clause notices (`LICENSES/numpy.txt`, `LICENSES/scipy.txt`).

| Name | Native signature |
| --- | --- |
| `QuantileTransformer[dtype]` | `QuantileTransformer(*, n_quantiles=1000, output_distribution="uniform", ignore_implicit_zeros=False, subsample=10000, random_state=None, copy=True)`; learned `n_quantiles_`, `quantiles_` (Float64, `(n_quantiles_, n_features)`), `references_` (Float64) |
| `PowerTransformer[dtype]` | `PowerTransformer(method="yeo-johnson", *, standardize=True, copy=True)`; learned `lambdas_` (X's dtype) |
| `quantile_transform` | `quantile_transform(X, *, axis=0, n_quantiles=1000, output_distribution="uniform", ignore_implicit_zeros=False, subsample=100000, random_state=None, copy=True) -> NDArray` |
| `power_transform` | `power_transform(X, method="yeo-johnson", *, standardize=True, copy=True) -> NDArray` |

`subsample` and `random_state` are `Optional[Int]` (`None` disables subsampling /
selects the default stream). Inputs are rank-2 Float32/Float64 arrays, any
strided view; NaN is ignored in fit and kept in transform, infinity is rejected.

```mojo
from moxie._random import RandomState
from moxie.preprocessing import QuantileTransformer, PowerTransformer, quantile_transform

var qt = QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=0)
var Z = qt.fit_transform(X)
var X_back = qt.inverse_transform(Z)
var rng = RandomState(42)
qt.fit(X, rng)                      # sklearn random_state=RandomState(42): advances rng
var pt = PowerTransformer("box-cox")
var G = pt.fit_transform(X)         # lambdas in pt.lambdas_
pt.transform_inplace(X)             # copy=False, no allocation
var per_row = quantile_transform(X, axis=1, n_quantiles=10)
```

## Behavior

QuantileTransformer follows sklearn's dense path: `n_quantiles_ =
max(1, min(n_quantiles, n_samples))`, `references_ = linspace(0, 1)`, and
`quantiles_` are NumPy's `nanpercentile(..., method="linear")` per feature (an
all-NaN feature gives NaN quantiles). With `subsample < n_samples` the rows are
`arange(n)` shuffled by NumPy's legacy `RandomState.shuffle` and truncated,
exactly as sklearn's `resample(replace=False)`, using `moxie._random`. Transform
averages the forward and reversed `np.interp` (NumPy's search-with-guess
semantics, including repeated quantiles), maps values equal to (uniform) or
within 1e-7 of (normal) the fitted bounds to the output bounds, and for the
normal output applies `norm.ppf` clipped at `ppf(1e-7 - eps)`. The inverse uses
`norm.cdf`. Float32 data rounds at the same points as sklearn's float32 arrays.
`quantile_transform(axis=1)` fits on a transposed view (no copy).

PowerTransformer follows sklearn: Yeo-Johnson features whose float64 variance is
indistinguishable from constant get lambda 1 and are left unchanged by
`fit`/`fit_transform` (but `transform` applies lambda 1, as sklearn does);
other lambdas maximize SciPy's log-likelihood on the non-NaN values:
Yeo-Johnson with `fminbound` on SciPy's overflow-safe bounds (`xtol=1.48e-8`,
bounds in the data dtype), Box-Cox with `brent` from the bracket `(-2, 2)`
(`tol=1.48e-8`, `maxiter=500`) followed by SciPy's `ymax = finfo.max / 10000`
guard, which constrains lambda through the Lambert W (k=-1) branch.
`lambdas_` is stored in X's dtype and transforms use the stored lambdas, so
Float32 transforms replay sklearn's float32 ufunc arithmetic. `standardize=True`
uses the family `StandardScaler[dtype]` fitted on the transformed data. Errors
match sklearn: Box-Cox rejects `nanmin(X) <= 0` in fit and transform, an all-NaN
feature raises (`Column must not be all nan.` for Box-Cox, sklearn's
`not enough values to unpack (expected 2, got 0)` for Yeo-Johnson), a constant
Box-Cox feature raises `Data must not be constant.`; a bracket failure raises
`RuntimeError:` as SciPy's `BracketError`.

## Native adaptations

* `random_state` is `Optional[Int]`. An integer reseeds a fresh NumPy-compatible
  `RandomState` on every fit (sklearn's `check_random_state(int)`); `None` leases
  the process default stream of `moxie._random` (reseed with
  `seed_global_random_state`), which is not NumPy's global stream. A RandomState
  instance cannot be stored as a parameter: `fit(X, mut rng: RandomState)` draws
  from and advances a caller-owned state instead, matching a fitted
  `random_state=np.random.RandomState(...)` sequence.
* `set_params(subsample=..., random_state=...)` take `Optional[Optional[Int]]`:
  the outer `None` leaves the value, an inner `None` sets the parameter to None.
* `copy=False` cannot mutate a borrowed argument; `transform_inplace` and
  `inverse_transform_inplace` implement it for writable arrays.
* There is no warning channel: `ignore_implicit_zeros=True` on dense input,
  `n_quantiles > n_samples` and SciPy's lambda-constraint warning are silent
  (behavior is otherwise identical). Sparse input is unavailable by type.
* Quirk preserved: for Float32 input whose first feature is all-NaN, NumPy's
  `apply_along_axis` rounds every quantile to float32; the values are rounded
  likewise but `quantiles_` stays a Float64 array.

## Memory

QuantileTransformer fit holds one Float64 workspace of `min(n_samples,
subsample)` values plus `quantiles_`; subsampling adds sklearn's Int64
permutation of `n_samples` rows (no row copy of X). Transforms allocate only the
result (none in place). PowerTransformer fit holds a gathered Float64 column and
two (Yeo-Johnson) or three (Box-Cox) Float64 workspaces of `n_samples`; with
`standardize=True` or `fit_transform` it also holds the transformed table
(the result). Transforms allocate only the result plus O(features) state.

## Parity

`tests/parity/test_quantiles.py` (1725 checks) compares against the pinned
sklearn with NumPy 2.5.3 / SciPy 1.18.1: Float32/Float64 Gaussian, NaN, all-NaN
column, tied/discrete, toy, bounds, single-row, near-zero and 2500-row data;
`n_quantiles` 1/5/17/300/1000 with and without seeded subsampling; uniform and
normal outputs; `quantiles_`, `references_`, `n_quantiles_`, fit_transform,
out-of-range/NaN transforms, inverse transforms (including out-of-range
inputs) and in-place transforms; column-major and negative-stride inputs;
`quantile_transform` on both axes; integer seeds (including 2**32-1) and an
advancing RandomState; feature names. PowerTransformer: log-normal, mixed-sign,
negative, NaN, constant-column, strongly non-Gaussian, extreme-skew, year-like
(Box-Cox overflow guard in Float64) and small-spread Float32/Float64 data
(Box-Cox overflow guard), both methods, `standardize` on/off, fit vs
fit_transform, transform of new data, inverse transforms, `power_transform`,
transforms with fixed lambdas (0, 1, 2, negative) and the error cases above.
Native contracts: `tests/test_quantiles.mojo`.

Tolerances and why:

* `quantiles_`, `references_`, `n_quantiles_` and seeded subsamples: exact (same
  percentile arithmetic on the same rows).
* Quantile transforms: Float64 rtol 1e-12 (atol 1e-15); Float32 rtol 2.4e-7
  (two float32 ulps). The interpolation is replayed operation by operation and
  `ndtr`/`ndtri` are SciPy's Cephes routines (bit-identical on a 200k-point
  grid); remaining differences are final-bit rounding.
* Fixed-lambda power transforms: Float64 rtol 1e-12; Float32 rtol 5e-7 (log1p
  is Kahan's construction on libm `log`, within 2 ulp of libm `log1p`).
* Lambdas: atol 2e-7 + rtol 2e-7. The native objectives agree with SciPy's to
  about one ulp, but near the optimum the log-likelihood is flat, so one-ulp
  differences can flip a Brent comparison; both runs then stop within the
  minimizer's resolution (`xtol` 1.48e-8 relative, widened by the objective's
  rounding noise to about sqrt(eps * |llf| / curvature), ~1e-7 at these sizes).
  Observed worst: 3.6e-7 relative (5.6e-8 absolute).
* Outputs depending on fitted lambdas: Float64 rtol/atol 1e-6, Float32 2e-5,
  the lambda tolerance propagated through `d transform / d lambda` and the
  standardization.

Not covered: sparse input, pandas output, warnings, and SciPy's non-default
optimizer options (sklearn never passes them).
