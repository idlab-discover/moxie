# StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler and helpers

Conventions: [api.md](../api.md). Source: `src/moxie/preprocessing/_scalers.mojo`.

| Name | Native signature |
| --- | --- |
| `StandardScaler[dtype]` | `StandardScaler(*, copy=True, with_mean=True, with_std=True)` |
| `MinMaxScaler[dtype]` | `MinMaxScaler(feature_range=(0, 1), *, copy=True, clip=False)` |
| `MaxAbsScaler[dtype]` | `MaxAbsScaler(*, copy=True, clip=False)` |
| `RobustScaler[dtype]` | `RobustScaler(*, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True, unit_variance=False)` |
| `scale` | `scale(X, *, axis=0, with_mean=True, with_std=True, copy=True) -> NDArray` |
| `minmax_scale` | `minmax_scale(X, feature_range=(0, 1), *, axis=0, copy=True) -> NDArray` |
| `maxabs_scale` | `maxabs_scale(X, *, axis=0, copy=True) -> NDArray` |
| `robust_scale` | `robust_scale(X, *, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True, unit_variance=False) -> NDArray` |

```mojo
from moxie.preprocessing import StandardScaler, RobustScaler, minmax_scale

var scaler = StandardScaler()
scaler.partial_fit(batch1)
scaler.partial_fit(batch2, sample_weight=w2)   # NDArray weights, or a Float64
var Z = scaler.transform(X)
print(scaler.n_samples_seen_.value())          # raises if counts differ by feature
scaler.transform_inplace(X)                    # copy=False: no allocation

var rows = minmax_scale(X, (-1.0, 1.0), axis=1)
var robust = RobustScaler[DType.float32](unit_variance=True)
```

## Behavior

Learned attributes follow sklearn's dtypes: StandardScaler `mean_`, `var_` and
`scale_` are Float64 (and `None` when the corresponding statistic is not
computed); MinMaxScaler and MaxAbsScaler attributes keep X's dtype;
RobustScaler `center_` keeps X's dtype and `scale_` is Float64. NaN is ignored
when fitting and kept when transforming; infinity is rejected. Near-constant
features (scale below 10*eps) are scaled by 1. `scale` uses input-precision
statistics with sklearn's two residual-mean correction passes. Signed sample
weights are accepted; all-zero weights raise.

## Native adaptations

* `n_samples_seen_` is a `SampleCount` record: `values` holds one Float64 count
  per feature, `is_scalar` is True when sklearn would store a scalar (counts
  agree across features), and `is_integer` marks sklearn's int64 count (neither
  centering nor scaling, unweighted). `value()` returns the scalar.
  MinMaxScaler and MaxAbsScaler keep sklearn's `Int` count.
* Weights are an `NDArray[dtype]` overload or a scalar `Float64` overload of
  `fit`/`partial_fit`; a scalar is never expanded to a vector.
* `copy` is recorded but borrowed inputs are never modified;
  `transform_inplace` / `inverse_transform_inplace` implement copy=False.
  `transform(X, copy)` is accepted for parity.
* StandardScaler `partial_fit` raises `ValueError` if `with_mean`/`with_std`
  change between calls in a way that makes the stored moments incompatible
  (sklearn fails or ignores it); call `fit` to restart.
* The functions accept 1-D vectors (axis 0 only for `scale`) and return the
  input's rank.

## Memory

Measured by `pixi run memory` (`algorithms/scalers/tests/memory/acceptance.py`),
for n rows, p features and d bytes per element:

| Operation | Additional live storage |
| --- | --- |
| Every `fit`/`partial_fit` except RobustScaler, weighted or not | O(p) state |
| RobustScaler fit | O(p) state plus one 8*n-byte Float64 column |
| Copying transforms, inverses and axis-0 functions | the n*p*d result plus O(p) |
| `*_inplace` methods | O(p) state |
| `scale(axis=1)` | result plus three n-element vectors |
| `minmax_scale` / `maxabs_scale` / `robust_scale` with axis=1 | result plus two n-element vectors (Float64 for robust) |

Masks, casts, broadcast weights and residual tables are never materialized,
and axis=1 functions read borrowed transposed views. The gate allows 32 MiB of
fixed overhead, because the Mojo runtime keeps up to about 24 MiB of freed blocks
resident. It also rejects a deliberate full-table temporary.

## Parity

`tests/parity/test_scalers.py`, 1407 checks: Float32/Float64 Gaussian,
ill-conditioned, NaN, all-NaN column, constant, single-row, overflow and tall
inputs; every with_mean/with_std pair; batched, weighted, signed-weight and
scalar-weight partial fits; Fortran-order and negative-stride views; three
feature ranges with clipping; five robust configurations; all functions on both
axes and on vectors; and invalid-input, unfitted and feature-count errors.
Tolerances: learned statistics rtol 1e-11. Transforms use Float64 rtol 1e-11
and Float32 rtol 2e-6. Weighted transforms add, per column, the
recursive-summation bound n*eps*sum|w x|/|sum w|/scale, because sklearn's
weighted sum is a BLAS matmul whose order and FMA use depend on the BLAS build.
