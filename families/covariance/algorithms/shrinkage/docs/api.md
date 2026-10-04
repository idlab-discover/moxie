# Shrinkage API

```mojo
from moxie.covariance import LedoitWolf, OAS, ledoit_wolf, ledoit_wolf_shrinkage, oas

var lw = LedoitWolf(block_size=1000)   # [dtype=DType.float64]
lw.fit(X)
var s = lw.shrinkage_()                # Float64 in [0, 1]
var C = lw.covariance_()               # (1 - s) S + s tr(S)/p I
var pair = ledoit_wolf(X)              # Tuple[NDArray, Float64]
var alpha = ledoit_wolf_shrinkage(X, assume_centered=False, block_size=1000)
var o = oas(X)                         # Tuple[NDArray, Float64]
```

| sklearn | Mojo |
| --- | --- |
| `LedoitWolf(*, store_precision=True, assume_centered=False, block_size=1000)` | same keywords; `LedoitWolf[dtype]` |
| `OAS(*, store_precision=True, assume_centered=False)` | same |
| `shrinkage_`, `location_`, `covariance_`, `precision_`, `n_features_in_` | accessor methods (`precision_()` is Optional) |
| inherited `get_precision`, `score`, `mahalanobis`, `error_norm` | same, via the shared `CovarianceModel` |
| `ledoit_wolf(X, *, assume_centered, block_size)` / `oas(X, *, assume_centered)` | return `Tuple[NDArray[dtype], Float64]` |
| `ledoit_wolf_shrinkage(X, assume_centered=False, block_size=1000)` | returns Float64 |

`block_size` must be >= 1 and only partitions the Gram accumulation.
One feature gives shrinkage 0 and covariance mean(X**2) (after centering).
A single sample gives Ledoit-Wolf shrinkage 0 and OAS shrinkage 1 (zero denominator).

Float32 estimators preserve Float32 centering, products, blocked Ledoit–Wolf
moments and coefficient arithmetic. OAS widens its scalar formula only after
squaring and reducing the Float32 empirical covariance. The one-feature branch
also squares and averages in the input dtype. Overflow can produce a NaN
coefficient and a rejected nonfinite covariance, just as in the reference;
no hidden widening or clipping makes these fits appear successful.

The native matrix product and BLAS can still differ in low bits. Pairwise sums
use the reference's eight-stream reduction order; block sizes can affect
Float32 rounding. The boundary suite uses exact outcome/nonfinite comparisons,
exact coefficients for binary cancellation fixtures, and explicit scale-aware
finite tolerances. This is not a bitwise BLAS compatibility guarantee.

The internal `shrinkage_statistics` / `AdaptiveShrinkageEngine` retain their
widened Float64 moment contract for native engine consumers. Their rounded
Float32-input coefficients need not equal the public sklearn arithmetic path.
Errors are Mojo `Error`s with sklearn-style prefixes.

Single-sample warnings use category-prefixed stderr. Numerical failures can
leave updated location and shrinkage with the previous covariance/precision,
following sklearn's assignment order. On a first numerical failure, location,
feature count and an assigned shrinkage remain accessible; covariance-dependent
methods still reject the unfitted estimate. Invalid inputs/parameters preserve estimates. These differences are covered by the [family audit](../../../docs/parity.md).
