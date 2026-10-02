# Normalizer, Binarizer, KernelCenterer and helpers

Conventions: [api.md](../api.md). Source: `src/moxie/preprocessing/_simple.mojo`.

| Name | Native signature |
| --- | --- |
| `Normalizer[dtype]` | `Normalizer(norm="l2", *, copy=True)`; norm in l1, l2, max |
| `Binarizer[dtype]` | `Binarizer(*, threshold=0.0, copy=True)` |
| `KernelCenterer[dtype]` | `KernelCenterer()`; learned `K_fit_rows_` (vector), `K_fit_all_` (scalar) |
| `normalize` | `normalize(X, norm="l2", *, axis=1, copy=True, return_norm=False) -> NDArray` |
| `binarize` | `binarize(X, *, threshold=0.0, copy=True) -> NDArray` |
| `add_dummy_feature` | `add_dummy_feature(X, value=1.0) -> NDArray[float64]` |

```mojo
from moxie.preprocessing import Normalizer, normalize
from moxie.preprocessing._simple import normalize_with_norms

var unit_rows = normalize(X, "l1")
var result = normalize_with_norms(X, "l2", axis=0)   # (X_normalized, norms)
var model = Normalizer("max")
model.transform_inplace(X)                            # no allocation
```

Behavior follows sklearn: zero-norm vectors (norm below 10*eps) are left as is
and their returned norm is 1; Binarizer maps values strictly greater than the
threshold (cast to X's dtype) to 1. NaN and infinity are rejected. Normalizer and
Binarizer are stateless: `transform` works before `fit`, and after `fit` checks
the feature count. KernelCenterer requires a square training kernel and centers
`K(X_test, X_fit)` matrices with `n_fit` columns; feature names are
`kernelcenterer0, ...`. `add_dummy_feature` returns Float64 for Float32 input,
like sklearn's dense `hstack`.

Native adaptations: `normalize(..., return_norm=True)` raises because the
result type would change; `normalize_with_norms` returns the tuple.
`normalize_inplace`, `binarize_inplace` and each estimator's
`transform_inplace(mut X)` implement copy=False without allocating.

Memory: copying transforms allocate the result only; norms and centering use
O(1) (Normalizer/Binarizer) or O(n_fit) (KernelCenterer) extra storage.

Parity (`tests/parity/test_simple.py`): Float32/Float64 Gaussian, zero-row,
subnormal, wide and single-element inputs; every norm/axis pair with returned
norms; three thresholds; dummy values; linear-kernel centering of training and
test kernels; NaN, invalid norm, non-square and unfitted errors. Tolerances:
Float64 rtol 1e-12, Float32 rtol 4e-6 with atol 1e-6 for cancelling kernels,
because reduction order differs from NumPy's pairwise/SIMD loops.
