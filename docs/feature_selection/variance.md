# Dense VarianceThreshold

`VarianceThreshold` removes columns whose population variance is not strictly
larger than `threshold` (default `0.0`). It is available from the root assembled
`moxie.feature_selection` package and the univariate consumer assembly.

```mojo
import numojo as nm
from moxie.feature_selection import VarianceThreshold

var X = nm.zeros[DType.float32](nm.Shape(3, 2))
X.unsafe_set(1, 1)
X.unsafe_set(3, 2)
X.unsafe_set(5, 3)
var selector = VarianceThreshold[DType.float32]()
var selected = selector.fit_transform(X)  # (3, 1), Float32
var variances = selector.variances_()     # (2,), Float64
```

The default compile-time dtype is Float64. Fit and transform accept rank-2
Float32/Float64 NuMojo arrays with the estimator's dtype, including strided views.
Statistics use Float64 regardless of input dtype. Fit ignores NaNs in each column;
all-NaN columns have NaN variance and are removed. Infinity, empty inputs, negative
or nonfinite thresholds, and fits with no finite variance above threshold raise.
At threshold zero the reference's range cap is preserved, including its behavior
for large-range features. Transform preserves NaNs and the input dtype;
inverse_transform rejects nonfinite values, matching the reference's validation.
Both mapping operations delegate to the family SelectionState.

`get_support(indices=False)`, `transform`, `inverse_transform`,
`fit_transform`, `get_feature_names_out`, configuration cloning, `get_params`
and `set_params` are implemented. Changing `threshold` updates support from the
stored variances without refitting, including an empty selection. Clone returns
an unfitted estimator. Returned variances, masks, indices and names are owned.

Native adaptations follow the family conventions: fit returns a mutable borrow
of the fitted selector, learned attributes are accessors (`variances_()`, `n_features_in_()`,
`feature_names_in_()`), parameter dictionaries become a typed record, and
`fit_named(X, y, names)` supplies names explicitly. The ignored `y` accepts an
optional native array. A failed fit preserves the previous fitted state; sklearn
can leave partially updated attributes after failure. This transactional native
behavior is intentional and tested. Python warnings, metadata routing, dataframe
output, integer/object arrays and cross-dtype transform calls are not provided.
Sparse input remains deferred; no implicit densification is performed.

The implementation composes approved NuMojo sum, finite subtraction and
multiplication on one compact Float64 column at a time. It does not call
NuMojo variance/ddof or use a replacement array container. This is a correctness
baseline, with no performance claim.

Run `pixi run build`, `pixi run test`, and `pixi run parity` at the repository
root. Both isolated consumers exercise Float32 selection. The parity harness
executes the dense fit and support-mask bodies from local scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`, with a small dense validation adapter.
It covers both dtypes, reversed row/column views, missing values, constants,
threshold equality and adjacent representable thresholds, single rows, infinity
rejection and large offsets. Padded views use NaN sentinels outside the logical
input. Extreme fixtures exercise squared-difference overflow, sum overflow,
underflow, and the range cap with and without missing values. Float32 conversion
of the largest fixtures overflows to infinity and must be rejected.

Masks, transformed values and zero results compare exactly; other variances
allow rtol=1e-10 and atol=1e-12 for reduction-order differences. NaN and infinity
locations must match. The runner compiles bounded batches under the shared
compiler limits and saves generated sources, expected values and native output
under `builds/variance-parity-source`. After assembly, run the focused comparison
with `pixi run python tests/parity_variance.py`. These fixtures do not certify
arbitrary ill-conditioned inputs or full Python estimator parity.
