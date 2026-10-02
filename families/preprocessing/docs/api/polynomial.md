# PolynomialFeatures, SplineTransformer

Conventions: [api.md](../api.md). Source: `src/moxie/preprocessing/_polynomial.mojo`;
`linspace` and `_weighted_percentile` live in the shared `_percentiles.mojo`.

| Name | Native signature |
| --- | --- |
| `PolynomialFeatures[dtype]` | `PolynomialFeatures(degree=2, *, interaction_only=False, include_bias=True, order="C")`; overload `PolynomialFeatures(degree: Tuple[Int, Int], ...)` for `(min_degree, max_degree)` |
| `SplineTransformer[dtype]` | `SplineTransformer(n_knots=5, degree=3, *, knots="uniform", extrapolation="constant", include_bias=True, order="C", handle_missing="error", sparse_output=False)`; overload with `knots: NDArray[float64]` of shape (n_knots, n_features) |

Both take a rank-2 `NDArray[dtype]` (Float32 or Float64, any strides) and
return a new `NDArray[dtype]`. `order="F"` returns a column-major array;
`"C"` (the default) returns a row-major one.

```mojo
from numojo import NDArray, Shape, zeros
from moxie.preprocessing import PolynomialFeatures, SplineTransformer

var X = zeros[DType.float64](Shape(3, 2))   # fill with data
var poly = PolynomialFeatures(2)
var P = poly.fit_transform(X)                # columns 1, x0, x1, x0^2, x0 x1, x1^2
var names = poly.get_feature_names_out()     # ["1", "x0", "x1", "x0^2", "x0 x1", "x1^2"]
var powers = poly.powers_()                  # Int64 (6, 2)
var ranged = PolynomialFeatures((2, 3), interaction_only=True)

var spline = SplineTransformer(n_knots=4, degree=3, extrapolation="periodic")
spline.fit(X, sample_weight=w)               # weights affect knot placement only
var S = spline.transform(X)                  # 2 features * 3 periodic splines
var knots = spline.bsplines_[0].t            # Float64 knot vector of feature 0
```

## Behavior

**PolynomialFeatures.** The output columns follow sklearn's order: bias, then
`itertools.combinations_with_replacement` (or `combinations` when
`interaction_only`) for each degree. Each column is the product of an earlier
column and one input column, computed in the input dtype in sklearn's order.
Values therefore match sklearn bit for bit, in Float32 as well. Degree
validation and its error messages match sklearn, including the check that
the output has more columns than int64 can index. That check counts the
columns exactly, with arbitrary-precision integers like Python's. Feature
names use sklearn's format: `x0^2 x1`, and `1` for the bias column.

**SplineTransformer.** Knots are placed as in sklearn:
- `"uniform"` spaces knots evenly from the minimum to the maximum. NaNs and
  zero-weight rows are ignored.
- `"quantile"` uses `np.nanpercentile(method="linear")`, or sklearn's
  `_weighted_percentile` when `sample_weight` is given.
- An explicit array is validated like sklearn's (at least 2 rows, one column
  per feature, strictly increasing).

`degree` extra knots are then added on each side. Periodic splines continue
the spacing of the opposite end. Basis values come from a port of SciPy
1.18.1's `_find_interval` and `_deBoor_D` recursion, evaluated in Float64.
All five `extrapolation` modes are supported:
- `constant` uses the boundary values.
- `linear` uses boundary values plus first-derivative slopes.
- `continue` evaluates the polynomial pieces outside the range.
- `periodic` maps x into the period with NumPy's `remainder` (exact `fmod`).
- `error` raises `` `X` contains values beyond the limits of the knots. ``

`include_bias=False` drops the last spline of each feature. Feature names are
`x0_sp_0, ...`. With `handle_missing="zeros"`, a NaN cell encodes as an
all-zero spline block. As in sklearn, NaN and infinity checks are then
skipped entirely. `sample_weight` may be an `NDArray[dtype]` or a `Float64`
scalar, and is validated like sklearn's `_check_sample_weight`.

Learned attributes: `bsplines_` (`List[BSpline]`), `n_features_in_`,
`feature_names_in_` and `n_features_out_`.

## Native adaptations

* **Tuple degree and knot arrays.** `degree=(min, max)` is a constructor
  overload taking `Tuple[Int, Int]`, stored in the `degree_range` field.
  `degree` then holds `max`. Knot arrays use an overload storing
  `knot_positions`. Both appear in the `Params` records. `set_params(degree=2)`
  or `set_params(knots="quantile")` clears the alternative form. Float degrees
  and 3-tuples are unrepresentable, so those sklearn errors cannot occur.
* **`powers_()` is a method.** It computes the powers on demand, as sklearn's
  property does. This keeps the fitted state at O(degree × n_features) instead
  of O(n_output × n_features).
* **`bsplines_[j]` is a `BSpline` record, not a SciPy object.** It has:
  - `t`, the Float64 knots;
  - `k`, the degree;
  - `extrapolate` and `periodic` flags;
  - `n_splines()`, which is `c.shape[1]`;
  - `coefficients()`, which builds sklearn's identity-like `c` (wrapped for
    periodic splines);
  - `evaluate(x, nu=0)`, with SciPy's `spl(x, nu)` semantics.
* **Configuration is snapshotted at fit.** `transform` and the feature names
  use the configuration recorded at `fit`. sklearn reads `extrapolation`,
  `include_bias`, `order`, `handle_missing` and `interaction_only` again at
  transform time. Natively, `set_params` takes effect at the next fit.
* **`sparse_output=True` raises `ValueError` at fit.** Sparse output is deferred.
  Sparse PolynomialFeatures input is unavailable by type.
* **Float32 quantile knots with `sample_weight`.** sklearn computes these knots
  in Float32 and stores Float32 knot vectors, which SciPy then rejects at
  `transform` (`ValueError: Expected a 1-dim C contiguous array of dtype`).
  Natively, the knots are the same Float32 values widened to Float64, and
  `transform` works.
* **sklearn defects not reproduced:**
  - With `extrapolation="linear"` and `degree <= 1`, sklearn increments its
    loop-shared `degree` once per feature. Every feature after the first then
    uses shifted boundary knots. Natively, each feature uses its own degree,
    which equals fitting that feature alone in sklearn.
  - With `degree=0` and `extrapolation="constant"`, a value above the range
    makes sklearn fail on a broadcast (`f_max[-0:]`). Natively, the row is all
    zero, the same as sklearn's encoding below the range.
  - When an expansion has zero output columns, sklearn's `powers_` and
    `get_feature_names_out` raise. Natively, they return an empty
    `(0, n_features)` array and an empty list. An example is
    `interaction_only` with `min_degree` greater than `n_features` and no bias.

## Memory

PolynomialFeatures keeps O(degree × n_features) fitted state: a list of
multiplication blocks. A transform allocates the result plus one row
workspace of the full expansion width. It never allocates sklearn's
intermediate full-width matrix for `min_degree > 1`.

SplineTransformer keeps one knot vector per feature, O(n_features × (n_knots +
2·degree)). Fitting needs O(n_samples) workspace per feature for quantiles.
A transform allocates the result plus O(n_splines) scratch. Inputs are read
in place through their strides.

## Parity

`tests/parity/test_polynomial.py` runs 4,925 checks. All of them compare
exactly (`rtol=0, atol=0`), in Float32 and Float64:

| Area | What is covered |
| --- | --- |
| PolynomialFeatures | Degrees 0–4 and several `(min, max)` tuples; `interaction_only` and `include_bias` both ways; C and F order; outputs, `n_output_features_`, `powers_` and names. Inputs: one, three and five features; strided and negative-stride input; named features; refit. |
| SplineTransformer | Degrees 0–4 with 2–6 knots; uniform, quantile and explicit knots; all five extrapolation modes, with test points inside, at and beyond the range. Outputs, every knot vector, `c`, `n_features_out_` and names. |
| Weights, missing values, edge cases | Zero, integer, scalar and negative sample weights; `handle_missing="zeros"` with NaN, all-NaN columns and infinity; constant features; sklearn's docstring and base-knot examples. |
| Errors | 35 explicit error cases, plus `error`-mode transforms beyond the knots. |
| Primitives | `BSpline.evaluate` for nu = 0, 1, 2 against SciPy; `_fmod` and `_remainder` against NumPy, including signed zeros, subnormals and 1e300. |

Exact agreement is expected:
- Each polynomial column is one IEEE product in the input dtype, in sklearn's
  order.
- Spline code ports SciPy's recurrences and NumPy's `linspace`, `nanpercentile`
  and `remainder` arithmetic operation by operation. Accumulations run in
  SciPy's order.
- Float32 spline output is the same Float64 value, rounded once.

The parity driver prints Float32 values through Float64, because Mojo 1.0's
Float32 text is not always round-trip exact. The other preprocessing parity
groups pass unchanged with this fix.

Error messages match sklearn's prefix and fields. Messages that list choices
use a fixed sorted order; Python prints a set in hash order.
