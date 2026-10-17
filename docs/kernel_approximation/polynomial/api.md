# Dense polynomial count sketch

Import `PolynomialCountSketch` from `moxie.kernel_approximation`. It implements the
family's `KernelFeatureMap` contract. See [the example](../../../tests/kernel_approximation/polynomial/example.mojo).

| Estimator | Constructor keywords and defaults |
| --- | --- |
| `PolynomialCountSketch[dtype=DType.float64]` | `gamma: Float64=1.0`, `degree: Int=2`, `coef0: Float64=0.0`, `n_components: Int=100`, `random_state: Optional[Int]=None` |

Tensor Sketch approximates the feature map of `(gamma <x, y> + coef0)^degree`.
`fit` draws `indexHash_` (all `randint(0, n_components)` values in C order)
and then `bitHash_` (`choice([-1, 1])`, i.e. `randint(0, 2)` mapped to -1/+1),
each an Int64 array of shape `(degree, n_features + (coef0 != 0))`, from the
shared sklearn-compatible `RandomState`; the data is only validated. `transform`
builds, per sample and degree, the signed count sketch of
`[sqrt(gamma) x, sqrt(coef0)]`, multiplies the `degree` spectra
(mojo-linalg `FFTPlan`, radix-2 or Bluestein for any `n_components`), inverse
transforms and keeps the real part. Output is **Float64 for both input dtypes**
(sklearn's `count_sketches` buffer is float64), shape `(n_samples, n_components)`.

Language adaptations (tested in `tests/kernel_approximation/polynomial/contracts.mojo`):

- `random_state=None` requires `fit_with_rng(X, context)` with a mutable
  shared `RandomState` (for example `moxie._random.entropy_state()`); integer
  seeds rebuild the stream on each `fit`.
- Negative `coef0` is accepted by `fit` (as in sklearn), but `transform`
  raises `ValueError`: sklearn evaluates `np.sqrt(coef0) = NaN` and returns an
  all-NaN matrix. Parity asserts exactly this correspondence.
- Shrinking `n_components` after fit so that a stored hash is out of range
  raises (sklearn raises an IndexError); changing `coef0` between zero and
  nonzero after fit raises the reference feature-count error.
- Typed `get_params()`/`set_params(record)`, named partial `set_params`,
  `clone` (unfitted) and native `List[String]` feature names
  `polynomialcountsketch<k>` of the fitted width.

Cost: O(n * degree * (p + M log M)) time with M = n_components (power of two)
or the Bluestein length (next power of two >= 2*n_components-1); O(M) scratch
plus the O(n * n_components) output. The per-row loop is single-threaded.

## Validation

`pixi run parity` executes the unedited pinned sklearn `fit`/`transform` bodies
(revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`, with `scipy.fft` as the
module imports it) on 394 cases: tiny/tall/wide/zero data, Float32 and Float64,
degrees 1-5, widths 1-128 (powers of two and not), coef0 0/2.5, gamma 0, seeds
up to 2^32-1, and negative coef0. Hashes and the next RNG draw must be
identical; outputs must satisfy
`|native - sklearn| <= 16 * degree * eps64 * log2(M) * max(1, ||x'||_1)^degree`
per row (observed worst 0.022 of the bound). It also runs the full
`test_polynomial_count_sketch` grid (18 settings) with sklearn's thresholds.
