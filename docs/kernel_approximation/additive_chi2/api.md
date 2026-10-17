# Dense additive chi-squared feature map

Import `AdditiveChi2Sampler` from `moxie.kernel_approximation`. It implements the
family's `KernelFeatureMap` contract on NuMojo matrices. See
[the executable example](../../../tests/kernel_approximation/additive_chi2/example.mojo).

| Estimator | Constructor keywords and defaults |
| --- | --- |
| `AdditiveChi2Sampler[dtype=DType.float64]` | `sample_steps: Int=2`, `sample_interval: Optional[Float64]=None` |

For each nonnegative entry `x` and interval `L` the map produces
`sqrt(x L)` and, for `j = 1 .. sample_steps-1`,
`sqrt(2 x L / cosh(pi j L)) * cos(j L log x)` and the matching `sin`.
Output shape is `(n_samples, n_features * (2*sample_steps - 1))`; columns are
the blocks `[sqrt | cos1 | sin1 | cos2 | sin2 | ...]`, each `n_features` wide,
exactly like sklearn's `np.hstack`. Zero entries map to zero in every block.
With `sample_interval=None` the defaults are 0.8, 0.5, 0.4 for 1, 2, 3 steps;
other step counts require an explicit interval. Interval 0 is accepted and
maps everything to zero, as in sklearn.

The estimator is stateless: `transform(X)` works without `fit`. `fit(X, y=None)`
validates the parameters and nonnegative input and records `n_features_in_`;
after that, `transform` also rejects a different feature count. Input and
output share the compile-time Float32/Float64 dtype; valid strided views are
accepted. Errors are Mojo `Error`s with `ValueError`/`NotFittedError` prefixes
naming the estimator and offending value (negative or nonfinite input, invalid
`sample_steps`/`sample_interval`, missing interval for 4+ steps).

Language adaptations (tested in `tests/kernel_approximation/additive_chi2/contracts.mojo`):

- `transform` re-validates parameters. sklearn validates them only in `fit`, so
  an invalid `sample_steps` set after construction reaches its transform
  arithmetic; here it raises instead of returning a malformed matrix.
- `get_params()` returns a typed `AdditiveChi2SamplerParams` record;
  `set_params(record)` replaces it and `set_params(sample_steps=..., sample_interval=...)`
  updates named fields. Resetting the interval to None needs a full record.
- `clone(model)` copies configuration into an unfitted estimator.
- `get_feature_names_out(input_features=None)` requires `fit` (sklearn checks
  `n_features_in_`) and returns a native `List[String]` of
  `additivechi2sampler_<name>_sqrt`, `_cos<j>`, `_sin<j>` in column order;
  names default to `x0 .. x{p-1}`. Pandas feature names are binding work.
- `__sklearn_is_fitted__()` reports whether `fit` was called; sklearn's
  `requires_fit=False` tag corresponds to transform not checking it.

Numerics: Float32 follows NumPy's dtype promotion: `sqrt`, `log` and the phase
`j L log x` are computed in the input dtype, the step factor
`sqrt(2 x L / cosh(pi j L))` in Float64 (NumPy's scalar `cosh` is Float64), and
the product is rounded to the input dtype. Float64 `log` and `cosh` use the
LLVM intrinsics (platform libm) because Mojo 1.0's `std.math` versions
measured up to 1.7e7 and 1e5 ulp away from libm.
Time is O(n p s) and the only allocation is the O(n p (2s-1)) output.

## Validation

`pixi run parity` executes the unedited `fit`, `transform`, `_transform_dense`
and `get_feature_names_out` bodies from the pinned local sklearn source
(revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`, cleanliness and code
filename verified) with NumPy validation stubs, over tall/wide/tiny/
sparse-pattern/wide-range/all-zero data, steps 1-5 with default and explicit
intervals, in Float32 and Float64, plus parameter/domain error cases and the
kernel approximation check of sklearn's test suite. Tolerance:
`|native - sklearn| <= 16 * eps(dtype) * max(1, max|sklearn|)` per case (a few
ulp of the output scale for a short chain of elementary functions); feature
names must be identical. Reports are written under ignored `docs/private/`.
