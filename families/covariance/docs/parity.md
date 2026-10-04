# Covariance behavior and validation

Reference: scikit-learn `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
All 16 public names are implemented. This is dense native support, not full
scikit-learn behavioral parity. Algorithm API guides retain detailed signatures:
[empirical](../algorithms/empirical/docs/api.md),
[shrinkage](../algorithms/shrinkage/docs/api.md),
[robust](../algorithms/robust/docs/api.md),
[graphical](../algorithms/graphical/docs/api.md).

## Supported behavior

All eight estimators have typed configuration records, keyword partial updates,
unfitted cloning, independent learned snapshots, repeat fits and learned-state
accessors. Fit/set_params return a mutable borrow of the original estimator.
Nullable updates distinguish omission from explicit None. Parameter changes
retain learned state until refitting. Contracts cover Float32/Float64, strided
inputs, validation, unfitted errors, shapes, defaults and ownership.

Empirical and shrinkage estimators implement precision caching, Mahalanobis,
Gaussian score and error norms. Both robust estimators expose correction and
reweighting; EllipticEnvelope adds fit_predict, signed labels, decision values
and weighted accuracy. GraphicalLasso and GraphicalLassoCV support coordinate
descent and LARS, alpha zero, cost histories and convergence diagnostics. CV
supports explicit grids and materialized custom splits with typed results.

## Limits callers must retain

- Dense Float32/Float64 only. Static array dtypes differ from option-dependent
  NumPy promotion. Python metadata routing, feature names, reflection and
  bindings are unsupported. Covariance persistence is not implemented.
- `fast_mcd` custom covariance callables and parallel GraphicalLassoCV are
  unsupported. Robust comparison suites normalize reference unstable tie
  ordering; the audit deliberately does not.
- Diagnostics are `Category: message` stderr lines, not Python warning objects.
  Python warning filters, stack locations and warning-as-error are unavailable.
  Graphical verbose iterations use stdout; CV refinement uses stderr. CV
  suppresses convergence warnings during search and reports the final solve.
- Invalid input/parameters preserve fitted state. Numerical failed fits publish
  auxiliary state in sklearn computation order and can leave old covariance
  alongside new location, shrinkage, robust raw state or CV results. Inference
  rejects mismatched feature counts and missing covariance; refit before use.
- Rank-deficient and near-singular likelihoods can differ in finiteness and
  value. Nonsymmetric precision uses pivoted QR; symmetric precision uses LDLᵀ.
  Determinant sign <= 0 yields -inf, without a rank cutoff. Alpha-zero inversion
  tries Cholesky then LDLᵀ; it must not be replaced with a pseudoinverse.
- The graphical solver computes in Float64 and retains widened precision for
  inference, while public arrays retain the static dtype. Some reference
  Float32 CD paths fail with buffer-dtype errors; native success is an extension.
  Widened-oracle diagnostics are not counted as original-dtype parity.
- Float32 shrinkage follows reference intermediates, including overflow and
  partial-state publication; BLAS product rounding can still differ. Error norms
  require equal matrix shapes. Likelihood permits broadcastable rank-2 operands
  with square precision, but not vectors or higher ranks. Enabling precision
  caching after a cache-free fit recomputes natively where sklearn returns None.

## Reproducible gates and observations

Run from the Moxie root:

```sh
pixi run covariance-test
pixi run covariance-parity
pixi run covariance-audit
pixi run covariance-numerical-audit
# After building; intentionally fails on any observed difference:
pixi run python families/covariance/tests/audit.py --strict
pixi run python families/covariance/tests/numerical_focus.py --strict
```

`covariance-test` compiles the full namespace, all algorithm contracts, fluent
and parameter contracts, both isolated consumers and negative compile/assembly
probes. It checks diagnostic channels, failed-refit state, 126 nonsymmetric
likelihood cases and singular-boundary fixtures.

`covariance-parity` runs all four algorithm suites (including graphical LARS
and CV), keyword/refit comparisons, and the strict repaired-gap gate. Tolerances,
conditioning allowances and any reference adaptations are stated in each
`algorithms/*/tests/parity.py`; rank-deficient score checks are not numerical
parity claims. The repaired-gap gate requires Float32 shrinkage overflow and
first-fit/refit state, invalid-input preservation and diagnostic categories to
match without widening those reference inputs.

The source-family audit recorded 102 cases, 2,889 comparisons and 38 differing
observations: 28 reference Float32 CD failures/resulting state differences and
10 ill-conditioned inverse/cost/singular-score observations. Observations are
not distinct defects. Normal audit completion is not a parity pass; strict mode
rejects every difference. Raw inputs, warnings, environment and results remain
under ignored `builds/` directories, never tracked as development worklogs.

`tests/singular_boundaries.py` crosses native/reference precision with both
determinant evaluators and records exact rational signs of stored matrices,
condition numbers, relative inverse errors and scaled residuals. Exact binary
singular/sign fixtures and well-conditioned inverses are gates. Small residuals
at condition numbers near 1e16 do not establish forward or score agreement.
Cholesky reduced the audited Float64 alpha-zero inverse error from about 1.8%
to 1.6e-11, while costs still differ near condition number 5.9e14. A future LU
backend belongs to the numerical owner and must pass these identical-matrix
probes; its algorithm name alone cannot promise LAPACK rounding parity.
