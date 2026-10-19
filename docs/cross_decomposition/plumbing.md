# Cross-decomposition ownership and contracts

`src/moxie/cross_decomposition/` is the sole active implementation owner.
`__init__.mojo` and the assembly manifest export concrete estimators from `_pls`
and `_plssvd`; no `_api` or `_api_types` scaffold participates in assembly.

| Module | Responsibility |
| --- | --- |
| `_base` | Paired transformer and optional regression traits |
| `_contracts` | Float64 Matrix, block validation, owning result records, TwoBlockEngine |
| `_pls` | PLSParams, PLSRegression, PLSCanonical, CCA, PLSEngine and shared numerical operations |
| `_plssvd` | PLSSVDParams and PLSSVD, consuming PLSEngine |

Moxie owns framework traits and validation. `shared/numerics` owns `_dense`;
external NuMojo owns array storage and accepted operations. External mojo-linalg
owns SVD. Earlier family documents named decomposition's unfinished SVD engine;
the implemented PLS code instead imports `svd` and `SVDFactor` from mojo-linalg.
No decomposition backend is required by cross decomposition.

`Matrix` is `NDArray[DType.float64]`. Sample rows and target columns are
validated together. `OwnedTarget` normalizes vector targets and remembers their
rank. `DenseBlocks`, `PairedScores`, `Scaling`, `WeightStep`, `Loadings` and
`ProjectionWeights` own their array fields. Fitted estimator copies own
independent learned buffers. Mutable input aliases must not overlap.

`TwoBlockEngine` requires centering/scaling, leading weights, deflation,
projection weights and rotations. One PLSEngine implements all five:

- Scaling uses sample standard deviation with ddof=1 and substitutes one for
  zero scales. Disabling scaling still centers both blocks.
- Mode B uses the legacy pseudoinverse threshold `max(s) * 1e6 * eps64`.
  Rotations use `max(shape) * eps64 * max(s)` instead.
- NIPALS starts at the first nonconstant target column. It stops on squared
  weight difference below tolerance or a single target. Residual columns below
  `10 * eps64` are masked; constant targets stop extraction with remaining
  components zero-filled. Iteration exhaustion records a diagnostic.
- SVD leading weights and PLSSVD projection weights share the same factorization
  adapter and sign convention: the first largest-absolute X weight is positive.
- Regression deflates y using X scores; canonical deflation uses Y scores.
  Rotations use `weights @ pinv(loadings.T @ weights)`.

The migration changes ownership and fluent mutation signatures. It preserves
numerical operation order and backend calls. The child parity suites retain
exact comparisons where defined and objective/subspace comparisons where the
reference itself is sign-degenerate or perturbation-sensitive.

Consumers assemble current sources through `DEPENDENCIES.json`. They must use
Moxie's modules and public exports instead of adding old family/child projects.
Parameter records and engine/result contracts remain available from their
original module names. No historical framework copy is needed. Consumers that
require mutation with `copy=False` continue to call `fit_inplace`.
