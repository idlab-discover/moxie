# NeighborhoodComponentsAnalysis

`moxie.neighbors.NeighborhoodComponentsAnalysis` learns a linear embedding by
maximizing expected leave-one-out nearest-neighbor classification accuracy.
Its source is `algorithms/nca/src/moxie/neighbors/_nca.mojo`; the private
`_nca_math` module contains its objective and initialization compositions.
`_nca_lbfgs` and `_nca_linesearch` implement its private unbounded optimizer.
It consumes the family's distance/target helpers and existing shared storage,
reductions, RNG and native SVD dependencies. Training and transformation are pure
Mojo. See [the runnable example](../examples/nca.mojo).

```mojo
from moxie.neighbors import NeighborhoodComponentsAnalysis

var nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
nca.fit(X, y)
var embedded = nca.transform(X)
var names = nca.get_feature_names_out()
```

## Constructor and methods

The constructor preserves `n_components=None`, `init="auto"`,
`warm_start=False`, `max_iter=50`, `tol=1e-5`, `callback=None`, `verbose=0`, and
`random_state=None` semantics subject to the native adaptations below.
`init` accepts `auto`, `pca`, `lda`, `identity`, `random`, or a Float64 matrix.
An explicit matrix is independently copied. Auto chooses LDA when the requested
output dimension does not exceed `min(n_features, n_classes - 1)`, otherwise PCA
when it is below `min(n_features, n_samples)`, otherwise identity.

Methods are `fit(X,y)`, `transform(X)`, `fit_transform(X,y)`,
`get_feature_names_out(input_features=None)`, `get_params(deep=True)`,
`set_params(params)`, named partial `set_params`, and the shared `clone` helper.
There is no classifier `predict`, `score`, or neighbor-query capability.
The estimator implements SupervisedEstimator, TransformerMixin and FitTransformer.
Fitted attributes are `components_`, `n_features_in_`, `n_iter_`, and
`random_state_`. `n_iter_` starts at one for the initial objective evaluation,
then increments after each accepted optimizer step, as in the reference.
Feature names are `neighborhoodcomponentsanalysis0`, etc.

Normal refits restart initialization; warm starts retain the previous embedding,
including its output dimension if `n_components` subsequently changes. Feature
count changes in a warm fit raise. Invalid input does not replace an existing
fit. Configuration changes do not erase the fit; cloning produces an unfitted
estimator with independently owned configuration.

## Native adaptations and limits

- Input is `NDArray[float32]` or `[float64]`, selected by the estimator's first
  compile-time parameter. The third parameter Y (after Callback) selects target
  storage: Int64/Float32/Float64 NDArray or StringTarget, independent of feature
  dtype. Rank-one and single-column classification targets use shared label
  encoding. Arbitrary Python objects and sparse inputs are not supported.
- Computation, learned components and transformed output use Float64, including
  Float32 inputs. This avoids dependence on a Float32 SVD/optimizer trajectory.
  PCA uses economy full SVD for every shape; LDA uses the reference's two SVD
  stages and rank cutoffs. Singular-vector signs and repeated-subspace bases can
  differ from LAPACK. Zero-rank LDA initialization raises.
- `fit` and `set_params` mutate in place. Parameter access uses `NCAParams` from
  `get_params`; explicit initial matrices occupy `init_matrix` with `init="array"`.
  Named updates cover all parameters, with a separate `set_params(init=matrix)`.
  Omitted or None named updates leave configuration unchanged; to reset
  `n_components` or `random_state` to None, edit the typed Params record.
- `callback=None` is represented by the default `NoNCACallback` type. For a
  callback, supply a default-constructible native struct implementing NCACallback
  as the second compile-time parameter and pass it with `callback=`. Its mutable
  `__call__(transformation, n_iter)` receives an immutable Float64 flat array and
  the reference's one-based iteration index. Callback state is owned by the
  estimator and available through `get_params().callback`. Exceptions propagate.
- Integer seeds reproduce the adopted shared legacy RandomState normal stream.
  `random_state=None` creates an independent native entropy-seeded stream; it
  does not borrow NumPy's process-global RNG. To advance an application-owned
  stream, call `fit(X,y,random_state=rng)` with a mutable shared RandomState.
  `random_state_` is an independent post-initialization snapshot.
- The local unbounded L-BFGS optimizer stores ten correction pairs and uses
  SciPy 1.18.1 L-BFGS-B's More–Thuente line search (`ftol=1e-3`, `gtol=0.9`,
  `xtol=0.1`, at most 20 trials). The first step is `min(1/||d||, 1e10)`;
  later searches start at one. Search warnings accept the current trial.
  Failed searches restore the accepted point and retry after clearing history;
  a failure with empty history stops without calling the callback.
- NCA's `tol` sets both the gradient infinity-norm threshold and relative
  objective-decrease threshold. After an accepted step, callbacks run first,
  then the `max_iter` and 15,000-evaluation budgets are checked, then gradient
  and objective convergence. The evaluation budget uses strict `>` and can be
  exceeded during a search, matching SciPy. Curvature updates use the reference
  machine-epsilon cutoff. Repeated evaluations at the same point are cached.
- This is NCA's unbounded path, not a public box-constrained solver. Two-loop
  inverse-Hessian products and native objective reductions differ in rounding
  from SciPy's compact factorization/BLAS. Bitwise components, iteration counts
  on arbitrary ill-conditioned problems, and identical local optima are not
  guaranteed, even with matching line-search and stopping rules.
- `verbose > 0` prints iteration/objective diagnostics and a convergence warning
  on an iteration/search limit. Python warning objects and exact log formatting
  are not exposed. Errors use the family's prefixed Mojo Error convention.
- Named input columns, pandas output, metadata routing, Python reflection and
  persistence follow the exclusions in [interfaces.md](interfaces.md).

The objective uses off-diagonal stabilized softmax and analytic gradients.
Training has quadratic sample storage plus ten embedding-sized correction pairs;
no performance claim is made. Extreme inputs that overflow squared distances
raise instead of publishing nonfinite results.

## Validation

Run `pixi run build` and `pixi run test`. Both independent downstream consumers
compile and fit NCA. Contracts exercise both dtypes, invalid inputs, initializers,
ownership, cloning, refitting, warm starts, callback indices, stream advancement,
feature names, and strided inputs.

`tests/parity_nca.py` requires a compiled copy of the pinned local sklearn
revision, just like the tree parity suites. It verifies source bytes against
`4358542a1bd39af2e7e1cbace8b9fed067473c5b` before comparing. For example:

```sh
PYTHONPATH=/path/to/oracle-site-packages \
SKLEARN_ORACLE_PYTHON=/path/to/oracle-python pixi run parity
```

Objective and gradient comparisons use absolute/relative tolerances of 3e-12
and 3e-11; independent finite-difference error must be below 3e-5. PCA components
use 2e-10, LDA metric matrices use 2e-9 to allow singular-vector sign differences.
Twelve mixed-label trajectory fixtures from identical explicit initial matrices
require exact callback indices and iteration counts, and compare all accepted
components at `atol=3e-6, rtol=3e-7`. These tolerances allow accumulated reduction
and Hessian rounding on nonconvex paths. Separated-data fixtures including
spectral initialization additionally require expected success above 17.99/18
and agreement with sklearn within 0.01. Toy collapse, one-class, all-singleton,
large-distance and nearest-neighbor geometry checks complement those fixtures.
This does not establish full sklearn estimator-check parity.

`tests/parity_nca_solver.py` compares the private search directly with SciPy
1.18.1: all four interpolation branches in both bracket states, 24 complete
search traces, the modified-objective stage, eight invalid search settings,
and unbounded quadratic, quartic, Rosenbrock, linear and inconsistent-gradient
problems. Tests compare evaluation/callback traces,
iteration and evaluation counts, termination reasons, warning acceptance and
rollback/retry. Scalar interpolation uses `3e-14` relative/absolute tolerance;
small solver trajectories use `2e-7` for compact-versus-two-loop rounding.
The native `test_nca_solver` contracts also cover budget/convergence precedence,
the strict evaluation-budget boundary, initial convergence and failed searches.
SciPy is validation-only; no additional numerical runtime dependency is used.
