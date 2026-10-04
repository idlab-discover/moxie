# Graphical lasso API

```mojo
from moxie.covariance import GraphicalLasso, graphical_lasso

var model = GraphicalLasso(alpha=0.05)          # [dtype=DType.float64]
model.fit(X)                                    # or covariance="precomputed" with X = emp_cov
var P = model.precision_().value()              # sparse (dense storage) precision
var costs = model.costs_()                      # List[(objective, dual gap)]
var result = graphical_lasso(emp_cov, 0.05, mode="lars")     # GraphicalLassoResult
```

| sklearn | Mojo |
| --- | --- |
| `GraphicalLasso(alpha=0.01, *, mode='cd', covariance=None, tol=1e-4, enet_tol=1e-4, max_iter=100, verbose=False, eps=finfo(float64).eps, assume_centered=False)` | same keywords; `covariance: Optional[String]` |
| `covariance_`, `precision_`, `location_`, `costs_`, `n_iter_`, `n_features_in_` | accessor methods |
| `graphical_lasso(emp_cov, alpha, *, ..., return_costs=False, return_n_iter=False)` | returns `GraphicalLassoResult` with covariance, precision, costs and n_iter always present (Float64) |

Both `mode='cd'` (default) and `mode='lars'` are supported, including CV.
LARS follows sklearn's least-angle Gram path with interpolation to the requested
alpha and `eps`-floored Cholesky pivots. `enet_tol` affects only CD; the inner
LARS path has sklearn's fixed 500-step limit, independent of outer `max_iter`.

Diagnostics use native stderr warnings and stdout iteration output (see below).
Ill-conditioning raises
`FloatingPointError:`-prefixed Errors like sklearn. Computation is Float64,
also for Float32 input (sklearn's Float32 precomputed/assume_centered path
can fail in CD's Cython kernel with a buffer dtype mismatch). alpha=0 inverts
emp_cov with Cholesky when positive definite, falling back to symmetric LDLᵀ
when Cholesky reports a non-positive pivot (singular raises LinAlgError).
This follows SciPy's positive-definite structure selection without jitter or
a pseudoinverse fallback. Near-singular inverse entries and derived costs can
still differ because the backend arithmetic and inverse construction differ.

For positive alpha, `max_iter=0` returns the initialized matrices, empty costs
and `n_iter_=1`, matching the reference counter convention. This is not
convergence or one completed update. With alpha zero, `n_iter_=0`. Exhausting
the iteration budget returns the estimate with a convergence warning; inspect
the final dual gap when costs are present.

## GraphicalLassoCV

```mojo
from moxie.covariance import GraphicalLassoCV

var model = GraphicalLassoCV(alphas=4, n_refinements=4, cv=5)
model.fit(X)
var alpha = model.alpha_()
var results = model.cv_results_()
# results.alphas, mean_test_score, std_test_score: Float64 vectors
# results.split_test_score: Float64 matrix (n_candidates, n_folds)
```

Constructor defaults match sklearn: `alphas=4`, `n_refinements=4`, `cv=None`
(unshuffled five-fold KFold), `tol=1e-4`, `enet_tol=1e-4`, `max_iter=100`,
`mode="cd"`, `n_jobs=None`, `verbose=False`, `eps=Float64 epsilon`, and
`assume_centered=False`. Integer `alphas` and `List[Float64]` convert to
`AlphaGrid`; explicit lists preserve evaluation order and skip refinement.
Results sort alphas descending, preserve duplicates, and append alpha zero.
Alpha zero is a reporting baseline, never a selection candidate. Automatic
Float32 alpha grids can differ in the last few digits because native grid
construction uses Float64; parity checks also verify the selected grid index. Tied mean
scores select the smallest alpha. Fold paths always center training and test
sets separately; `assume_centered` controls the full-data covariance/refit.

Native adaptations:

- Learned attributes are accessor methods, including `alpha_()`,
  `cv_results_()`, `costs_()` and `n_iter_()`. CV results use the typed record
  above instead of dynamically named dictionary keys. Arrays are owned copies.
- `get_params`/`set_params` use `GraphicalLassoCVParams`; cloning copies
  configuration and discards fitted state. `fit` mutates in place.
- `cv` accepts `Optional[Int]`. For custom splitters/iterables, materialize
  `List[IndexSplit]` from native splitter cursors and call
  `model.fit(X, splits=splits)`. These indices are validated before use.
- Execution is serial: `n_jobs=None` and `1` are supported; other values
  raise `NotImplementedError`.
- `verbose=True` prints refinement progress. Python warning objects and
  joblib timing/progress output are not reproduced.
- Empty alpha lists and automatic grids with zero off-diagonal covariance
  raise clear errors. Explicit positive alphas remain available. A one-point
  grid is handled without sklearn's refinement indexing failure.
- Singular empirical alpha-zero baselines inherit determinant-rounding
  differences: finite versus `-inf` scores can differ from sklearn. This
  reporting-only row does not participate in alpha selection.
- Float64 computation and static output dtype follow GraphicalLasso above.
  Metadata routing and pandas feature names remain outside the native API.

The implementation reuses the graphical solver and model-selection's
`KFold`, row selection and `cross_val_score` (for the empirical baseline).
See [family parity](../../../docs/parity.md) for remaining family gaps.

## Parameter updates

`set_params` accepts a typed record from `get_params()` or keyword partial
updates. Omitted keywords preserve constructor settings; supplied values do not
clear fitted state. Clones contain the updated configuration and no learned
state. Validation remains deferred until fit, and successful refits replace
learned state. Both methods return a mutable borrow of the original estimator, supporting
chained updates and fitting.

Numerical failures can leave a new location with the old covariance/precision.
CV also publishes alpha and CV results before its final solve, following sklearn.
Invalid-input/constructor-parameter rejection preserves previous estimates. See the [failure-state audit](../../../docs/parity.md).

`GraphicalLasso` accepts `covariance=None`; `GraphicalLassoCV` accepts `cv=None`
and `n_jobs=None`. Typed `Optional` values work too. Alpha updates accept integer grid sizes, `List[Float64]` values, or the same
`AlphaGrid` representation as construction:

```mojo
from moxie.covariance._graph_lasso import AlphaGrid
model.set_params(alphas=AlphaGrid([Float64(0.3), Float64(0.1)]), cv=None)
```

Alpha lists are copied into configuration records and clones.

Graphical nonconvergence emits `ConvergenceWarning: ...` on stderr. Grid search
suppresses convergence warnings; the final CV fit reports them. `verbose=True`
prints solver iteration/cost/dual gap on stdout (CV refinement progress uses
stderr). Formatting differs from Python. Float64 solver precision is retained
for inference even when public array snapshots are Float32.
