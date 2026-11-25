# ElasticNet and Lasso

Use `from moxie.linear_model import ElasticNet, Lasso` with the family or child
assembled `builds/deps` namespace. See `examples/elastic_net.mojo`.

Both support Float32/Float64 (compile-time dtype), cyclic and random
selection, alpha, fit_intercept, max_iter, tol, positive, warm_start, independent multi-output
regression, nonnegative array weights and scalar weights. ElasticNet also takes
l1_ratio in [0,1]. Scalar weights cancel from the average-loss objective, as in
the pinned reference. Array weights are normalized to sum to n_samples before
shared weighted centering and square-root rescaling. Lasso uses l1_ratio=1.

Constructors keep sklearn defaults: alpha=1, fit_intercept=True,
precompute=False, max_iter=1000, copy_X=True, tol=1e-4, warm_start=False,
positive=False, random_state=None, selection="cyclic"; ElasticNet l1_ratio=0.5.
Inputs are always preserved, including when copy_X=False. Fit mutates in place.
`get_params` returns a typed `ElasticNetParams` or `LassoParams` record.
`set_params` accepts that record or keyword updates for every constructor
parameter. Omitted keywords preserve their values. To reset a seed, set
`random_state = None` on the record and pass it back to `set_params`.
The method returns a mutable borrow for chaining with `fit`.
Fit validates the configuration. Clone copies configuration only.
Failed fits retain the previous learned state.

```mojo
var model = Lasso()
_ = model.set_params(selection=String("random"), random_state=0).fit(X, y)
```

Learned `coef_` is (features,) for a vector or one-column target, otherwise
(outputs, features). `intercept_` is always (outputs,), including zeros without
an intercept. `predict` returns a vector for a single output and a matrix for
multiple outputs. `score` is weighted, uniformly averaged R2.

Mojo adaptations: `dual_gap_` is always an array (one entry per output),
`n_iter_` is always a List[Int]. `converged_` is a List[Bool] reporting whether
the dual-gap criterion was met. Iteration-limited fits retain their computed
solution; consult `converged_` in place of Python's ConvergenceWarning. Warm
starts reuse coefficients and reconstruct the intercept with the new means;
incompatible feature/output counts raise. A converged warm start may take
zero iterations, as in the pinned local reference.

Open dense work includes Gram precomputation and supplied Gram arrays,
multi-task group penalties, and CV variants. `selection="random"` draws
coordinates with replacement from the active set. `random_state` accepts
a 32-bit unsigned seed as `Int`, or `None` for Moxie's shared default
stream. An integer resets the stream at each fit and for each independently
fitted target, including warm refits. The cyclic solver skips RNG draws.
Python `check_input=False`, mutable RandomState objects, metadata routing
and feature names have no native interface.

## Single-target regularization paths

Import `enet_path` and `lasso_path` from `moxie.linear_model`, or call
`ElasticNet[dtype].path` and `Lasso[dtype].path`. The path takes a dense
Float32 or Float64 feature matrix and a same-dtype target vector. It uses the
data as supplied. Center the inputs first when a fit with an intercept is
required. The residual solver starts each alpha from the preceding solution.
The default selection is cyclic. For random selection, the path uses one
MT19937 stream and takes a fresh solver seed at each alpha, including points
that converge without a sweep. Optional `coef_init` is a same-dtype vector
with one entry per feature. Pass a view (for example, `coef_init=initial.view()`)
to supply an optional array while retaining the original handle. The path
preserves caller inputs. It packs the feature
matrix by column once and reuses that buffer for the full path.

Use `n_alphas` for the automatic grid count (default 100). Its first point
is the maximum absolute feature–target correlation divided by
`n_samples * l1_ratio`, with a one-sided maximum when `positive=True`.
`eps` (default `1e-3`) scales the last point. Correlations accumulate in
Float64, and grids below the Float64 resolution floor use the constant value
`1e-15`. `alphas` accepts an explicit Float64 vector and takes precedence over
the generated grid. For a reusable grid, pass `alphas=grid.view()`. Explicit alphas are sorted descending and duplicates are
retained. Set `l1_ratio=0` with an explicit grid to compute a pure L2 path. The
default ratio is 0.5 for `enet_path` and 1 for `lasso_path`.

The native `CoordinatePathResult[dtype]` always contains `alphas` (Float64,
`(n_alphas,)`), `coefs` (input dtype, `(n_features, n_alphas)`) and
`dual_gaps` (Float64, `(n_alphas,)`). `n_iters` is an
`Optional[List[Int]]` populated by `return_n_iter=True`. Set
`early_stopping=False` to perform a sweep even when the initial coefficients
meet the gap criterion. Each gap uses the average-loss scale of the public
objective.

The native path uses the residual engine and always validates inputs.
The dense multi-task and Gram path engines remain open.
