# ElasticNet and Lasso

Use `from moxie.linear_model import ElasticNet, Lasso` with the family or child
assembled `builds/deps` namespace. See `examples/elastic_net.mojo`.

Both support Float32/Float64 (compile-time dtype), cyclic selection, alpha,
fit_intercept, max_iter, tol, positive, warm_start, independent multi-output
regression, nonnegative array weights and scalar weights. ElasticNet also takes
l1_ratio in [0,1]. Scalar weights cancel from the average-loss objective, as in
the pinned reference. Array weights are normalized to sum to n_samples before
shared weighted centering and square-root rescaling. Lasso uses l1_ratio=1.

Constructors keep sklearn defaults: alpha=1, fit_intercept=True,
precompute=False, max_iter=1000, copy_X=True, tol=1e-4, warm_start=False,
positive=False, random_state=None, selection="cyclic"; ElasticNet l1_ratio=0.5.
Inputs are always preserved, including when copy_X=False. Fit mutates in place.
`get_params` and `set_params` use typed `ElasticNetParams`/`LassoParams` records;
keyword updates are provided for alpha, tol, max_iter and warm_start. Clone
copies configuration only. Failed fits retain the previous learned state.

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

Unsupported: selection="random" (shared our_rand_r missing), precompute=True
(Gram engine unfinished), supplied Gram arrays, multi-task group penalties,
path functions and CV variants. Random_state accepts None or a 32-bit unsigned
seed represented as Int; mutable RandomState objects are not exposed. These
are unfinished dense branches, not sparse deferrals. Python check_input=False,
metadata routing and feature names have no native interface.
