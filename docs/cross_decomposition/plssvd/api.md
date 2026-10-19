# PLSSVD (dense, pure Mojo)

```mojo
from moxie.cross_decomposition import PLSSVD
var pls = PLSSVD(n_components=2)            # scale=True, copy=True
pls.fit(X, y)                               # y (n,) or (n,q)
var scores = pls.transform(X, y)            # PairedScores(x_scores, y_scores), Float64
var x_only = pls.transform(X)
```

Follows `PLSSVD` in sklearn `_pls.py` at 4358542a: centering/scaling (ddof=1,
zero std -> 1), one economy SVD of `X.T @ y`, first `n_components` vectors,
u-based `svd_flip`. Learned: `x_weights_ (p,k)`, `y_weights_ (q,k)`,
`n_features_in_`. Bound `n_components <= min(n, p, q)` with scikit-learn's
message. Transform converts inputs to Float64 before centering, as upstream.

Mojo adaptations: `PLSSVD[dtype=float64]` over NuMojo arrays; `copy` stored as
`copy_input`; `fit` never mutates inputs and `fit_inplace(mut X, mut y)`
reproduces copy=False (Float64 inputs end centered/scaled); `PLSSVDParams`
record plus keyword `set_params`; paired results are a `PairedScores` record;
`get_feature_names_out()` returns `List[String]` (`plssvd0`, ...). No
predict/score/inverse_transform, matching scikit-learn. Singular vectors for
exactly zero singular values (k above the rank of X.T @ y) are an arbitrary
orthonormal completion, as with LAPACK, but may differ from it.

`fit`, both `set_params` forms, and `fit_inplace` return a mutable borrow of
the estimator for fluent chaining. Shared-trait calls use current Moxie
contracts. See the [family guide](../README.md) for validation commands.
