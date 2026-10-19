# PLSRegression, PLSCanonical, CCA (dense, pure Mojo)

```mojo
from moxie.cross_decomposition import PLSRegression, PLSCanonical, CCA
var pls = PLSRegression(n_components=2)          # scale=True, max_iter=500, tol=1e-6, copy=True
pls.fit(X, y)                                    # X (n,p), y (n,) or (n,q), NuMojo NDArray
var y_hat = pls.predict(X)                       # (n,) after a 1-D-y fit, else (n,q)
var scores = pls.transform(X, y)                 # PairedScores(x_scores, y_scores)
var r2 = pls.score(X, y)                         # uniform-average R^2 (optional weights)
var can = PLSCanonical(2, algorithm="svd")
var cca = CCA(1)
```

Behaviour follows `sklearn/cross_decomposition/_pls.py` at 4358542a: NIPALS
power method (mode A for PLSRegression/PLSCanonical, mode B with the legacy
`pinv2` cutoff `max(s)*1e6*eps` for CCA), PLSCanonical `algorithm='svd'`,
regression/canonical deflation, `_svd_flip_1d` signs, rotations
`W pinv(P'W)` with SciPy's cutoff `max(shape)*eps*max(s)`, `coef_`,
`intercept_`, `n_iter_`, constant-residual early stop and the same
ValueError/InvalidParameterError messages (first line) as scikit-learn.

Learned attributes: `x_weights_, y_weights_, x_loadings_, y_loadings_,
x_rotations_, y_rotations_, coef_ (q,p), intercept_ (q,), n_iter_,
n_features_in_`, training scores `x_scores_, y_scores_`.

## Mojo adaptations (each covered by tests/contracts.mojo)

| scikit-learn | Moxie |
| --- | --- |
| three classes over private `_PLS` | `_PLS[flavor, dtype]` with aliases `PLSRegression[dtype=float64]` etc.; only PLSCanonical's constructor takes `algorithm` |
| `copy` attribute | constructor/`set_params` keyword `copy`; stored as `copy_input` (`copy` clashes with `Copyable.copy()`) |
| `fit(X, y)` with copy=False mutates X, y | `fit` never mutates; `fit_inplace(mut X, mut y)` honours `copy` (Float64 inputs end as the deflated residuals, Float32 inputs are converted and untouched) |
| `transform(X, y, copy=False)`, `predict(X, copy=False)` | `transform_inplace`, `predict_inplace` |
| `transform(X, y)` tuple / `inverse_transform(X, y)` tuple | `PairedScores` record (`x_scores`, `y_scores` fields hold X and y blocks) |
| `get_params()` dict | `PLSParams` record; keyword `set_params(...)` overload; `algorithm` other than 'nipals' rejected for PLSRegression/CCA |
| Python warnings | `diagnostics_: List[String]` ("ConvergenceWarning: Maximum number of iterations reached", "UserWarning: y residual is constant at iteration k") |
| `_x_scores` private on PLSCanonical/CCA | exposed as `x_scores_`/`y_scores_` on all three |
| float64 outputs for float32 input | same: inputs `NDArray[dtype]`, outputs Float64; inference centers/scales in input dtype like NumPy's in-place ops |
| `get_feature_names_out()` | returns `List[String]` (`plsregression0`, ...) |

Numerical failures that make SciPy raise (NaN reaching an SVD after a zero
deflation denominator) raise `LinAlgError: ...` here; NaN in inputs is
rejected with scikit-learn's first-line message (its imputation hint is
omitted). Pandas output, feature_names_in_, metadata routing and pickling are
binding work, not implemented.

`PLSEngine` also exposes `projection_weights` (PLSSVD) and `rotations`, and
`moxie.cross_decomposition._pls.pinv(a, rtol, context)`.

`fit`, both `set_params` forms, and `fit_inplace` return a mutable borrow of
the estimator for fluent chaining. Shared-trait calls use current Moxie
contracts. See the [family guide](../README.md) for validation commands.
