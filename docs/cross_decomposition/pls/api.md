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
| Python warnings | `fit_warnings: List[FitWarning]` records the category and message in fit order. The legacy `diagnostics_` list formats each record as `category: message`. |
| `_x_scores` private on PLSCanonical/CCA | exposed as `x_scores_`/`y_scores_` on all three |
| float64 outputs for float32 input | same: inputs `NDArray[dtype]`, outputs Float64; inference centers/scales in input dtype like NumPy's in-place ops |
| `get_feature_names_out(input_features=None)` | returns `List[String]` (`plsregression0`, ...); supplied names are length-checked only (no `feature_names_in_`). `FeatureNamesOut` and `InputFeatureCount` (`get_n_features_in()`) are declared, so pipelines report names and input width |

Read `model.fit_warnings` after fitting, including on a fitted Pipeline step.
NIPALS records `ConvergenceWarning` with message
`Maximum number of iterations reached` for a component that reaches the cap,
and `UserWarning` with message `y residual is constant at iteration k` when
extraction stops on a constant target residual. Each successful fit replaces
the list. Failed fits preserve it, and unfitted clones start with an empty list.
See the shared [fit warning contract](../../interfaces.md#fit-warnings).

`set_score_request(sample_weight=...)` configures the request for a parent
that forwards weights to `score`. It takes `True`, `False`, `None`, or a
`String` alias. Omitting the keyword retains the request. The setter returns a
mutable borrow, and `get_metadata_routing().score` returns an independent
`SampleWeightRequest`. The typed API is always available in Mojo. Cloning
preserves the request, as do fitting and native archive round trips. Direct
calls accept weights through the `score` argument regardless of the request.
See the [shared sample-weight contract](../../interfaces.md#sample-weight-requests).
Parent forwarding is owned by each composition. Mojo exposes the mutating
prediction and transform operations as separate methods, so they take no
`copy` metadata.

Numerical failures that make SciPy raise (NaN reaching an SVD after a zero
deflation denominator) raise `LinAlgError: ...` here; NaN in inputs is
rejected with scikit-learn's first-line message (its imputation hint is
omitted). Pandas output, fitted input feature names, and pickling still require
shared support.

`PLSEngine` also exposes `projection_weights` (PLSSVD) and `rotations`, and
`moxie.cross_decomposition._pls.pinv(a, rtol, context)`.

`fit`, both `set_params` forms, and `fit_inplace` return a mutable borrow of
the estimator for fluent chaining. Shared-trait calls use current Moxie
contracts. See the [family guide](../README.md) for validation commands.
