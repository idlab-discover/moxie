# Ridge, RidgeClassifier, ridge_regression (dense)

Ports of `sklearn.linear_model.Ridge`, `RidgeClassifier` and `ridge_regression`
(reference 4358542a1b) for dense Float32/Float64 NuMojo arrays.

```mojo
from moxie.linear_model import Ridge, RidgeClassifier, ridge_regression, ClassWeight
var model = Ridge(alpha=1.0, fit_intercept=True, solver="auto", tol=1e-4)
model.fit(X, y)                      # y (n,) or (n, k); fit(X, y, sample_weight=w)
var clf = RidgeClassifier[DType.float64, DType.int64](class_weight=ClassWeight.balanced_weights())
clf.fit(X, labels)                   # integer labels (n,) or 0/1 indicators (n, L)
var result = ridge_regression(X, y, 1.0, solver="lsqr", return_n_iter=True)
```

Constructor defaults follow sklearn: `alpha=1.0` (`ScalarOrVector`: a Float64
or one value per target), `fit_intercept=True`, `copy_X=True`,
`max_iter=None`, `tol=1e-4`, `solver="auto"`, `positive=False`,
`random_state=None`; RidgeClassifier adds `class_weight=None`.

Learned: `coef_` ((p,) for a single target/binary problem, else (k, p)),
`intercept_` ((k,); length one for a single target), `n_iter_` (per-target LSQR
iterations, otherwise None), `solver_` (resolved solver, `"svd"` after a
singular Cholesky fallback), `n_features_in_`; RidgeClassifier also `classes_`
and `decision_function`. `ridge_regression` returns `RidgeRegressionResult`
(`coef`, `n_iter` when `return_n_iter`, `intercept` unused).

## Solvers

| solver | Implementation |
| --- | --- |
| auto | cholesky (positive=True -> lbfgs), as sklearn for dense input |
| cholesky | Float64-accumulated Gram + mojo-linalg Cholesky; dual (kernel) form when n_features > n_samples; singular -> svd (lstsq for a shared alpha in dual form) |
| svd | mojo-linalg SVD, s > 1e-15, d = s/(s^2+alpha) |
| lsqr | mojo-linalg damped LSQR (damp = sqrt(alpha), atol=btol=tol, iter_lim=max_iter) |
| sparse_cg | mojo-linalg CG on X^T X + alpha I, or X X^T + alpha I when n_features > n_samples (max_iter ignored there, as sklearn) |
| lbfgs | mojo-optimize L-BFGS-B (ftol=gtol=tol, bounds [0, inf)) for positive=True |
| sag, saga | **Not implemented** (NotImplementedError): need the SAG engine |

## Mojo adaptations

- Labels are integers (`label_dtype`, default Int64); string labels are not
  supported. Explicit `class_weight` dictionaries map `Int` labels; for
  multilabel targets only `balanced` is supported. Rank-2 y must be 0/1.
- Negative sample weights are rejected (sklearn silently produces NaN through
  sqrt). Scalar fit weights are supported by the estimators; the standalone
  ridge_regression helper still takes an optional weight vector.
- `intercept_` is always an array; `copy_X` never mutates inputs.
- Iterative solvers run in Float64 for Float32 data (SciPy's CG runs float32).

## Performance baseline

`pixi run python tools/benchmark.py` -> `benchmarks/ridge_fit.json`. Single
thread, Ryzen 9 5950X: 200x10 fits take ~0.1 ms for cholesky/lsqr/sparse_cg
(sklearn 0.4-0.8 ms, dominated by Python overhead); 2000x50 cholesky 7 ms vs
0.8 ms, lsqr 5 ms vs 1.3 ms; the scalar Jacobi `svd` solver is the slowest.

Scalar nonnegative `sample_weight` is accepted by `Ridge.fit` and
`RidgeClassifier.fit` and expanded to the input dtype. Column-vector class
labels have the same classes and vector predictions as rank-one labels.
`score` validates weight shapes/finiteness before accessing their entries.
