# GaussianMixture (dense, Float32/Float64)

```mojo
from moxie.mixture import GaussianMixture

var model = GaussianMixture[DType.float64](3, covariance_type="full", random_state=0)
model.fit(X)                        # X: NDArray[DType.float64] (n_samples, n_features)
var labels = model.predict(X)       # NDArray[DType.int64] (n_samples,)
var proba = model.predict_proba(X)  # (n_samples, n_components), rows sum to 1
var ll = model.score(X)             # mean log-likelihood (not a sum)
var drawn = model.sample(100)       # (points Float64 (100, d), labels Int64 (100,))
print(model.bic(X), model.aic(X))
```

Constructor parameters and defaults follow `sklearn.mixture.GaussianMixture`:
`n_components=1, covariance_type="full", tol=1e-3, reg_covar=1e-6,
max_iter=100, n_init=1, init_params="kmeans", weights_init=None,
means_init=None, precisions_init=None, random_state=None, warm_start=False,
verbose=0, verbose_interval=10`. Every covariance type (`full`, `tied`,
`diag`, `spherical`) and every `init_params` (`kmeans`, `k-means++`, `random`,
`random_from_data`) is implemented.

Methods: `fit`, `fit_predict`, `predict`, `predict_proba`, `score_samples`,
`score`, `sample`, `bic`, `aic`, `get_params`, `set_params`,
`__sklearn_clone__` (via `moxie.base.clone`), `__sklearn_is_fitted__`.
Learned attributes: `weights_`, `means_`, `covariances_`, `precisions_`,
`precisions_cholesky_`, `converged_`, `n_iter_`, `lower_bound_`,
`lower_bounds_`, `n_features_in_`. Layouts: see the family plumbing table
(`full (k,d,d)`, `tied (d,d)`, `diag (k,d)`, `spherical (k,)`).

## Mojo adaptations (contract-tested in `tests/mixture/gaussian/test_gaussian.mojo`)

- The dtype is a compile-time parameter. X, `weights_init`, `means_init`,
  `precisions_init` and learned arrays share it; `sample` returns Float64 points.
- `random_state: Optional[Int]`. An integer reseeds a fresh `RandomState` at
  each `fit`/`sample` (sklearn `check_random_state(int)`). `None` uses
  `entropy_state()` per call (not reproducible). The sklearn "RandomState
  instance" mode is `fit_with_rng(X, rng)`, `fit_predict_with_rng(X, rng)`,
  `sample_with_rng(n, rng)`, which borrow and advance a caller-owned stream.
- `fit`, `fit_with_rng`, and `set_params` mutate in place and return a mutable borrow of the estimator; learned state is committed only
  after a successful fit (sklearn may leave partial state on an exception).
- Errors are Mojo `Error`s whose text starts with the Python exception class
  (`ValueError: ...`, `NotFittedError: ...`, `AttributeError: ...`).
- ConvergenceWarning: printed to stderr and stored in `convergence_warning_`
  (empty when converged or `max_iter=0`).
- Before fit, learned arrays are empty (sklearn raises AttributeError).
- `get_params` returns the typed `GaussianMixtureParams` record; `set_params`
  takes it. A keyword overload, `set_params(n_components=2, covariance_type="diag")`,
  updates only the named parameters (sklearn `set_params(**params)`); a full
  record is needed to reset an optional parameter to None. Metadata routing, `feature_names_in_`, pickling and array API
  dispatch are Python binding work and not provided.
- Parameter validation uses the family `EMControl`, which additionally rejects
  non-finite `tol`/`reg_covar` (sklearn accepts `inf`).
- Inputs may be strided NuMojo views; they are read in place (no copy).

## Numerical notes

Arithmetic accumulates in Float64 and rounds to the estimator dtype wherever
NumPy materializes an array, keeping sklearn's expanded formulas. Cholesky and
triangular solves use mojo-linalg on Float64 work matrices. Logs/exps use the
LLVM intrinsics (libm) because stdlib `log`/`exp` are ~1e-10 accurate.
KMeans initialization is a local translation of sklearn's k-means++ + Lloyd
(`_kmeans_init.mojo`); with several simultaneously empty clusters the
relocation order may differ from NumPy's `argpartition`. Full/tied `sample`
draws from the symmetric part of the covariance through
`RandomState.multivariate_normal` (Jacobi SVD): draw consumption, labels and
Mahalanobis norms match NumPy, exact coordinates may differ by factor orientation.
