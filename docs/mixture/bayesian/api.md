# BayesianGaussianMixture (dense, Float32/Float64)

```mojo
from moxie.mixture import BayesianGaussianMixture

var model = BayesianGaussianMixture[DType.float64](
    n_components=5, weight_concentration_prior_type="dirichlet_process",
    weight_concentration_prior=0.01, random_state=0)
model.fit(X)
var labels = model.predict(X)
print(model.weights_.unsafe_get(0), model.lower_bound_)
```

Keyword-only constructor with sklearn names and defaults: `n_components=1,
covariance_type="full", tol=1e-3, reg_covar=1e-6, max_iter=100, n_init=1,
init_params="kmeans", weight_concentration_prior_type="dirichlet_process",
weight_concentration_prior=None, mean_precision_prior=None, mean_prior=None,
degrees_of_freedom_prior=None, covariance_prior=None, random_state=None,
warm_start=False, verbose=0, verbose_interval=10`. Both weight prior types and
all four covariance types are implemented. Methods: `fit`, `fit_predict`,
`predict`, `predict_proba`, `score_samples`, `score`, `sample`, typed
`get_params`/`set_params`, clone, plus `fit_with_rng`, `fit_predict_with_rng`,
`sample_with_rng`. No `bic`/`aic` (Gaussian-only, as in sklearn).

Learned attributes: `weights_`, `means_`, `covariances_`, `precisions_`,
`precisions_cholesky_`, `converged_`, `n_iter_`, `lower_bound_`,
`lower_bounds_`, `weight_concentration_prior_`, `weight_concentration_`,
`mean_precision_prior_`, `mean_precision_`, `mean_prior_`,
`degrees_of_freedom_prior_`, `degrees_of_freedom_`, `covariance_prior_`,
`n_features_in_`, `convergence_warning_`.

Mojo adaptations (contract-tested in `tests/mixture/bayesian/test_bayesian.mojo`), in addition
to those of GaussianMixture (compile-time dtype, `Optional[Int]` seed,
in-place fit, `Error` prefixes, stderr ConvergenceWarning):
- Scalar priors are `Optional[Float64]` and must be finite and positive
  (sklearn also accepts `inf`); resolved `*_prior_` scalars are Float64.
- `mean_prior`/`covariance_prior` are `Optional[NDArray[dtype]]`; a spherical
  `covariance_prior` is a positive shape-(1,) array (sklearn: a float).
  `covariance_prior_` is (d,d), (d,) or (1,).
- `weight_concentration_` is a `List` of two arrays (Dirichlet process:
  stick-breaking a, b) or one array (Dirichlet distribution).
- Tied `degrees_of_freedom_` has shape (1,) (sklearn: a scalar).
- Float32 storage throughout; the reference's Float32 path is mixed precision
  (np.cov priors and full/tied covariances in Float64).
digamma is `moxie._mixture_linalg.special_gamma.digamma`; gammaln wraps stdlib `lgamma`;
betaln is gammaln-based (SciPy's extreme-ratio asymptotic branch is absent).
