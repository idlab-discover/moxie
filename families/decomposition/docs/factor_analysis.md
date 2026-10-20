# Dense FactorAnalysis

`from moxie.decomposition import FactorAnalysis` fits the linear Gaussian
latent-variable model `x = W^T z + mu + e`, `e ~ N(0, diag(psi))`, by the
SVD-based EM of Barber (2012, Alg. 21.1), following the local scikit-learn
`sklearn/decomposition/_factor_analysis.py` at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

```mojo
var fa = FactorAnalysis(n_components=2, rotation=Optional[String]("varimax"))
var latent = fa.fit_transform(X)     # NDArray[float64], (n_samples, 2)
var loglik = fa.score(X)
```

See [the example](../examples/factor_analysis.mojo).

## API

Defaults follow sklearn: `n_components=None` (all features), `tol=1e-2`,
`copy=True`, `max_iter=1000`, `noise_variance_init=None` (`List[Float64]`),
`svd_method="randomized"`, `iterated_power=3`, `rotation=None`
(`"varimax"`/`"quartimax"`), `random_state=0`.
Learned: `components_ (k, p)`, `noise_variance_ (p)`, `mean_ (p)` as
`Optional[NDArray[float64]]`, `loglike_: List[Float64]`, `n_iter_`, `n_features_in_`.
Methods: `fit`, `fit_transform`, `transform` (posterior latent mean),
`get_covariance`, `get_precision` (matrix inversion lemma, like sklearn),
`score_samples`, `score` (mean log-likelihood), `get_feature_names_out`
(`factoranalysis0`, ...), typed `get_params`/`set_params`
(`FactorAnalysisParams`; `copy_input` is sklearn's `copy`), `__sklearn_clone__`.

Language adaptations, each covered by contract tests:

- As in the reference (`validate_data(dtype=np.float64)`), fitting and all
  learned/returned arrays are Float64 for Float32 or Float64 input.
- `copy=False` needs `fit_inplace(mut X)`; for Float64 input it leaves X
  centered like sklearn. `fit` with `copy=False` raises instead of mutating an
  immutable borrow.
- The reference's `ConvergenceWarning` has no native warning channel: check
  `n_iter_ == max_iter`.
- `svd_method="lapack"` uses the shared engine's exact Jacobi SVD (not LAPACK),
  `"randomized"` the shared randomized SVD with one seeded RandomState stream
  across EM iterations, as in sklearn. Inverses and log-determinants use
  mojo-linalg's pivoted symmetric LDL^T; singular systems raise `LinAlgError`.

## Validation

`pixi run parity` runs 186 cases against the pinned source: tall and wide
shapes, both SVD methods, `n_components` None/0/1/2/3, both rotations, tight
`tol`, truncated `max_iter`, custom `noise_variance_init`, several
`iterated_power`/seeds, a constant feature (SMALL floor) and scaled data, in
Float64 and Float32 input. The EM iteration count must match exactly. Values
use relative/absolute tolerance 1e-7 (scaled by the data magnitude), 1e-6 for
precision and latent scores. Component rows and latent columns are compared
up to one sign per factor because the reference keeps LAPACK's arbitrary
singular-vector signs. Wide `n_components=None` fits are excluded: centered
data of rank n-1 then contributes a rounding-noise `log(0)` term to the
likelihood of both implementations.
