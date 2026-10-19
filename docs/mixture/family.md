# Mixture family contracts

| Trait | Shared capabilities | Additional methods |
| --- | --- | --- |
| `MixtureDensity` | `DensityMixin`, `Predictor`, `ProbabilityPredictor` | `sample` |
| `InformationCriteria` | `InputEstimator` | `aic`, `bic` |

Both concrete estimators implement `MixtureDensity`. Only GaussianMixture
implements `InformationCriteria`. Responsibilities describe mixture components,
not supervised classes. Result types are estimator-associated types; sampling
returns points and component labels.

Reference revision: scikit-learn `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
The local reference files are `sklearn/mixture/_base.py`,
`_gaussian_mixture.py`, `_bayesian_mixture.py`, the mixture tests, and
`doc/modules/mixture.rst`. Shared estimator behavior follows `sklearn/base.py`
and `doc/developers/develop.rst`. The retained scikit-learn notice is
`LICENSES/mixture-scikit-learn.txt` in Moxie.

See the [public API](api-surface.md), [shared numerical contracts](plumbing.md),
and [Moxie estimator interfaces](../interfaces.md). The original generic
framework tests and estimator example were byte-identical to Moxie's existing
copies and remain there. Mixture-specific tests and the plumbing example are
under `tests/mixture/` and `examples/mixture_plumbing.mojo`.
