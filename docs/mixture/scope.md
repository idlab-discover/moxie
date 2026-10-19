# Implemented scope

Both dense estimators and their shared numerical/lifecycle modules are owned by
Moxie core. See the [guide](README.md), [Gaussian scope](gaussian/scope.md), and
[Bayesian scope](bayesian/scope.md). Sparse input is unsupported and is never
silently densified. Optional information criteria belong only to Gaussian.

`MixtureDensity` and `InformationCriteria` remain public capability contracts.
The private `BaseMixture` Python base class is not exported. Mutating `fit`,
`fit_with_rng`, and `set_params` return a mutable borrow of the estimator under
the current Moxie contract. `fit_predict` returns labels. Typed parameter
records replace dynamic keyword dictionaries; Python metadata routing and
reflection remain absent.
