# linear_model family interfaces

The verified family assembly exports dense `LinearRegression`, `Ridge`,
`RidgeClassifier`, `ridge_regression`, `ElasticNet`, `Lasso` and `BayesianRidge`
from Moxie-owned implementation modules. Run
`pixi run build`, then use `-I builds/deps` and import from `moxie.linear_model`.
Historical `_api.mojo` declarations remain in the original Git history;
they are not installed or supported public algorithms.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](interfaces.md) for typed parameter records,
borrowed fluent fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `LinearRegressor` | `RegressorMixin` | Inherited only |
| `LinearClassifier` | `ClassifierMixin, DecisionFunction` | Inherited only |
| `ProbabilisticLinearClassifier` | `LinearClassifier, LogProbabilityPredictor` | Inherited only |
| `OnlineLinearRegressor` | `IncrementalRegressor` | Inherited only |
| `OnlineLinearClassifier` | `IncrementalClassifier, DecisionFunction` | Inherited only |

## Behavioral distinctions

LinearModel supplies predict; LinearClassifierMixin supplies decisions. Probability depends on estimator/loss. GLMs, robust models, multi-task and CV estimators retain their own constructors. coef_ and intercept_ are learned state.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../../../../scikit-learn/sklearn/linear_model`.
- Guide and mathematical references: `../../../../scikit-learn/doc/modules/linear_model.rst`.
- Shared contracts: `../../../../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`LinearModelLoss`, `CD_Algo`, `ElasticNet`, `Lasso`, `LinearModelCV`, `LassoCV`, `ElasticNetCV`, `MultiTaskElasticNet`, `MultiTaskLasso`, `MultiTaskElasticNetCV`, `MultiTaskLassoCV`, `Lars`, `LassoLars`, `LarsCV`, `LassoLarsCV`, `LassoLarsIC`, `BaseSGD`, `BaseSGDClassifier`, `SGDClassifier`, `BaseSGDRegressor`, `SGDRegressor`, `SGDOneClassSVM`, `LinearModel`, `MultiOutputLinearModel`, `LinearClassifierMixin`, `SparseCoefMixin`, `LinearRegression`, `OrthogonalMatchingPursuit`, `OrthogonalMatchingPursuitCV`, `Perceptron`, `PassiveAggressiveClassifier`, `PassiveAggressiveRegressor`, `Ridge`, `RidgeClassifier`, `RidgeCV`, `RidgeClassifierCV`, `RANSACRegressor`, `TheilSenRegressor`, `BayesianRidge`, `ARDRegression`, `QuantileRegressor`, `HuberRegressor`, `LogisticRegression`, `LogisticRegressionCV`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.

Dense storage, shared owner interfaces and child dependencies are defined in
[plumbing.md](plumbing.md). Read [scope.md](scope.md) for tested scope and sparse
exclusions before implementing an estimator.
