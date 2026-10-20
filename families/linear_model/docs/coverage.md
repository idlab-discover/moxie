# Dense API coverage

The target is every dense public API in the local scikit-learn
`sklearn/linear_model/__init__.py` at revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

Run `pixi run build` followed by `pixi run python tools/coverage.py` to compare
that reference with the actual generated public imports. Export availability
is not evidence of complete behavior. Historical `_api.mojo` declarations remain in Git history and are excluded
from the assembled package.

| Public names | Current implementation and remaining dense work |
| --- | --- |
| LinearRegression | Dense least squares and nonnegative least squares; see least_squares scope and parity suite |
| Ridge, RidgeClassifier, ridge_regression | Dense direct, iterative and positive solvers; SAG/SAGA missing |
| ElasticNet, Lasso | Cyclic residual solver; random selection and Gram input missing |
| enet_path, lasso_path | Missing |
| MultiTaskElasticNet, MultiTaskLasso | Missing block coordinate solver and estimators |
| RidgeCV, RidgeClassifierCV | Missing estimators and validation integration |
| ElasticNetCV, LassoCV, MultiTaskElasticNetCV, MultiTaskLassoCV | Missing |
| Lars, LassoLars, lars_path, lars_path_gram | Missing |
| LarsCV, LassoLarsCV, LassoLarsIC | Missing |
| OrthogonalMatchingPursuit, orthogonal_mp, orthogonal_mp_gram, OrthogonalMatchingPursuitCV | Missing |
| BayesianRidge | Dense empirical-Bayes fitting, full posterior covariance and predictive uncertainty; see [API and validation](../algorithms/bayesian/docs/api.md) |
| ARDRegression | Missing |
| LogisticRegression, LogisticRegressionCV | Missing |
| PoissonRegressor, GammaRegressor, TweedieRegressor | Missing |
| HuberRegressor, QuantileRegressor, RANSACRegressor, TheilSenRegressor | Missing |
| SGDClassifier, SGDRegressor, SGDOneClassSVM | Missing |
| Perceptron, PassiveAggressiveClassifier, PassiveAggressiveRegressor | Missing |

Sparse matrices and sparse coefficient conversion remain outside the dense
scope. The missing dense entries above are unfinished work, not exclusions.
Moxie owns the supported subset and all remaining family work. Integration
does not imply that the unfinished dense algorithms are implemented.

Build and native contract commands are `pixi run build` and `pixi run test`.
The test command runs the family contracts, real fluent estimator consumers,
child contracts, isolated namespace consumers and rejection checks.
`pixi run parity` runs the child differential suites against local reference
sources. `pixi run start` runs the public estimator example. Keep raw results
under ignored `builds/`; do not infer acceptance from this inventory.
