# Supported public API

`moxie.linear_model` exports six implemented estimators and one solver function:

| Public names | Contract |
| --- | --- |
| LinearRegression, LinearRegressionParams | [Least squares](../algorithms/least_squares/docs/api.md) |
| Ridge, RidgeParams, RidgeClassifier, RidgeClassifierParams, ridge_regression, RidgeRegressionResult, ClassWeight | [Ridge](../algorithms/ridge/docs/api.md) |
| ElasticNet, ElasticNetParams, Lasso, LassoParams | [Cyclic coordinate estimators](../algorithms/elastic_net/docs/api.md) |
| BayesianRidge, BayesianRidgeParams | [Bayesian ridge](../algorithms/bayesian/docs/api.md) |

The capability traits `LinearRegressor`, `LinearClassifier`,
`ProbabilisticLinearClassifier`, `OnlineLinearRegressor`, and
`OnlineLinearClassifier` remain public. A trait is not an implemented estimator.

The typed parameter records, borrowed fluent mutation returns, errors, array
shapes and unsupported modes are documented in the child contracts and Moxie's
[shared interfaces](../../../docs/interfaces.md). The historical `_api.mojo`
declarations are not exported or installed; their signatures remain available
in the standalone repository's recorded Git history. Missing dense APIs are
listed in [coverage.md](coverage.md). No full scikit-learn parity is claimed.
