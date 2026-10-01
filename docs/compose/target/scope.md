# Scope

Dense `NDArray[float32/float64]` targets of rank 1 or 2 (rank ≥ 3 reaches the
transformer, which for functions rejects it as sklearn does). Any regressor
satisfying `RegressorMixin` with `NDArray[dtype]` targets and predictions;
any `InverseTransformer & UnsupervisedEstimator` target transformer on
`NDArray[dtype]`.

Not provided: sparse `X` (owned by the regressor), sparse `y` (rejected by
the reference too), metadata routing objects and
`**fit_params`/`**predict_params` beyond `sample_weight`, HTML display.

Dependencies: local compose helpers and the live LinearRegression dependency
from moxie-linear_model, declared with its numerical closure in Moxie’s
`DEPENDENCIES.json`. Run `pixi run test` and `pixi run parity` from Moxie.

Matrix regressors also accept native `ColumnTable` through fit/predict/score
overloads, with owned feature names and prediction name/order validation.
Pandas/polars bindings are intentionally excluded.
