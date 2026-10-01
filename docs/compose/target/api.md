# TransformedTargetRegressor

`moxie.compose.TransformedTargetRegressor` regresses on a transformed target:
`fit` computes `regressor.fit(X, transformer.transform(y))` (or `func(y)`),
`predict` computes `transformer.inverse_transform(regressor.predict(X))` (or
`inverse_func(...)`) and `score` is R² in the original target space. Reference:
`sklearn/compose/_target.py` at the revision in `../../interfaces.md`.

```mojo
from moxie.compose import TransformedTargetRegressor

def log_y(y: NDArray[DType.float64]) raises -> NDArray[DType.float64]: ...
def exp_y(y: NDArray[DType.float64]) raises -> NDArray[DType.float64]: ...

var model = TransformedTargetRegressor(func=log_y, inverse_func=exp_y)  # LinearRegression
model.fit(X, y)                     # y: (n,) or (n, k)
var y_hat = model.predict(X_new)    # same rank as the training y
var r2 = model.score(X, y)

var ridge_like = TransformedTargetRegressor(MyRegressor(...), transformer=MyScaler())
```

## Type parameters and constructor

`TransformedTargetRegressor[dtype=float64, R=LinearRegression[dtype], T=TargetFunctionTransformer[dtype]]`;
`R` and `T` are inferred from the constructor arguments.

| Parameter | Default | Meaning (as sklearn) |
| --- | --- | --- |
| `regressor` | `None` | Regressor configuration, cloned at every fit. `None` = `LinearRegression()` |
| `transformer` | `None` | Target transformer, cloned at every fit; exclusive with `func`/`inverse_func` |
| `func`, `inverse_func` | `None` | `TargetFunction[dtype]`: `def(NDArray[dtype]) raises -> NDArray[dtype]`; both or neither |
| `check_inverse` | `True` | Warn when the sampled round trip `inverse(transform(y[::max(1, n // 10)]))` is not `allclose` |

Learned: `regressor_`, `transformer_` (`Optional`, set together by a successful
fit), `n_features_in_`, optional `feature_names_in_`, `warnings_`. Methods: `fit(X, y)`, `fit(X, y, sample_weight=...)`
(only when `R` is a `WeightedSupervisedEstimator`; weights reach the regressor,
not the transformer, as in sklearn), `predict(X)`, `score(X, y, sample_weight=None)`,
`get_params`, `set_params`, `__sklearn_clone__`.

## Mojo differences

- Component types are static. `regressor=None` requires `R` to be the default
  `LinearRegression[dtype]` and `transformer=None` (identity or functions)
  requires `T` to be the default `TargetFunctionTransformer[dtype]`; any other
  combination raises `ValueError` at fit. `R.Target`/`R.Prediction` and
  `T.Input`/`T.Transformed` must be `NDArray[dtype]` (compile-time check);
  `X` is passed through unchanged as `R.Input`.
- `transformer_` built from functions is a `TargetFunctionTransformer`: the
  compose-local equivalent of `FunctionTransformer(func, inverse_func,
  validate=True, check_inverse=check_inverse)` (2-D, nonempty, finite input;
  fitted feature count; its own round-trip warning with rtol=1e-7, atol=1e-9 on
  `X[::max(1, n // 100)]`). It has no `kw_args`, `feature_names_out`,
  `accept_sparse` or `set_output`.
- Parameters: `get_params()` returns a `TransformedTargetRegressorParams`
  record holding copies of the component configurations; nested parameters are
  `params.regressor.value().get_params()`. Keyword `set_params` updates only
  supplied values; `regressor_params=`/`transformer_params=` take the nested
  records and are applied after component replacement (sklearn's ordering).
  Reset a value to `None` through the record overload.
- Warnings (`UserWarning`, `UndefinedMetricWarning`) are printed to stderr with
  their category prefix; a successful fit keeps its warnings in `warnings_`.
  Errors are Mojo `Error`s whose text starts with the sklearn exception type and
  message (`ValueError: Input y contains NaN.`).
- `n_features_in_` is recorded from `X.shape[1]` when `R.Input` is a dense
  matrix (sklearn reads `regressor_.n_features_in_`). For matrix regressors,
  `fit`, weighted `fit`, `predict` and `score` also accept native `ColumnTable`.
  These overloads retain `feature_names_in_` and require matching names/order
  at prediction. An array refit clears names. Metadata routing and sparse `X`
  remain unavailable; pandas/polars integration is excluded.
- `score` accumulates R² in Float64 (sklearn uses the input floating dtype).
- A failed fit leaves the previous learned state untouched (sklearn may leave
  a new `transformer_` behind).

## Evidence

`pixi run test` runs `tests/compose/target/contracts.mojo` (lifecycle, shapes, functions,
warnings, configuration and target errors, cloning, parameters, custom
components, weights, Float32) and the example. `pixi run parity` compares 38
Float32/Float64 cases with the pinned `_target.py`, `_function_transformer.py`
and `linear_model/_base.py`; tolerances and the documented warning/message
exemptions are stated in `tests/compose/target/parity.py`.
