"""Regress on log(y) and predict in the original target space."""
from numojo import NDArray, Shape, zeros
from std.math import log, exp
from moxie.compose import TransformedTargetRegressor


def elementwise_log(y: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
    var out = y.copy()
    for i in range(out.size):
        out.unsafe_set(i, log(out.unsafe_get(i)))
    return out^


def elementwise_exp(y: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
    var out = y.copy()
    for i in range(out.size):
        out.unsafe_set(i, exp(out.unsafe_get(i)))
    return out^


def main() raises:
    # y grows exponentially with the single feature: y = exp(0.5 * x + 1).
    var X = zeros[DType.float64](Shape(5, 1))
    var y = zeros[DType.float64](Shape(5))
    for i in range(5):
        X.unsafe_set(i, Float64(i))
        y.unsafe_set(i, exp(0.5 * Float64(i) + 1.0))

    # regressor=None uses LinearRegression; func/inverse_func transform y.
    var model = TransformedTargetRegressor(func=elementwise_log, inverse_func=elementwise_exp)
    _ = model.fit(X, y)
    print("coef_ on log(y):", model.regressor_.value().coef_.unsafe_get(0))
    print("R^2 in target space:", model.score(X, y))
    var query = zeros[DType.float64](Shape(1, 1))
    query.unsafe_set(0, 5.0)
    print("prediction at x=5:", model.predict(query).unsafe_get(0), "expected:", exp(3.5))
