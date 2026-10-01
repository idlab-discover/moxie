"""Real compose/linear_model consumer; no estimator doubles or Python runtime."""
from moxie.base import clone
from moxie.compose import ColumnEntry, ColumnTransformer, NoTransformer, TargetFunctionTransformer, TransformedTargetRegressor
from moxie.linear_model import LinearRegression
from numojo import NDArray, Shape, zeros
from std.math import sqrt
from std.testing import assert_equal, assert_false, assert_true, assert_raises
from parity_io import ResultLine


def root[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    var output = x.copy()
    for i in range(output.size):
        output.unsafe_set(i, sqrt(output.unsafe_get(i)))
    return output^


def square[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    var output = x.copy()
    for i in range(output.size):
        var value = output.unsafe_get(i)
        output.unsafe_set(i, value * value)
    return output^


def run[dtype: DType](jobs: Int) raises:
    comptime Function = TargetFunctionTransformer[dtype]
    var feature = Function(root[dtype], square[dtype])
    var columns = ColumnTransformer(
        ColumnEntry[Function, dtype]("root", feature.copy(), [0]),
        ColumnEntry[NoTransformer[dtype], dtype]("raw", "passthrough", [2]),
        n_jobs=jobs,
    )
    var regressor = LinearRegression[dtype]()
    var model = TransformedTargetRegressor[dtype](
        regressor.copy(), transformer=Function(root[dtype], square[dtype]))
    var X = zeros[dtype](Shape(12, 3))
    var y = zeros[dtype](Shape(12))
    var weights = zeros[dtype](Shape(12))
    for i in range(12):
        X.unsafe_set(i * 3, Float64((i + 1) * (i + 1)).cast[dtype]())
        X.unsafe_set(i * 3 + 1, Float64(100 + i).cast[dtype]())
        X.unsafe_set(i * 3 + 2, Float64(i % 3).cast[dtype]())
        var value = Float64(3 + 2 * (i + 1) + i % 3)
        y.unsafe_set(i, (value * value + Float64(i % 2)).cast[dtype]())
        weights.unsafe_set(i, Float64(i % 4).cast[dtype]())
    with assert_raises():
        _ = columns.transform(X)
    var transformed = columns.fit_transform(X, y.copy())
    assert_false(feature.__sklearn_is_fitted__())
    assert_true(columns.named_transformers_[Function]("root").__sklearn_is_fitted__())
    model.fit(transformed, y, sample_weight=weights)
    assert_false(regressor.__sklearn_is_fitted__())
    var fresh_columns = clone(columns)
    var fresh_model = clone(model)
    assert_false(fresh_columns.__sklearn_is_fitted__())
    assert_false(fresh_model.__sklearn_is_fitted__())
    with assert_raises():
        _ = fresh_model.predict(transformed)
    var result = ResultLine(String(Int(dtype == DType.float32), "-", jobs))
    result.add("transformed", transformed)
    result.add("prediction", model.predict(columns.transform(X)))
    result.scalar("score", model.score(transformed, y, sample_weight=weights.copy()))
    assert_equal(columns.output_indices_.value()["root"].start, 0)
    assert_equal(columns.output_indices_.value()["raw"].stop, 2)
    # A failed fit must preserve the last successfully fitted model.
    var empty = zeros[dtype](Shape(0))
    with assert_raises():
        model.fit(transformed, empty)
    result.add("after_failure", model.predict(transformed))
    # Normal fits replace learned state; cloned configuration reproduces a refit.
    for i in range(y.size):
        y.unsafe_set(i, y.unsafe_get(i) * 4)
    columns.fit(X, y.copy())
    model.fit(columns.transform(X), y)
    fresh_columns.fit(X, y.copy())
    fresh_model.fit(fresh_columns.transform(X), y)
    result.add("refit", model.predict(columns.transform(X)))
    result.add("clone_refit", fresh_model.predict(fresh_columns.transform(X)))
    result.emit()


def main() raises:
    run[DType.float64](1)
    run[DType.float64](2)
    run[DType.float32](1)
    run[DType.float32](2)
