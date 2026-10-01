"""The family namespace exposes and composes the real public estimators."""
from moxie.compose import ColumnTransformer, ColumnEntry, ColumnTable, make_column_selector, make_column_transformer, TransformedTargetRegressor
from numojo import Shape, zeros
from std.testing import assert_almost_equal, assert_equal, assert_false, assert_raises
from moxie.base import clone


def main() raises:
    var X = zeros[DType.float64](Shape(4, 2))
    var y = zeros[DType.float64](Shape(4))
    for i in range(4):
        X.unsafe_set(i * 2, Float64(i))
        X.unsafe_set(i * 2 + 1, Float64(10 + i))
        y.unsafe_set(i, Float64(2 * i + 1))
    var columns = make_column_transformer(ColumnEntry("passthrough", List[Int](0, __list_literal__=None)))
    var Xt = columns.fit_transform(X)
    var model = TransformedTargetRegressor()
    model.fit(Xt, y)
    var prediction = model.predict(columns.transform(X))
    for i in range(4):
        assert_almost_equal(Float64(prediction.unsafe_get(i)), Float64(y.unsafe_get(i)), atol=1e-10)
    var names: List[String] = ["age", "income"]
    var table = ColumnTable(X, names)
    model.fit(table, y)
    assert_equal(model.feature_names_in_.value()[0], "age")
    assert_almost_equal(model.score(table, y), 1.0, atol=1e-10)
    var reversed: List[String] = ["income", "age"]
    with assert_raises():
        _ = model.predict(ColumnTable(X, reversed))
    var weights = zeros[DType.float64](Shape(4))
    for i in range(4):
        weights.unsafe_set(i, 1)
    model.fit(table, y, sample_weight=weights)
    assert_equal(model.feature_names_in_.value()[1], "income")
    var fresh = clone(model)
    assert_false(Bool(fresh.feature_names_in_))
    model.fit(X, y)
    assert_false(Bool(model.feature_names_in_))
    print("PASS: unified compose public API and named target regression")
