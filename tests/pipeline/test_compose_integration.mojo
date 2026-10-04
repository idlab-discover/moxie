"""Compose and FeatureUnion feeding a real LinearRegression in both dense dtypes."""
from moxie.base import clone
from moxie.compose import ColumnEntry, ColumnTransformer, NoTransformer, ColumnCons, EndColumns
from moxie.linear_model import LinearRegression
from moxie.pipeline import Pipeline, FeatureUnion, UnionPair, union_step
from numojo import Shape, zeros
from std.testing import assert_equal, assert_false, assert_raises
from parity_io import ResultLine


def check[dtype: DType]() raises:
    var left = ColumnTransformer(
        ColumnEntry[NoTransformer[dtype], dtype]("selected", "passthrough", [2, 0]))
    var right = ColumnTransformer(
        ColumnEntry[NoTransformer[dtype], dtype]("selected", "passthrough", [1]))
    var weights = Dict[String, Float64]()
    weights["right"] = 0.5
    var union = FeatureUnion(UnionPair(union_step[dtype]("left", left^), union_step[dtype]("right", right^)),
                             transformer_weights=weights.copy())
    var pipe = Pipeline(union^, LinearRegression[dtype]())
    var X = zeros[dtype](Shape(12, 3))
    var y = zeros[dtype](Shape(12))
    for i in range(12):
        X.unsafe_set(i * 3, Scalar[dtype](i))
        X.unsafe_set(i * 3 + 1, Scalar[dtype](i % 3))
        X.unsafe_set(i * 3 + 2, Scalar[dtype](i % 2))
        y.unsafe_set(i, Scalar[dtype](3 + 2 * i - 4 * (i % 3) + 5 * (i % 2)))
    with assert_raises():
        _ = pipe.n_features_in_()
    with assert_raises():
        _ = pipe.transformer.get_feature_names_out()
    _ = pipe.fit(X, y)
    assert_equal(pipe.n_features_in_(), 3)
    var names = pipe.transformer.get_feature_names_out()
    assert_equal(names[0], "left__selected__x2")
    assert_equal(names[1], "left__selected__x0")
    assert_equal(names[2], "right__selected__x1")
    var given: List[String] = ["a", "b", "c"]
    var named = pipe.transformer.get_feature_names_out(given.copy())
    assert_equal(named[0], "left__selected__c")
    assert_equal(named[2], "right__selected__b")
    assert_equal(given[0], "a")
    with assert_raises():
        _ = pipe.transformer.get_feature_names_out(List[String]())
    with assert_raises():
        _ = pipe.get_feature_names_out()
    # The shared optional names contract also works for nested ColumnTransformer.
    var inner = clone(pipe.transformer.transformer_list.first.estimator)
    comptime Inner = ColumnTransformer[ColumnCons[NoTransformer[dtype], EndColumns[dtype]]]
    var outer = ColumnTransformer(ColumnEntry[Inner, dtype]("nested", inner^, [0, 1, 2]))
    _ = outer.fit(X)
    assert_equal(outer.get_feature_names_out()[0], "nested__selected__x2")
    var output = ResultLine(String(Int(dtype == DType.float32)))
    output.add("transformed", pipe.transformer.transform(X))
    output.add("prediction", pipe.predict(X))
    output.scalar("score", pipe.score(X, y))
    output.scalar("features", Float64(pipe.n_features_in_()))
    var fresh = clone(pipe)
    assert_false(fresh.__sklearn_is_fitted__())
    with assert_raises():
        _ = fresh.n_features_in_()
    for i in range(y.size):
        y.unsafe_set(i, y.unsafe_get(i) + 7)
    _ = fresh.fit(X, y)
    output.add("clone_refit", fresh.predict(X))
    output.add("original", pipe.predict(X))
    _ = pipe.fit(X, y)
    output.add("refit", pipe.predict(X))
    output.emit()


def main() raises:
    check[DType.float64]()
    check[DType.float32]()
