import numojo as nm
from std.math import isnan, abs
from std.testing import assert_equal, assert_false, assert_true, assert_raises
from moxie import clone
from moxie.impute import KNNImputer
from moxie.impute._plumbing import MissingValue
from moxie.neighbors._contracts import DenseMetric
from moxie.neighbors._distance import Metric
from moxie.neighbors._query_engine import NeighborWeights


struct Ones(NeighborWeights):
    def __init__(out self):
        pass
    def apply(self, distances: nm.NDArray[DType.float64], offsets: List[Int]) raises -> nm.NDArray[DType.float64]:
        return nm.ones[DType.float64](nm.Shape(distances.shape[0]))


struct WrongWeights(NeighborWeights):
    def __init__(out self):
        pass
    def apply(self, distances: nm.NDArray[DType.float64], offsets: List[Int]) raises -> nm.NDArray[DType.float64]:
        return nm.ones[DType.float64](nm.Shape(1, distances.shape[0]))


struct L1(DenseMetric):
    comptime dtype = DType.float64
    def __init__(out self):
        pass
    def validate_features(self, n_features: Int) raises:
        pass
    def distance(self, x: nm.NDArray[DType.float64], i: Int, y: nm.NDArray[DType.float64], j: Int) raises -> Float64:
        var total = Float64(0)
        for col in range(x.shape[1]):
            var a = x.unsafe_get(i * Int(x.strides[0]) + col * Int(x.strides[1]))
            var b = y.unsafe_get(j * Int(y.strides[0]) + col * Int(y.strides[1]))
            if not isnan(a) and not isnan(b): total += abs(a-b)
        return total


def check[dtype: DType]() raises:
    var X = nm.zeros[dtype](nm.Shape(4, 3))
    var data: List[Float64] = [1, 2, -1, 3, 4, 3, -1, 6, 5, 8, 8, 7]
    for i in range(12): X.unsafe_set(i, data[i].cast[dtype]())
    var model = KNNImputer[dtype](n_neighbors=2, missing_values=MissingValue[dtype](-1), copy=False)
    with assert_raises(): _ = model.transform(X)
    var output = model.fit_transform(X)
    assert_equal(output.unsafe_get(2), SIMD[dtype, 1](4))
    assert_equal(output.unsafe_get(6), SIMD[dtype, 1](5.5))
    assert_equal(X.unsafe_get(2), SIMD[dtype, 1](-1))
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.get_params().n_neighbors, 2)
    var fitted_copy = model.copy()
    X.unsafe_set(5, 99)
    assert_equal(model.transform(X).unsafe_get(2), SIMD[dtype, 1](4))
    assert_equal(fitted_copy.transform(X).unsafe_get(2), SIMD[dtype, 1](4))
    _ = model.set_params(n_neighbors=1)
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises(): _ = model.transform(X)
    var invalid = KNNImputer[dtype](n_neighbors=0)
    with assert_raises(): _ = invalid.fit(X)
    _ = invalid.set_params(n_neighbors=1, metric="euclidean")
    with assert_raises(): _ = invalid.fit(X)
    _ = invalid.set_params(metric="nan_euclidean", weights="invalid")
    with assert_raises(): _ = invalid.fit(X)

    # A globally nearest row lacks the target. Select eligible donors first.
    var train = nm.zeros[dtype](nm.Shape(4, 3))
    var rows: List[Float64] = [-1, 0, -1, 100, 0.125, -1, 10, 10, -1, 20, 20, -1]
    for i in range(12): train.unsafe_set(i, rows[i].cast[dtype]())
    var query = nm.zeros[dtype](nm.Shape(1, 3))
    query.unsafe_set(0, -1)
    query.unsafe_set(2, 42)
    _ = model.set_params(add_indicator=True)
    var names: List[String] = ["a", "b", "empty"]
    model.fit_named(train, names)
    names[0] = "changed"
    assert_equal(model.get_feature_names_out(), [String("a"), String("b"), String("missingindicator_a"), String("missingindicator_empty")])
    with assert_raises(): _ = model.get_feature_names_out(names.copy())
    var predicted = model.transform(query)
    assert_equal(predicted.shape[1], 4)
    assert_equal(predicted.unsafe_get(0), SIMD[dtype, 1](100))
    assert_equal(predicted.unsafe_get(2), SIMD[dtype, 1](1))
    assert_equal(predicted.unsafe_get(3), SIMD[dtype, 1](0))
    _ = model.set_params(keep_empty_features=True)
    _ = model.fit(train)
    predicted = model.transform(query)
    assert_equal(predicted.shape[1], 5)
    assert_equal(predicted.unsafe_get(2), SIMD[dtype, 1](0))
    assert_equal(model.get_feature_names_out(), [String("x0"), String("x1"), String("missingindicator_x0"), String("missingindicator_x2")])
    var bad = nm.full[dtype](nm.Shape(1, 3), Float64("inf").cast[dtype]())
    with assert_raises(): _ = model.fit(bad)
    assert_true(model.__sklearn_is_fitted__())
    with assert_raises(): _ = model.transform(bad)
    var wrong = nm.zeros[dtype](nm.Shape(1, 2))
    with assert_raises(): _ = model.transform(wrong)
    var empty = nm.full[dtype](nm.Shape(2, 3), SIMD[dtype, 1](-1))
    _ = model.set_params(keep_empty_features=False, add_indicator=False)
    assert_equal(model.fit_transform(empty).shape[1], 0)
    _ = model.set_params(keep_empty_features=True)
    var zeros = model.fit_transform(empty)
    for i in range(zeros.size): assert_equal(zeros.unsafe_get(i), SIMD[dtype, 1](0))
    # Equal-distance boundaries deterministically select the first fitted donor.
    var tied = nm.zeros[dtype](nm.Shape(3, 2))
    tied.unsafe_set(0, 10)
    tied.unsafe_set(2, 20)
    tied.unsafe_set(4, 30)
    var many = nm.zeros[dtype](nm.Shape(257, 2))
    for i in range(257): many.unsafe_set(2 * i, -1)
    _ = model.set_params(keep_empty_features=False)
    _ = model.fit(tied)
    assert_false(Bool(model.feature_names_in_))
    var batched = model.transform(many)
    for i in range(257): assert_equal(batched.unsafe_get(2 * i), SIMD[dtype, 1](10))
    var reversed = many.view_with_layout(nm.Shape(257, 2), nm.Strides(-2, 1), 512)
    assert_equal(model.transform(reversed).unsafe_get(0), SIMD[dtype, 1](10))
    # Negative strides and offsets retain logical row/column access.
    var view = train.view_with_layout(nm.Shape(3, 3), nm.Strides(-3, 1), 9)
    _ = model.fit(view)
    assert_equal(model.transform(query).unsafe_get(0), SIMD[dtype, 1](100))


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    # Reciprocal overflow must use sklearn's exact-match weighting convention.
    var tiny = nm.zeros[DType.float32](nm.Shape(2, 2))
    tiny.unsafe_set(0, 2)
    tiny.unsafe_set(1, Float32(1e-40))
    tiny.unsafe_set(2, 6)
    tiny.unsafe_set(3, Float32(2e-40))
    var tiny_query = nm.zeros[DType.float32](nm.Shape(1, 2))
    tiny_query.unsafe_set(0, Float64("nan").cast[DType.float32]())
    var tiny_model = KNNImputer[DType.float32](weights="distance", n_neighbors=2)
    _ = tiny_model.fit(tiny)
    assert_equal(tiny_model.transform(tiny_query).unsafe_get(0), Float32(4))
    var X = nm.zeros[DType.float64](nm.Shape(3, 2))
    X.unsafe_set(0, 2)
    X.unsafe_set(2, 4)
    X.unsafe_set(4, 8)
    X.unsafe_set(5, Float64("nan"))
    var query = nm.full[DType.float64](nm.Shape(1, 2), Float64("nan"))
    query.unsafe_set(1, 0)
    # Builtin uniform ignores NaN distance; custom ones deliberately includes it.
    var uniform = KNNImputer(n_neighbors=3)
    _ = uniform.fit(X)
    assert_equal(uniform.transform(query).unsafe_get(0), Float64(3))
    var custom = KNNImputer[DType.float64, Metric[DType.float64], Ones](n_neighbors=3, weights="callable", weights_callback=Ones())
    _ = custom.fit(X)
    assert_equal(custom.transform(query).unsafe_get(0), Float64(14)/3)
    var wrong = KNNImputer[DType.float64, Metric[DType.float64], WrongWeights](weights="callable", weights_callback=WrongWeights())
    _ = wrong.fit(X)
    with assert_raises(): _ = wrong.transform(query)
    query.unsafe_set(1, Float64("nan"))
    assert_equal(uniform.transform(query).unsafe_get(0), Float64(14)/3)
    var metric = KNNImputer[DType.float64, L1](metric="callable", metric_callback=L1(), n_neighbors=3)
    _ = metric.fit(X)
    assert_equal(metric.transform(query).unsafe_get(0), Float64(14)/3)
    print("PASS: KNN donors, ownership, lifecycle, callbacks, names and empty features")
