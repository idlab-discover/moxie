import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier

def main() raises:
    var X = nm.zeros[DType.float32](nm.Shape(6, 1))
    var y = nm.zeros[DType.float64](nm.Shape(6))
    for i in range(6):
        X.unsafe_set(i, Float32(10 + 10 * (i % 3)))
        y.unsafe_set(i, Float64(i % 3 == 1))
    var model = DecisionTreeClassifier(categorical_features=Optional[List[Int]]([0]), random_state=0)
    _ = model.fit(X, y)
    assert_equal(model.tree_.n_categories[0], 3)
    assert_true(model.is_categorical_.value()[0])
    var prediction = model.predict(X)
    for i in range(6):
        assert_equal(prediction.unsafe_get(i), y.unsafe_get(i))
    var query = nm.full[DType.float32](nm.Shape(2, 1), 999)
    query.unsafe_set(1, Float32(Float64("nan")))
    var unknown = model.apply(query)
    assert_equal(unknown.unsafe_get(0), unknown.unsafe_get(1))
    var copied = model.__sklearn_clone__()
    _ = copied.fit(X, y)
    var clone = model.__sklearn_clone__()
    assert_true(not clone.__sklearn_is_fitted__())
    X.unsafe_set(0, -42)
    _ = model.fit(X, y)
    assert_equal(copied.tree_.n_categories[0], 3)
    assert_equal(model.tree_.n_categories[0], 4)
    var params = model.get_params()
    params.categorical_features = None
    _ = model.set_params(params)
    _ = model.fit(X, y)
    assert_true(not model.is_categorical_)
    assert_equal(model.tree_.n_categories[0], -1)
    var invalid = DecisionTreeClassifier(categorical_features=Optional[List[Int]]([1]))
    with assert_raises():
        _ = invalid.fit(X, y)
    var monotonic = DecisionTreeClassifier(categorical_features=Optional[List[Int]]([0]), monotonic_cst=Optional[List[Int]]([1]))
    with assert_raises():
        _ = monotonic.fit(X, y)
    var zero_monotonic = DecisionTreeClassifier(categorical_features=Optional[List[Int]]([0]), monotonic_cst=Optional[List[Int]]([0]))
    _ = zero_monotonic.fit(X, y)
    with assert_raises():
        _ = zero_monotonic.predict(query)
    var mae = DecisionTreeRegressor(categorical_features=Optional[List[Int]]([0]), criterion="absolute_error")
    with assert_raises():
        _ = mae.fit(X, y)
    var multiclass = DecisionTreeClassifier(categorical_features=Optional[List[Int]]([0]))
    for i in range(6):
        y.unsafe_set(i, Float64(i % 3))
    with assert_raises():
        _ = multiclass.fit(X, y)
    var multi = nm.zeros[DType.float64](nm.Shape(6, 2))
    with assert_raises():
        _ = multiclass.fit(X, multi)
    var failed_before = copied.tree_.node_count
    with assert_raises():
        _ = copied.fit(nm.full[DType.float32](nm.Shape(6, 1), Float32(Float64("inf"))), y)
    assert_equal(copied.tree_.node_count, failed_before)
    var backing = nm.zeros[DType.float32](nm.Shape(12))
    var strided = backing.view_with_layout(nm.Shape(6, 1), nm.NDArrayStrides(2, 1), 0)
    for i in range(6):
        strided.unsafe_set(i * 2, X.unsafe_get(i))
    var random = ExtraTreeClassifier(categorical_features=Optional[List[Int]]([0]), random_state=0)
    _ = random.fit(strided, y)
    var large = nm.zeros[DType.float32](nm.Shape(257, 1))
    var target = nm.zeros[DType.float64](nm.Shape(257))
    for i in range(257):
        large.unsafe_set(i, Float32(i))
        target.unsafe_set(i, Float64(i % 2))
    with assert_raises():
        _ = copied.fit(large, target)
    var limit = large.view_with_layout(nm.Shape(256, 1), nm.NDArrayStrides(1, 1), 0)
    var limit_target = target.view_with_layout(nm.Shape(256), nm.NDArrayStrides(1), 0)
    _ = copied.fit(limit, limit_target)
    assert_equal(copied.tree_.n_categories[0], 256)
    _ = random.fit(large, target)
    assert_equal(random.tree_.n_categories[0], 257)
    print("PASS categorical encoding, unknown routing, ownership, lifecycle and rejection contracts")
