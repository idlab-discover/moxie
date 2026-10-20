from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.math import abs
from numojo import zeros, Shape, NDArrayStrides
from moxie.base import clone
from moxie.manifold import ClassicalMDS


def check[dtype: DType]() raises:
    var x = zeros[dtype](Shape(5, 2))
    for i in range(5):
        x.unsafe_set(i * 2, SIMD[dtype, 1](i))
        x.unsafe_set(i * 2 + 1, SIMD[dtype, 1](i * i))
    var model = ClassicalMDS[dtype]()
    assert_false(model.__sklearn_is_fitted__())
    var y = model.fit_transform(x)
    assert_equal(y.shape[0], 5)
    assert_equal(y.shape[1], 2)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.n_features_in_, 2)
    var again = model.fit_transform(x)
    for i in range(y.size):
        assert_true(abs(y.unsafe_get(i) - again.unsafe_get(i)) < 1e-10)
    y.unsafe_set(0, 999)
    assert_true(model.embedding_.value().unsafe_get(0) != 999)
    var c = clone(model)
    assert_false(c.__sklearn_is_fitted__())
    _ = c.set_params(n_components=1).fit(x.view_with_layout(Shape(5, 2), NDArrayStrides(-2, 1), 8))
    assert_equal(c.embedding_.value().shape[1], 1)
    _ = c.set_params(n_components=0)
    with assert_raises():
        _ = c.fit(x)
    _ = c.set_params(n_components=2, metric="precomputed")
    with assert_raises():
        _ = c.fit(x)
    _ = c.set_params(metric="unknown")
    with assert_raises():
        _ = c.fit(x)
    var params = model.get_params()
    params.metric = "minkowski"
    var options = Dict[String, Float64]()
    options["p"] = 1
    params.metric_params = options.copy()
    _ = model.set_params(params).fit(x)
    var copy = clone(model)
    options["p"] = 0
    assert_equal(copy.metric_params.value()["p"], Float64(1))
    params.metric_params = options.copy()
    _ = model.set_params(params)
    with assert_raises():
        _ = model.fit(x)
    model = ClassicalMDS[dtype](2)
    var invalid = x.copy()
    invalid.unsafe_set(0, Float64("nan").cast[dtype]())
    with assert_raises():
        _ = model.fit(invalid)
    var zero = zeros[dtype](Shape(3, 2))
    _ = model.fit(zero)
    assert_equal(model.embedding_.value().unsafe_get(0), Float64(0))


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: ClassicalMDS lifecycle and dense contracts")
