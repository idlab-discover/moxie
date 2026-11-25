from numojo import zeros, Shape, NDArrayStrides
from std.testing import assert_equal, assert_raises
from moxie.base import clone
from moxie._random import RandomState
from moxie.neural_network import BernoulliRBM


def check[dtype: DType]() raises:
    var model = BernoulliRBM[dtype](2, batch_size=2, n_iter=3, random_state=42)
    var X = zeros[dtype](Shape(5, 3))
    for i in range(X.size):
        X.unsafe_set(i, SIMD[dtype, 1](i % 2))
    with assert_raises():
        _ = model.transform(X)
    var transformed = model.fit_transform(X)
    assert_equal(model.n_features_in_, 3)
    assert_equal(transformed.shape[1], 2)
    var names = model.get_feature_names_out()
    assert_equal(names[1], "bernoullirbm1")
    var first = model.components_.copy()
    _ = model.fit(X)
    for i in range(first.size):
        assert_equal(first.unsafe_get(i), model.components_.unsafe_get(i))
    var scores = model.score_samples(X)
    var second = model.score_samples(X)
    for i in range(scores.size):
        assert_equal(scores.unsafe_get(i), second.unsafe_get(i))
    var g = model.gibbs(X)
    assert_equal(g.shape[1], 3)
    var copy = clone(model)
    assert_equal(copy.__sklearn_is_fitted__(), False)
    var params = copy.get_params()
    params.n_components = 3
    _ = copy.set_params(params)
    _ = copy.fit(X)
    assert_equal(copy.components_.shape[0], 3)
    assert_equal(model.components_.shape[0], 2)
    var rng = RandomState(42)
    var context = BernoulliRBM[dtype](2, batch_size=2, n_iter=3)
    _ = context.fit(X, rng)
    _ = context.gibbs(X, rng)
    _ = context.score_samples(X, rng)
    var strided = X.view_with_layout(Shape(5, 3), NDArrayStrides(3, -1), 2)
    _ = context.transform(strided)
    with assert_raises():
        _ = context.transform(zeros[dtype](Shape(1, 2)))
    comptime if dtype == DType.float64:
        var online = BernoulliRBM[dtype](2, batch_size=2, random_state=42)
        online.partial_fit(X)
        online.partial_fit(X)
        assert_equal(online.n_features_in_, 3)
    else:
        var online = BernoulliRBM[dtype](2, batch_size=2, random_state=42)
        with assert_raises(contains="BernoulliRBM[float64]"):
            online.partial_fit(X)
        assert_equal(online.__sklearn_is_fitted__(), False)
        var rng_position = rng.state_position()
        with assert_raises(contains="BernoulliRBM[float64]"):
            online.partial_fit(X, rng)
        assert_equal(rng.state_position(), rng_position)
        model.partial_fit(X)
        assert_equal(model.components_.shape, Shape(2, 3))
        assert_equal(model.h_samples_.shape, Shape(2, 2))


def errors[dtype: DType]() raises:
    """Reference parameter constraints, validate_data and check_is_fitted failures."""
    var X = zeros[dtype](Shape(4, 3))
    for i in range(X.size):
        X.unsafe_set(i, SIMD[dtype, 1](i % 2))
    with assert_raises(contains="ValueError"):
        var bad_model = BernoulliRBM[dtype](0, random_state=0)
        _ = bad_model.fit(X)
    with assert_raises(contains="ValueError"):
        var bad_model = BernoulliRBM[dtype](2, learning_rate=0.0, random_state=0)
        _ = bad_model.fit(X)
    with assert_raises(contains="ValueError"):
        var bad_model = BernoulliRBM[dtype](2, batch_size=0, random_state=0)
        _ = bad_model.fit(X)
    with assert_raises(contains="ValueError"):
        var bad_model = BernoulliRBM[dtype](2, n_iter=-1, random_state=0)
        _ = bad_model.fit(X)
    # random_state=None needs the explicit context instead of a hidden global stream.
    with assert_raises(contains="random_state=None"):
        var bad_model = BernoulliRBM[dtype](2)
        _ = bad_model.fit(X)
    var unfitted = BernoulliRBM[dtype](2, random_state=0)
    with assert_raises(contains="NotFittedError"):
        _ = unfitted.score_samples(X)
    with assert_raises(contains="NotFittedError"):
        _ = unfitted.gibbs(X)
    with assert_raises(contains="NotFittedError"):
        _ = unfitted.get_feature_names_out()
    with assert_raises(contains="NotFittedError"):
        _ = unfitted.get_n_features_in()
    var bad = X.copy()
    bad.unsafe_set(4, SIMD[dtype, 1](0) / SIMD[dtype, 1](0))
    with assert_raises(contains="finite"):
        var bad_model = BernoulliRBM[dtype](2, random_state=0)
        _ = bad_model.fit(bad)
    with assert_raises(contains="expected samples"):
        var bad_model = BernoulliRBM[dtype](2, random_state=0)
        _ = bad_model.fit(zeros[dtype](Shape(0, 3)))
    var model = BernoulliRBM[dtype](2, n_iter=1, random_state=0)
    _ = model.fit(X)
    assert_equal(model.get_n_features_in(), 3)
    with assert_raises(contains="finite"):
        _ = model.transform(bad)
    with assert_raises(contains="feature"):
        _ = model.score_samples(zeros[dtype](Shape(2, 4)))
    with assert_raises(contains="feature"):
        _ = model.gibbs(zeros[dtype](Shape(2, 4)))
    var two: List[String] = ["a", "b"]
    var three: List[String] = ["a", "b", "c"]
    with assert_raises(contains="input feature names"):
        _ = model.get_feature_names_out(two^)
    var names = model.get_feature_names_out(three^)
    assert_equal(len(names), 2)
    assert_equal(names[0], "bernoullirbm0")
    with assert_raises(contains="feature"):
        model.partial_fit(zeros[dtype](Shape(2, 4)))
    with assert_raises(contains="finite"):
        model.partial_fit(bad)
    # A failed fit validation leaves the fitted state usable, as in sklearn.
    var params = model.get_params()
    params.n_components = 0
    with assert_raises(contains="ValueError"):
        _ = model.set_params(params).fit(X)
    assert_equal(model.components_.shape[0], 2)


def check_promoted_input() raises:
    var X = zeros[DType.float32](Shape(5, 3))
    var model = BernoulliRBM[DType.float64](2, batch_size=2, random_state=42)
    model.partial_fit(X)
    assert_equal(model.components_.shape, Shape(2, 3))
    assert_equal(model.h_samples_.shape, Shape(2, 2))
    assert_equal(model.get_n_features_in(), 3)
    with assert_raises(contains="feature"):
        model.partial_fit(zeros[DType.float32](Shape(1, 2)))


def check_named_params[dtype: DType]() raises:
    var X = zeros[dtype](Shape(3, 2))
    var model = BernoulliRBM[dtype](2, n_iter=1, verbose=1, random_state=42)
    _ = model.fit(X)
    var before = model.transform(X)
    _ = model.set_params(learning_rate=0.06)
    var params = model.get_params()
    assert_equal(params.learning_rate, 0.06)
    assert_equal(params.n_components, 2)
    assert_equal(params.batch_size, 10)
    assert_equal(params.n_iter, 1)
    assert_equal(params.verbose, 1)
    assert_equal(params.random_state.value(), 42)
    _ = model.set_params(n_components=3, batch_size=2, n_iter=0, verbose=0)
    var after = model.set_params().transform(X)
    assert_equal(after.shape, before.shape)
    for i in range(before.size):
        assert_equal(after.unsafe_get(i), before.unsafe_get(i))

    _ = model.set_params(random_state=Optional[Int](0))
    var unfitted = clone(model)
    assert_equal(unfitted.__sklearn_is_fitted__(), False)
    params = unfitted.get_params()
    assert_equal(params.random_state.value(), 0)
    assert_equal(params.n_iter, 0)
    assert_equal(params.verbose, 0)
    _ = model.set_params(random_state=Optional[Optional[Int]](Optional[Int]()))
    assert_equal(Bool(model.get_params().random_state), False)
    with assert_raises(contains="random_state=None"):
        _ = model.fit(X)
    var rng = RandomState(0)
    _ = model.fit(X, rng)
    _ = unfitted.fit(X)
    assert_equal(model.h_samples_.shape, Shape(2, 3))
    assert_equal(model.components_.shape, Shape(3, 2))
    for i in range(model.components_.size):
        assert_equal(model.components_.unsafe_get(i), unfitted.components_.unsafe_get(i))


def main() raises:
    errors[DType.float64]()
    errors[DType.float32]()
    check[DType.float64]()
    check[DType.float32]()
    check_promoted_input()
    check_named_params[DType.float64]()
    check_named_params[DType.float32]()
    print("PASS RBM estimator contracts including named parameter updates and Float32 incremental input")
