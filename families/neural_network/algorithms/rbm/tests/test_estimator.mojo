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
        with assert_raises():
            model.partial_fit(X)


def main() raises:
    check[DType.float64]()
    check[DType.float32]()
    print("PASS implemented estimator contracts; Float32 partial_fit remains unfinished")
