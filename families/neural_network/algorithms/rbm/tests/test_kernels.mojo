from numojo import zeros, Shape
from std.testing import assert_equal, assert_raises
from moxie._random import RandomState
from moxie.neural_network._rbm import BernoulliRBM, _sigmoid, _softplus


def check[dtype: DType]() raises:
    var rng = RandomState(42)
    var state = BernoulliRBM[dtype](2, batch_size=2)
    state.initialize(2, 1, 2, rng)
    var X = zeros[dtype](Shape(2, 1))
    X.unsafe_set(1, 1)
    for _ in range(42):
        state.update(X, 0.1, rng)
    var probs = state.mean_hiddens(X)
    assert_equal(probs.shape[0], 2)
    assert_equal(probs.shape[1], 2)
    var sample = state.gibbs(X, rng)
    assert_equal(sample.unsafe_get(0), False)
    assert_equal(sample.unsafe_get(1), True)
    var scores = state.score_samples(X, rng)
    assert_equal(scores.shape[0], 2)
    var snapshot = state.__sklearn_clone__()
    assert_equal(snapshot.__sklearn_is_fitted__(), False)
    assert_equal(state.components_.unsafe_get(0) != 42, True)
    assert_equal(X.unsafe_get(0), 0)
    assert_equal(X.unsafe_get(1), 1)
    with assert_raises():
        _ = state.mean_hiddens(zeros[dtype](Shape(2, 2)))
    with assert_raises():
        _ = state.mean_hiddens(zeros[dtype](Shape(0, 1)))
    assert_equal(_sigmoid(SIMD[dtype, 1](1000)), 1)
    assert_equal(_sigmoid(SIMD[dtype, 1](-1000)), 0)
    assert_equal(_softplus(SIMD[dtype, 1](1000)), 1000)


def main() raises:
    check[DType.float64]()
    check[DType.float32]()
    print("PASS RBM kernels")
