"""Native contracts for GraphicalLasso, graphical_lasso and the Gram CD kernel."""
import numojo as nm
from numojo import NDArray, Shape
from std.math import abs
from std.testing import assert_equal, assert_true, assert_false
from moxie.base import clone
from moxie._random import RandomState
from moxie.covariance import GraphicalLasso, graphical_lasso, empirical_covariance
from moxie.covariance._enet_gram import enet_coordinate_descent_gram
from moxie.covariance._graph_lasso import alpha_max


def has(message: String, error: Error) -> Bool:
    return String(error).find(message) >= 0


def data[dtype: DType](n: Int, p: Int) raises -> NDArray[dtype]:
    var rng = RandomState(1)
    var X = nm.zeros[dtype](Shape(n, p))
    for i in range(n):
        var shared = rng.standard_normal()
        for j in range(p):
            X.unsafe_set(i * p + j, (rng.standard_normal() + (shared if j < 2 else 0.0)).cast[dtype]())
    return X^


def check[dtype: DType]() raises:
    var X = data[dtype](50, 4)
    var model = GraphicalLasso[dtype](alpha=0.05)
    assert_false(model.__sklearn_is_fitted__())
    try:
        _ = model.costs_()
        assert_true(False)
    except e:
        assert_true(has("NotFittedError: This GraphicalLasso instance", e))
    model.fit(X)
    var P = model.get_precision()
    var C = model.covariance_()
    # Symmetric estimates and P ~ inverse(C) at convergence.
    for a in range(4):
        for b in range(4):
            assert_true(abs(Float64(P.unsafe_get(a * 4 + b)) - Float64(P.unsafe_get(b * 4 + a))) < 1e-5)
            var product: Float64 = 0
            for k in range(4):
                product += Float64(C.unsafe_get(a * 4 + k)) * Float64(P.unsafe_get(k * 4 + b))
            assert_true(abs(product - (1.0 if a == b else 0.0)) < 1e-3)
    assert_true(model.n_iter_() >= 1 and len(model.costs_()) == model.n_iter_())
    # alpha >= alpha_max gives a diagonal precision.
    var S = empirical_covariance(X)
    var S64 = nm.zeros[DType.float64](Shape(4, 4))
    for k in range(16):
        S64.unsafe_set(k, Float64(S.unsafe_get(k)))
    var strong = GraphicalLasso[dtype](alpha=alpha_max(S64) * 1.01)
    strong.fit(X)
    var diagonal = strong.get_precision()
    for a in range(4):
        for b in range(4):
            if a != b:
                assert_equal(Float64(diagonal.unsafe_get(a * 4 + b)), 0.0)
    # graphical_lasso on the empirical covariance equals the estimator.
    var result = graphical_lasso(S, 0.05)
    assert_true(abs(result.precision.unsafe_get(1) - Float64(P.unsafe_get(1))) < 1e-4)
    var precomputed = GraphicalLasso[dtype](alpha=0.05, covariance=String("precomputed"))
    precomputed.fit(S)
    assert_equal(Float64(precomputed.location_().unsafe_get(0)), 0.0)
    # Parameter validation and unsupported branches.
    var lars = GraphicalLasso[dtype](alpha=0.05, mode=String("lars"))
    lars.fit(X)
    assert_true(lars.error_norm(C) < 1e-7)
    var lars_copy = clone(lars)
    assert_equal(lars_copy.mode, "lars")
    assert_false(lars_copy.__sklearn_is_fitted__())
    lars.set_params(alpha=0.1, enet_tol=0.3)
    lars.fit(X)
    var fresh_lars = GraphicalLasso[dtype](alpha=0.1, mode="lars")
    fresh_lars.fit(X)
    assert_equal(lars.error_norm(fresh_lars.covariance_()), 0)
    var lars_zero = graphical_lasso(S, 0, mode="lars")
    assert_equal(lars_zero.n_iter, 0)
    var bad_mode = GraphicalLasso[dtype](mode=String("svd"))
    try:
        bad_mode.fit(X)
        assert_true(False)
    except e:
        assert_true(has("InvalidParameterError: The 'mode' parameter", e))
    var negative = GraphicalLasso[dtype](alpha=-1.0)
    try:
        negative.fit(X)
        assert_true(False)
    except e:
        assert_true(has("'alpha'", e))
    try:
        var narrow = GraphicalLasso[dtype]()
        narrow.fit(data[dtype](10, 1))
        assert_true(False)
    except e:
        assert_true(has("minimum of 2 is required", e))
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    assert_equal(copy.alpha, 0.05)
    # alpha = 0 inverts a regular covariance exactly.
    var zero = graphical_lasso(S, 0.0)
    assert_equal(zero.n_iter, 0)
    print("PASS graphical lasso contracts", dtype)


def check_enet() raises:
    # 1-D lasso with Q = [[2]], q = [3], alpha = 1 -> w = (3 - 1) / 2 = 1.
    var Q = nm.ones[DType.float64](Shape(1, 1))
    Q.unsafe_set(0, 2.0)
    var q = nm.ones[DType.float64](Shape(1))
    q.unsafe_set(0, 3.0)
    var result = enet_coordinate_descent_gram(nm.zeros[DType.float64](Shape(1)), 1.0, 0.0, Q, q, q, 100, 1e-10)
    assert_true(abs(result.w.unsafe_get(0) - 1.0) < 1e-12)
    # alpha above |q| gives zero.
    var zero = enet_coordinate_descent_gram(nm.zeros[DType.float64](Shape(1)), 5.0, 0.0, Q, q, q, 100, 1e-10)
    assert_equal(zero.w.unsafe_get(0), 0.0)
    print("PASS Gram coordinate descent")


def main() raises:
    check_enet()
    check[DType.float64]()
    check[DType.float32]()
