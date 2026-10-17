"""Native contracts for PolynomialCountSketch: lifecycle, params, hashes, dtype, domain, names."""
from numojo import NDArray, Shape, NDArrayStrides, zeros
from moxie.kernel_approximation import PolynomialCountSketch
from moxie.base import clone
from moxie._random import RandomState, entropy_state
from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal


def same(a: NDArray[DType.float64], b: NDArray[DType.float64]) raises:
    assert_equal(a.size, b.size)
    for k in range(a.size):
        assert_equal(a.unsafe_get(k), b.unsafe_get(k))


def contracts[dtype: DType]() raises:
    var x = zeros[dtype](Shape(3, 2))
    for k in range(6):
        x.unsafe_set(k, Scalar[dtype](Float64(k) / 5.0))
    var model = PolynomialCountSketch[dtype](random_state=42, n_components=16, coef0=1.0)
    var defaults = PolynomialCountSketch[dtype]()
    assert_equal(defaults.get_params().degree, 2)
    assert_equal(defaults.get_params().n_components, 100)
    assert_equal(defaults.get_params().gamma, 1.0)
    assert_equal(defaults.get_params().coef0, 0.0)
    with assert_raises(contains="random_state=None"):
        defaults.fit(x)
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(x)
    with assert_raises(contains="NotFittedError"):
        _ = model.get_feature_names_out()
    var z = model.fit_transform(x, zeros[dtype](Shape(3)))
    assert_equal(z.shape[0], 3)
    assert_equal(z.shape[1], 16)
    assert_equal(model.indexHash_.shape[0], 2)
    assert_equal(model.indexHash_.shape[1], 3)  # coef0 != 0 adds a column
    for k in range(model.bitHash_.size):
        var bit = model.bitHash_.unsafe_get(k)
        assert_true(bit == 1 or bit == -1)
        var slot = model.indexHash_.unsafe_get(k)
        assert_true(slot >= 0 and slot < 16)
    assert_equal(model.n_features_in_, 2)
    # Degree 1, gamma 1, coef0 0: the map is exactly the signed count sketch.
    var linear = PolynomialCountSketch[dtype](degree=1, n_components=8, random_state=3)
    var zl = linear.fit_transform(x)
    for i in range(3):
        for k in range(8):
            var bucket = 0.0
            for j in range(2):
                if Int(linear.indexHash_.unsafe_get(j)) == k:
                    bucket += Float64(linear.bitHash_.unsafe_get(j)) * Float64(x.unsafe_get(i * 2 + j))
            assert_almost_equal(zl.unsafe_get(i * 8 + k), bucket, atol=1e-6)
    # Determinism, repeat fit, independence of outputs and learned state.
    same(z, model.transform(x))
    assert_true(model.fit(x).__sklearn_is_fitted__())
    same(z, model.transform(x))
    z.unsafe_set(0, 123)
    assert_true(model.transform(x).unsafe_get(0) != 123)
    var c = clone(model)
    assert_false(c.__sklearn_is_fitted__())
    c.fit(x)
    for k in range(c.indexHash_.size):
        assert_equal(c.indexHash_.unsafe_get(k), model.indexHash_.unsafe_get(k))
    var original = model.indexHash_.unsafe_get(0)
    c.indexHash_.unsafe_set(0, original + 1)
    assert_equal(model.indexHash_.unsafe_get(0), original)
    # Explicit contexts consume the same stream as an integer seed.
    var rng = RandomState(42)
    var other = PolynomialCountSketch[dtype](n_components=16, coef0=1.0)
    other.fit_with_rng(x, rng)
    same(other.transform(x), model.transform(x))
    var context = entropy_state()
    _ = other.fit_transform_with_rng(x, context)
    # Names follow the fitted width; parameters are copies.
    var names = model.get_feature_names_out()
    assert_equal(len(names), 16)
    assert_equal(names[15], "polynomialcountsketch15")
    var given: List[String] = ["a", "b"]
    assert_equal(len(model.get_feature_names_out(given.copy())), 16)
    var short: List[String] = ["a"]
    with assert_raises():
        _ = model.get_feature_names_out(short.copy())
    var params = model.get_params()
    params.degree = 3
    assert_equal(model.get_params().degree, 2)
    # Changing coef0 from nonzero to zero changes the column count: refit needed.
    model.set_params(coef0=0.0)
    var reset = model.get_params()
    reset.coef0 = 0.0
    model.set_params(reset)
    with assert_raises(contains="does not match"):
        _ = model.transform(x)
    model.fit(x)
    assert_equal(model.indexHash_.shape[1], 2)
    # Shrinking n_components after fit cannot index the old hashes.
    model.set_params(n_components=1)
    var hashes_need_width = False
    for k in range(model.indexHash_.size):
        if model.indexHash_.unsafe_get(k) >= 1:
            hashes_need_width = True
    if hashes_need_width:
        with assert_raises(contains="n_components"):
            _ = model.transform(x)
    model.set_params(n_components=16)
    # Invalid parameters and inputs.
    var bad_gamma = PolynomialCountSketch[dtype](gamma=-1.0, random_state=0)
    with assert_raises(contains="gamma"):
        bad_gamma.fit(x)
    var bad_degree = PolynomialCountSketch[dtype](degree=0, random_state=0)
    with assert_raises(contains="degree"):
        bad_degree.fit(x)
    var bad_width = PolynomialCountSketch[dtype](n_components=0, random_state=0)
    with assert_raises(contains="n_components"):
        bad_width.fit(x)
    var negative = PolynomialCountSketch[dtype](coef0=-1.0, random_state=0)
    negative.fit(x)  # sklearn accepts negative coef0 in fit
    with assert_raises(contains="coef0"):
        _ = negative.transform(x)
    var nonfinite = x.copy()
    nonfinite.unsafe_set(0, Scalar[dtype](1) / Scalar[dtype](0))
    with assert_raises(contains="finite"):
        model.fit(nonfinite)
    with assert_raises(contains="finite"):
        _ = model.transform(nonfinite)
    with assert_raises():
        model.fit(zeros[dtype](Shape(0, 2)))
    with assert_raises():
        _ = model.transform(zeros[dtype](Shape(3, 3)))
    # Strided views follow logical order; gamma 0 maps to the constant sketch.
    var backing = zeros[dtype](Shape(3, 4))
    for k in range(12):
        backing.unsafe_set(k, Scalar[dtype](k) / 10)
    var strided = backing.view_with_layout(Shape(2, 2), NDArrayStrides(4, -1), 2)
    var dense = zeros[dtype](Shape(2, 2))
    for i in range(2):
        for j in range(2):
            dense.unsafe_set(i * 2 + j, strided.unsafe_get(i * 4 - j))
    same(model.transform(strided), model.transform(dense))
    var one = PolynomialCountSketch[dtype](n_components=1, degree=3, coef0=2.0, gamma=0.0, random_state=5)
    var constant = one.fit_transform(dense)
    # One bucket: (sum of signed sqrt(coef0))^3 with only the constant column nonzero.
    var s = Float64(one.bitHash_.unsafe_get(2)) * Float64(one.bitHash_.unsafe_get(5)) * Float64(one.bitHash_.unsafe_get(8))
    assert_almost_equal(constant.unsafe_get(0), s * 2.0 ** 1.5, atol=1e-12)


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    print("PASS native lifecycle, params, hashes, dtype, domain, strided, clone and RNG contracts")
