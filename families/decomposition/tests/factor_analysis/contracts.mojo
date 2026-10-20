"""FactorAnalysis lifecycle, ownership, invalid settings and dtype contracts."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.math import abs
from numojo import NDArray, Shape, zeros, NDArrayStrides
from moxie.base import clone
from moxie.decomposition import FactorAnalysis


def matrix[dtype: DType](n: Int, p: Int) raises -> NDArray[dtype]:
    var x = zeros[dtype](Shape(n, p))
    for i in range(n):
        for j in range(p): x.unsafe_set(i * p + j, Scalar[dtype](Float64((i * 7 + j * 3) % 11) - 4 + Float64(i * j) / 7))
    return x^


def rejects[dtype: DType](var model: FactorAnalysis[dtype], x: NDArray[dtype]) raises:
    with assert_raises(): model.fit(x)


def check[dtype: DType]() raises:
    var x = matrix[dtype](12, 4)
    var model = FactorAnalysis[dtype](2)
    with assert_raises(): _ = model.transform(x)
    with assert_raises(): _ = model.get_covariance()
    var z = model.fit_transform(x)
    assert_equal(z.shape[0], 12)
    assert_equal(z.shape[1], 2)
    assert_equal(model.components_.value().shape[1], 4)
    assert_true(model.n_iter_ >= 1 and model.n_iter_ == len(model.loglike_))
    assert_equal(model.get_feature_names_out()[0], "factoranalysis0")
    # Default random_state=0 makes repeated fits identical.
    var first = model.noise_variance_.value().unsafe_get(0)
    model.fit(x)
    assert_equal(model.noise_variance_.value().unsafe_get(0), first)
    var original = x.unsafe_get(0)
    x.unsafe_set(0, 500)
    assert_equal(model.noise_variance_.value().unsafe_get(0), first)
    x.unsafe_set(0, original)
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    copy.set_params(svd_method="lapack", rotation=Optional[String]("varimax"))
    copy.fit(x.view_with_layout(Shape(12, 4), NDArrayStrides(-4, 1), 44))
    assert_true(copy.score(x) < 0 or copy.score(x) >= 0)
    var precision = copy.get_precision()
    var covariance = copy.get_covariance()
    var product = covariance @ precision
    for i in range(4):
        for j in range(4):
            var expected: Float64 = 1.0 if i == j else 0.0
            assert_true(abs(Float64(product.unsafe_get(i * 4 + j)) - expected) < 1e-8)
    var none = FactorAnalysis[dtype](0)
    none.fit(x)
    assert_equal(none.transform(x).shape[1], 0)
    var full = FactorAnalysis[dtype]()
    full.fit(x)
    assert_equal(full.components_.value().shape[0], 4)
    # Invalid settings raise and leave the previous fit intact.
    rejects(FactorAnalysis[dtype](-1), x)
    rejects(FactorAnalysis[dtype](2, svd_method="arpack"), x)
    rejects(FactorAnalysis[dtype](2, rotation=Optional[String]("oblimin")), x)
    rejects(FactorAnalysis[dtype](2, max_iter=0), x)
    rejects(FactorAnalysis[dtype](2, tol=-1.0), x)
    rejects(FactorAnalysis[dtype](2, iterated_power=-1), x)
    var bad_init: List[Float64] = [1.0, 1.0]
    rejects(FactorAnalysis[dtype](2, noise_variance_init=bad_init^), x)
    rejects(FactorAnalysis[dtype](2, copy=False), x)
    with assert_raises(): _ = model.transform(zeros[dtype](Shape(3, 5)))
    with assert_raises(): model.fit(zeros[dtype](Shape(0, 4)))
    assert_equal(model.noise_variance_.value().unsafe_get(0), first)


def inplace() raises:
    var x = matrix[DType.float64](10, 3)
    var model = FactorAnalysis(1, copy=False)
    model.fit_inplace(x)
    for j in range(3):
        var total: Float64 = 0
        for i in range(10): total += x.unsafe_get(i * 3 + j)
        assert_true(abs(total) < 1e-10)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    inplace()
    print("PASS: FactorAnalysis contracts in Float32 and Float64")
