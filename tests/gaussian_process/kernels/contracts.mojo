"""Native kernel contracts: metadata, parameters, cloning, errors, gradients."""
from numojo import NDArray, NDArrayStrides, Shape, zeros
from std.collections import Dict
from std.math import abs, exp, inf, log
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from moxie.gaussian_process.kernels import (
    Bounds,
    CompoundKernel,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Exponentiation,
    Kernel,
    KernelParams,
    Matern,
    PairwiseKernel,
    Product,
    RBF,
    RationalQuadratic,
    Sum,
    WhiteKernel,
)
from moxie.gaussian_process._plumbing import Dense64, KernelContract


def data(n: Int, d: Int, seed: Int = 1) raises -> Dense64:
    var out = zeros[DType.float64](Shape(n, d))
    var state = seed
    for i in range(n * d):
        state = (state * 1103515245 + 12345) % 2147483648
        out.unsafe_set(i, Float64(state) / 2147483648.0 * 2.0 - 1.0)
    return out^


def two_bounds() raises -> Bounds:
    var low: List[Float64] = [1e-3, 1e-3]
    var high: List[Float64] = [1.0, 1.0]
    return Bounds(low, high)


def aniso() -> List[Float64]:
    return [0.5, 2.0, 1.2]


def assert_contract[K: KernelContract](k: K) raises:
    """Compile-time check that Kernel satisfies the family contract."""
    _ = k.theta()


def test_default_hyperparameters_and_theta() raises:
    var k: Kernel = ConstantKernel(2.0) * RBF(aniso()) + WhiteKernel(0.1, "fixed")
    assert_contract(k)
    var specs = k.hyperparameters()
    assert_equal(len(specs), 3)
    assert_equal(specs[0].name, "k1__k1__constant_value")
    assert_equal(specs[1].name, "k1__k2__length_scale")
    assert_equal(specs[1].n_elements, 3)
    assert_equal(specs[1].bounds.shape[0], 3)
    assert_equal(specs[2].name, "k2__noise_level")
    assert_true(specs[2].fixed)
    assert_equal(specs[2].bounds.shape[0], 0)
    assert_equal(k.n_dims(), 4)
    var theta = k.theta()
    assert_almost_equal(theta.item(0), log(2.0), atol=1e-15)
    assert_almost_equal(theta.item(3), log(1.2), atol=1e-15)
    var bounds = k.bounds()
    assert_equal(bounds.shape[0], 4)
    assert_almost_equal(bounds.item(0, 0), log(1e-5), atol=1e-12)


def test_rational_quadratic_theta_order_is_sorted() raises:
    var k: Kernel = RationalQuadratic(length_scale=3.0, alpha=0.5)
    var specs = k.hyperparameters()
    assert_equal(specs[0].name, "alpha")
    assert_equal(specs[1].name, "length_scale")
    assert_almost_equal(k.theta().item(0), log(0.5), atol=1e-15)
    var fixed_alpha: Kernel = RationalQuadratic(3.0, 0.5, alpha_bounds="fixed")
    assert_equal(fixed_alpha.n_dims(), 1)
    assert_almost_equal(fixed_alpha.theta().item(0), log(3.0), atol=1e-15)


def test_set_theta_and_clone_are_independent() raises:
    var k: Kernel = ConstantKernel(2.0) * RBF(1.0)
    var theta = zeros[DType.float64](Shape(2))
    theta.unsafe_set(0, log(3.0))
    theta.unsafe_set(1, log(0.25))
    var clone = k.clone_with_theta(theta)
    assert_almost_equal(k.theta().item(0), log(2.0), atol=1e-15)
    assert_almost_equal(clone.get_params()["k2__length_scale"].as_float("x"), 0.25, atol=1e-15)
    var copied = k.copy()
    copied.set_param("k1__constant_value", 7.0)
    assert_equal(k.get_params()["k1__constant_value"].as_float("x"), 2.0)
    with assert_raises(contains="theta has not the correct number of entries"):
        k.set_theta(zeros[DType.float64](Shape(3)))


def test_get_and_set_params() raises:
    var k: Kernel = Exponentiation(RBF(1.5) + DotProduct(0.3), 2.0)
    var keys = k.get_params().keys()
    assert_equal(keys[0], "kernel")
    assert_equal(keys[1], "exponent")
    assert_equal(keys[2], "kernel__k1")
    assert_equal(keys[4], "kernel__k1__length_scale")
    k.set_param("kernel__k2__sigma_0", 0.9)
    k.set_param("exponent", 3.0)
    assert_equal(k.exponent(), 3.0)
    assert_equal(k.kernel().k2().get_params()["sigma_0"].as_float("x"), 0.9)
    k.set_param("kernel__k1", Kernel(Matern(0.4, nu=2.5)))
    assert_equal(k.kernel().k1().name(), "Matern")
    k.set_param("kernel__k1__length_scale_bounds", Bounds("fixed"))
    assert_equal(k.n_dims(), 1)
    var params = KernelParams()
    params.set("kernel__k1__nu", 0.5)
    params.set("exponent", 1.0)
    k.set_params(params)
    assert_equal(k.kernel().k1().get_params()["nu"].as_float("x"), 0.5)
    with assert_raises(contains="Invalid parameter"):
        k.set_param("kernel__k3__x", 1.0)
    with assert_raises(contains="Invalid parameter"):
        k.set_param("length_scale", 1.0)
    with assert_raises(contains="expects a scalar"):
        k.set_param("exponent", "two")


def test_equality_and_repr() raises:
    var a: Kernel = ConstantKernel(4.0) * RBF(1.0)
    var b: Kernel = RBF(1.0) * ConstantKernel(4.0)
    assert_true(a == b)  # sklearn operator equality accepts swapped operands
    assert_false(a == Kernel(RBF(1.0)) * ConstantKernel(4.0, "fixed"))
    assert_equal(String(a), "2**2 * RBF(length_scale=1)")
    assert_equal(String(WhiteKernel(1e-6)), "WhiteKernel(noise_level=1e-06)")
    assert_equal(String(Matern(aniso(), nu=2.5)), "Matern(length_scale=[0.5, 2, 1.2], nu=2.5)")
    assert_equal(String(RationalQuadratic(0.1234, 12345.0)), "RationalQuadratic(alpha=1.23e+04, length_scale=0.123)")
    assert_equal(String(PairwiseKernel(gamma=0.5, metric="rbf")), "PairwiseKernel(gamma=0.5, metric=rbf)")
    assert_equal(String(2.0 + RBF(3.0)), "1.41**2 + RBF(length_scale=3)")


def test_predicates() raises:
    assert_true(Kernel(RBF(1.0)).is_stationary())
    assert_false(Kernel(DotProduct()).is_stationary())
    assert_true(Kernel(PairwiseKernel(metric="rbf")).is_stationary())
    assert_false(Kernel(PairwiseKernel(metric="linear")).is_stationary())
    assert_false((RBF(1.0) + DotProduct()).is_stationary())
    assert_false(Kernel(ConstantKernel()).requires_vector_input())
    assert_false((ConstantKernel() + WhiteKernel()).requires_vector_input())
    assert_true((ConstantKernel() * RBF()).requires_vector_input())


def test_precomputed_pairwise() raises:
    var X = data(3, 3)
    var kernel = Kernel(PairwiseKernel(metric="precomputed"))
    var result = kernel.evaluate(X, eval_gradient=True)
    for i in range(9):
        assert_equal(result.covariance.unsafe_get(i), X.unsafe_get(i))
        assert_equal(result.gradient.value().unsafe_get(i), 0.0)
    result.covariance.unsafe_set(0, 42.0)
    assert_true(X.unsafe_get(0) != 42.0)
    var rectangular = data(2, 3)
    var indexed = data(3, 5)
    var cross = kernel.evaluate(rectangular, indexed.copy(), eval_gradient=True)
    assert_equal(cross.covariance.shape[0], 2)
    assert_equal(cross.covariance.shape[1], 3)
    assert_equal(cross.gradient.value().shape[1], 3)
    for i in range(6):
        assert_equal(cross.covariance.unsafe_get(i), rectangular.unsafe_get(i))
        assert_equal(cross.gradient.value().unsafe_get(i), 0.0)
    var fixed = Kernel(PairwiseKernel(metric="precomputed", gamma_bounds="fixed"))
    var empty = fixed.evaluate(rectangular, indexed.copy(), eval_gradient=True)
    assert_equal(empty.gradient.value().shape[0], 2)
    assert_equal(empty.gradient.value().shape[1], 2)
    assert_equal(empty.gradient.value().shape[2], 0)
    with assert_raises(contains="Precomputed metric requires shape"):
        _ = kernel.evaluate(rectangular)
    with assert_raises(contains="Precomputed metric requires shape"):
        _ = kernel.diag(X)
    var column = data(3, 1)
    var diagonal = kernel.diag(column)
    for i in range(3):
        assert_equal(diagonal.item(i), column.item(i, 0))
    var product = kernel * kernel
    var composed = product.evaluate(rectangular, indexed.copy(), eval_gradient=True)
    assert_equal(composed.gradient.value().shape[1], 3)
    assert_equal(composed.gradient.value().shape[2], 2)
    for i in range(6):
        assert_equal(composed.covariance.unsafe_get(i),
                     rectangular.unsafe_get(i) * rectangular.unsafe_get(i))

def test_invalid_inputs_raise() raises:
    var X = data(4, 2)
    with assert_raises(contains="Anisotropic kernel must have the same number of dimensions"):
        _ = Kernel(RBF(aniso())).evaluate(X)
    with assert_raises(contains="kernel gradients require Y=None"):
        _ = Kernel(RBF(1.0)).evaluate(X, X.copy(), eval_gradient=True)
    with assert_raises(contains="features but Y has"):
        _ = Kernel(RBF(1.0)).evaluate(X, data(3, 5))
    with assert_raises(contains="PairwiseKernel metric must be one of"):
        _ = Kernel(PairwiseKernel(metric="unknown")).evaluate(X)
    with assert_raises(contains="X contains negative values"):
        _ = Kernel(PairwiseKernel(metric="chi2")).evaluate(X)
    with assert_raises(contains="should have either 1 or 3 dimensions"):
        _ = Kernel(RBF(aniso(), two_bounds())).bounds()
    with assert_raises(contains="must be a (lower, upper) pair"):
        _ = Kernel(RBF(1.0, "fix")).bounds()
    with assert_raises(contains="expected a 2-D dense array"):
        _ = Kernel(RBF(1.0)).evaluate(zeros[DType.float64](Shape(2, 2, 2)))
    with assert_raises(contains="CompoundKernel cannot be nested"):
        _ = (CompoundKernel([Kernel(RBF())]) + RBF()).evaluate(X)
    with assert_raises(contains="needs at least one kernel"):
        _ = CompoundKernel(List[Kernel]())
    with assert_raises(contains="only supports isotropic"):
        var rq: Kernel = RationalQuadratic()
        rq.set_param("length_scale", aniso())
        _ = rq.evaluate(data(3, 3))


def test_shapes_and_white_cross_covariance() raises:
    var X = data(5, 3)
    var Y = data(4, 3, 9)
    var k: Kernel = RBF(1.0) + WhiteKernel(0.5)
    var e = k.evaluate(X, eval_gradient=True)
    assert_equal(e.covariance.shape[0], 5)
    assert_equal(e.gradient.value().shape[2], 2)
    var cross = Kernel(WhiteKernel(0.5))(X, X.copy())
    for i in range(cross.size):
        assert_equal(cross.unsafe_get(i), 0.0)  # zero even for identical rows
    assert_equal(k.evaluate(X, Y^).covariance.shape[1], 4)
    var fixed = Kernel(RBF(1.0, "fixed")).evaluate(X, eval_gradient=True)
    assert_equal(fixed.gradient.value().shape[2], 0)
    var compound = CompoundKernel([Kernel(RBF(1.0)), Kernel(DotProduct(1.0))])
    var ce = compound.evaluate(X, eval_gradient=True)
    assert_equal(ce.covariance.ndim, 3)
    assert_equal(ce.covariance.shape[2], 2)
    assert_equal(ce.gradient.value().ndim, 4)
    assert_equal(compound.diag(X).shape[1], 2)
    with assert_raises(contains="equal numbers of free hyperparameters"):
        _ = CompoundKernel([Kernel(RBF(1.0)), Kernel(RBF(1.0, "fixed"))]).evaluate(X, eval_gradient=True)
    var tiny = data(1, 3)
    assert_equal(Kernel(Matern(1.0, nu=0.7)).evaluate(tiny).covariance.item(0, 0), 1.0)


def test_diag_matches_full_matrix() raises:
    var X = data(6, 3)
    var poly = Dict[String, Float64]()
    poly["degree"] = 2.0
    var kernels = List[Kernel]()
    kernels.append(ConstantKernel(2.0) * Matern(0.7, nu=0.8) + WhiteKernel(0.3))
    kernels.append(DotProduct(0.4) ** 2.0)
    kernels.append(PairwiseKernel(gamma=0.3, metric="poly", pairwise_kernels_kwargs=poly^))
    kernels.append(ExpSineSquared(1.1, 2.0) * RationalQuadratic(0.8, 1.3))
    for k in kernels:
        var full = k(X)
        var diag = k.diag(X)
        for i in range(6):
            assert_almost_equal(diag.item(i), full.item(i, i), atol=1e-12)


def check_gradient(k: Kernel, X: Dense64, rtol: Float64) raises:
    """Central differences of k(X) in log theta against the analytic gradient."""
    var e = k.evaluate(X, eval_gradient=True)
    ref G = e.gradient.value()
    var theta = k.theta()
    var n = X.shape[0]
    var h = 1e-6
    for c in range(theta.size):
        var up = theta.copy()
        var down = theta.copy()
        up.unsafe_set(c, theta.unsafe_get(c) + h)
        down.unsafe_set(c, theta.unsafe_get(c) - h)
        var Kp = k.clone_with_theta(up)(X)
        var Km = k.clone_with_theta(down)(X)
        for e2 in range(n * n):
            var numeric = (Kp.unsafe_get(e2) - Km.unsafe_get(e2)) / (2 * h)
            var analytic = G.unsafe_get(e2 * theta.size + c)
            if abs(numeric - analytic) > rtol * (1.0 + abs(numeric)):
                raise Error(
                    String(k) + " gradient " + String(c) + " entry " + String(e2)
                    + ": analytic " + String(analytic) + " numeric " + String(numeric)
                )


def test_analytic_gradients_match_finite_differences() raises:
    var X = data(6, 3)
    var kernels = List[Kernel]()
    kernels.append(ConstantKernel(2.0) * RBF(aniso()) + WhiteKernel(0.3))
    kernels.append(Kernel(RBF(1.3)))
    for nu in [0.5, 1.5, 2.5, inf[DType.float64](), 0.7]:
        kernels.append(Kernel(Matern(0.9, nu=nu)))
        kernels.append(Kernel(Matern(aniso(), nu=nu)))
    kernels.append(Kernel(RationalQuadratic(0.8, 2.5)))
    kernels.append(Kernel(ExpSineSquared(1.1, 2.3)))
    kernels.append(Kernel(DotProduct(0.7)) ** 1.5)
    kernels.append(Kernel(PairwiseKernel(gamma=0.4, metric="rbf")))
    kernels.append(Kernel(PairwiseKernel(gamma=0.4, metric="sigmoid")))
    kernels.append(ExpSineSquared(1.1, 2.3) * RationalQuadratic(0.8, 1.5) + 0.5)
    for k in kernels:
        check_gradient(k, X, 2e-5)


def test_float32_input_is_widened() raises:
    var X64 = data(4, 3)
    var X32 = zeros[DType.float32](Shape(4, 3))
    for i in range(X64.size):
        X32.unsafe_set(i, X64.unsafe_get(i).cast[DType.float32]())
    var k: Kernel = RBF(1.0)
    var K32 = k(X32)
    var K64 = k(X64)
    for i in range(K32.size):
        assert_almost_equal(K32.unsafe_get(i), K64.unsafe_get(i), atol=1e-6)


def test_strided_input_is_read_through_strides() raises:
    var X = data(6, 3)
    var view = X.view_with_layout(Shape(3, 3), NDArrayStrides(6, 1), 0)  # every other row
    var dense = zeros[DType.float64](Shape(3, 3))
    for i in range(3):
        for j in range(3):
            dense.unsafe_set(i * 3 + j, X.item(2 * i, j))
    var precomputed = Kernel(PairwiseKernel(metric="precomputed"))(view)
    for i in range(9):
        assert_equal(precomputed.unsafe_get(i), dense.unsafe_get(i))
    var k: Kernel = RBF(0.8)
    var a = k(view)
    var b = k(dense)
    for i in range(9):
        assert_equal(a.unsafe_get(i), b.unsafe_get(i))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
