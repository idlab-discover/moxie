"""Native kernel and GaussianProcessRegressor state round trips."""

from numojo import NDArray, NDArrayStrides, Shape, zeros
from std.math import inf, nan
from std.testing import TestSuite, assert_equal, assert_false, assert_raises, assert_true

from moxie.base import clone
from moxie.gaussian_process import FminLBFGSB
from moxie.gaussian_process.kernels import (
    Bounds,
    CompoundKernel,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    Matern,
    PairwiseKernel,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)
from moxie.gaussian_process.serialization import GaussianProcessRegressorSchema, KernelSchema
from moxie.serialization import (
    BinaryWriter,
    ModelArchive,
    dumps,
    load_model,
    loads,
    save_model,
)


def assert_same[dtype: DType](actual: NDArray[dtype], expected: NDArray[dtype]) raises:
    assert_true(actual.shape == expected.shape)
    for i in range(actual.size):
        assert_equal(actual.unsafe_get(i), expected.unsafe_get(i))


def test_kernel_tree_round_trip() raises:
    var kwargs: Dict[String, Float64] = {"coef0": 0.75, "degree": 2.0}
    var lower: List[Float64] = [0.1, 0.2]
    var upper: List[Float64] = [1.5, 2.5]
    var scales: List[Float64] = [0.7, 1.3]
    var leaf = RBF(scales, Bounds(lower, upper))
    var first = (ConstantKernel(2.0, "fixed") * leaf + WhiteKernel(0.25)) ** 2
    var second = (Matern(1.2, nu=inf[DType.float64]()) + DotProduct(0.4)) ** 2.0
    var third = RationalQuadratic(0.7, 1.2) * ExpSineSquared(1.3, 4.0)
    var fourth = Kernel(
        PairwiseKernel(gamma=0.8, metric="poly", pairwise_kernels_kwargs=kwargs^)
    )
    var kernel = CompoundKernel([first.copy(), second.copy(), third.copy(), fourth.copy()])
    var path = "builds/tests/gaussian_process/regression/kernel-state.moxie"
    save_model[KernelSchema](kernel, path)
    var restored = load_model[KernelSchema](path)
    assert_equal(String(restored), String(kernel))
    assert_equal(dumps(KernelSchema.encode(restored)), dumps(KernelSchema.encode(kernel)))
    var parts = restored.kernels()
    assert_true(String(parts[0]).endswith(" ** 2"))
    assert_true(String(parts[1]).endswith(" ** 2.0"))
    var params = parts[3].get_params()
    var mapping = params["pairwise_kernels_kwargs"].as_mapping("kwargs")
    assert_equal(mapping.value()["degree"], 2.0)

    var X = zeros[DType.float64](Shape(2, 2))
    X.unsafe_set(2, 1.0)
    X.unsafe_set(3, -1.0)
    assert_same(restored.evaluate(X).covariance, kernel.evaluate(X).covariance)
    parts[0].set_param("kernel__k1__k2__length_scale", 2.0)
    assert_true(String(parts[0]) != String(first))
    assert_equal(String(restored), String(kernel))


def test_unfitted_prior_and_optimizer() raises:
    comptime Schema = GaussianProcessRegressorSchema[]
    var optimizer = FminLBFGSB()
    optimizer.engine.history_size = 7
    optimizer.engine.max_line_search_steps = 15
    var model = Schema.Model(
        RBF(1.4) + WhiteKernel(0.2), alpha=0.025, optimizer=optimizer,
        n_restarts_optimizer=2, normalize_y=True, copy_X_train=False,
        n_targets=2, random_state=42,
    )
    var path = "builds/tests/gaussian_process/regression/gpr-unfitted.moxie"
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_false(restored.__sklearn_is_fitted__())
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
    assert_equal(restored.optimizer.value().engine.history_size, 7)
    assert_equal(restored.optimizer.value().engine.max_line_search_steps, 15)
    var X = zeros[DType.float64](Shape(3, 1))
    var prior = restored.predict(X, return_std=True)
    var expected = model.predict(X, return_std=True)
    assert_same(prior.mean, expected.mean)
    assert_same(prior.std.value(), expected.std.value())
    with assert_raises(contains="NotFittedError"):
        _ = restored.log_marginal_likelihood()

    var defaults = Schema.decode(loads(dumps(Schema.encode(Schema.Model()))))
    assert_false(Bool(defaults.kernel))
    assert_false(Bool(defaults.n_targets))
    assert_false(Bool(defaults.random_state))


def fitted_round_trip[dtype: DType, vector_target: Bool]() raises:
    comptime Schema = GaussianProcessRegressorSchema[dtype]
    var backing = zeros[dtype](Shape(6, 2))
    for i in range(backing.size):
        backing.unsafe_set(i, Scalar[dtype](i) / 4)
    var X = backing.view_with_layout(Shape(3, 1), NDArrayStrides(-4, 1), 9)
    var y_shape: List[Int] = [3]
    comptime if not vector_target:
        y_shape.append(2)
    var y = zeros[dtype](Shape(y_shape))
    for i in range(y.size):
        y.unsafe_set(i, Scalar[dtype](i % 3 - 1))
    var model = Schema.Model(
        RBF(0.75, "fixed") + WhiteKernel(0.01, "fixed"),
        alpha=0.005, optimizer=None, normalize_y=True,
    )
    _ = model.fit(X, y)
    var Xt = zeros[dtype](Shape(2, 1))
    Xt.unsafe_set(1, 0.5)
    var original = model.predict(Xt, return_std=False, return_cov=True)
    var theta = model.kernel_.theta()
    var original_lml = model.log_marginal_likelihood(theta.copy(), eval_gradient=True)
    var draws = model.sample_y(Xt, 3, random_state=2)

    # Store pending configuration independently of the learned factor and kernel.
    var noise: List[Float64] = [0.005, 0.005, 0.005]
    _ = model.set_params(
        kernel=Optional[Kernel](DotProduct(3.0)), n_targets=Optional[Int](9),
        alpha=noise^, normalize_y=False,
    )
    model.convergence_warnings_.append("ConvergenceWarning: test state")
    var path = "builds/tests/gaussian_process/regression/gpr-" + String(dtype) + ".moxie"
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_true(restored.__sklearn_is_fitted__())
    assert_false(clone(restored).__sklearn_is_fitted__())
    assert_false(Bool(restored.optimizer))
    assert_false(restored.normalize_y)
    assert_equal(restored.n_targets.value(), 9)
    assert_equal(String(restored.kernel.value()), "DotProduct(sigma_0=3)")
    assert_equal(String(restored.kernel_), String(model.kernel_))
    assert_equal(restored.convergence_warnings_, model.convergence_warnings_)
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))

    var prediction = restored.predict(Xt, return_std=False, return_cov=True)
    assert_same(prediction.mean, original.mean)
    assert_same(prediction.covariance.value(), original.covariance.value())
    assert_equal(restored.log_marginal_likelihood(theta^).value, original_lml.value)
    assert_same(restored.sample_y(Xt, 3, random_state=2), draws)
    for i in range(X.shape[0]):
        assert_equal(restored.X_train_.item(i, 0), X.item(i, 0))
    assert_equal(restored.y_train_.ndim, y.ndim)
    assert_equal(restored.alpha_.ndim, y.ndim)
    restored.X_train_.unsafe_set(0, -30)
    assert_equal(model.X_train_.item(0, 0), X.item(0, 0))
    _ = restored.set_params(n_targets=Optional[Int](None))
    _ = restored.fit(X, y)
    assert_equal(String(restored.kernel_), "DotProduct(sigma_0=3)")


def test_fitted_round_trips() raises:
    fitted_round_trip[DType.float64, True]()
    fitted_round_trip[DType.float32, False]()


def test_malformed_kernel_archive() raises:
    var archive = ModelArchive(KernelSchema.TypeId, KernelSchema.StateVersion)
    var writer = BinaryWriter()
    writer.uint(1)     # one node
    writer.integer(10) # exponentiation
    writer.uint(1)     # one child
    writer.integer(0)  # self-reference
    writer.scalar[DType.float64](2)
    writer.boolean(False)
    archive.manifest = writer^.finish()
    with assert_raises(contains="invalid kernel child index"):
        _ = KernelSchema.decode(loads(dumps(archive)))
    var valid = KernelSchema.encode(RBF(1.0))
    valid.manifest.append(0)
    with assert_raises(contains="trailing"):
        _ = KernelSchema.decode(valid)


def test_malformed_regressor_archive() raises:
    comptime Schema = GaussianProcessRegressorSchema[]
    var X = zeros[DType.float64](Shape(3, 1))
    var y = zeros[DType.float64](Shape(3))
    var model = Schema.Model(optimizer=None)
    _ = model.fit(X, y)
    var archive = Schema.encode(model)
    with assert_raises(contains="dtype mismatch"):
        _ = GaussianProcessRegressorSchema[DType.float32].decode(archive)
    var bad_version = archive.copy()
    bad_version.schema_version += 1
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = Schema.decode(bad_version)
    var bad_shape = archive.copy()
    bad_shape.payloads[0].shape = [1, 3]
    with assert_raises(contains="invalid shape"):
        _ = Schema.decode(loads(dumps(bad_shape)))
    var unknown_array = archive.copy()
    unknown_array.payloads[0].name = "unknown"
    with assert_raises(contains="missing array X_train_"):
        _ = Schema.decode(unknown_array)

    var bad_factor = archive.copy()
    var factor: List[Float64] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    var replacement = ModelArchive(Schema.TypeId, Schema.StateVersion)
    var factor_shape: List[Int] = [3, 3]
    replacement.add_array[DType.float64]("L_", factor_shape, factor)
    bad_factor.payloads[2] = replacement.payloads[0].copy()
    with assert_raises(contains="nonpositive GP Cholesky"):
        _ = Schema.decode(loads(dumps(bad_factor)))
    model.L_.unsafe_set(0, nan[DType.float64]())
    with assert_raises(contains="nonfinite GP array L_"):
        _ = Schema.encode(model)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
