"""Native GaussianProcessRegressor contracts (shapes, errors, lifecycle)."""
from numojo import NDArray, Shape, zeros
from std.math import abs, isfinite, sin, sqrt
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from moxie.base import clone
from moxie.gaussian_process import (
    FminLBFGSB,
    GPROptimizer,
    GaussianProcessRegressor,
    GenericGaussianProcessRegressor,
)
from moxie.gaussian_process._gpr import LogMarginalLikelihood
from moxie.gaussian_process._plumbing import Dense64, OptimizationResult
from moxie.gaussian_process.kernels import Bounds, ConstantKernel, DotProduct, Kernel, RBF, RationalQuadratic, WhiteKernel
from moxie._random import RandomState, check_random_state, seed_global_random_state


def data(n: Int, d: Int, seed: Int = 3) raises -> Dense64:
    var out = zeros[DType.float64](Shape(n, d))
    var state = seed
    for i in range(n * d):
        state = (state * 1103515245 + 12345) % 2147483648
        out.unsafe_set(i, Float64(state) / 2147483648.0 * 4.0 - 2.0)
    return out^


def target(X: Dense64) raises -> Dense64:
    var y = zeros[DType.float64](Shape(X.shape[0]))
    for i in range(X.shape[0]):
        y.unsafe_set(i, X.item(i, 0) * X.item(i, 0) - X.item(i, 1))
    return y^


def two_targets(X: Dense64) raises -> Dense64:
    var y = zeros[DType.float64](Shape(X.shape[0], 2))
    for i in range(X.shape[0]):
        y.unsafe_set(2 * i, X.item(i, 0))
        y.unsafe_set(2 * i + 1, X.item(i, 1) * 2.0 + 1.0)
    return y^


def test_prior_prediction_before_fit() raises:
    var gpr = GaussianProcessRegressor()
    assert_false(gpr.__sklearn_is_fitted__())
    var X = data(4, 2)
    var result = gpr.predict(X, return_std=True)
    assert_equal(result.mean.shape[0], 4)
    for i in range(4):
        assert_equal(result.mean.item(i), 0.0)
        assert_equal(result.std.value().item(i), 1.0)
    var prior = GenericGaussianProcessRegressor[DType.float64](RBF(1.0), n_targets=3)
    var cov = prior.predict(X, return_std=False, return_cov=True).covariance.value().copy()
    assert_equal(cov.ndim, 3)
    assert_equal(cov.shape[2], 3)
    with assert_raises(contains="At most one of return_std or return_cov"):
        _ = gpr.predict(X, return_std=True, return_cov=True)
    with assert_raises(contains="NotFittedError"):
        _ = gpr.log_marginal_likelihood()


def test_fit_shapes_single_and_multi_target() raises:
    var X = data(12, 2)
    var gpr = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-2))
    _ = gpr.fit(X, target(X))
    assert_true(gpr.__sklearn_is_fitted__())
    assert_equal(gpr.alpha_.ndim, 1)
    assert_equal(gpr.L_.shape[0], 12)
    assert_equal(gpr.n_features_in_, 2)
    var Xt = data(5, 2, 9)
    assert_equal(gpr.predict(Xt).ndim, 1)
    assert_equal(gpr.sample_y(Xt, 4).shape[1], 4)
    var multi = GaussianProcessRegressor(RBF(1.0), n_targets=2)
    _ = multi.fit(X, two_targets(X))
    var r = multi.predict(Xt, return_std=True)
    assert_equal(r.mean.shape[1], 2)
    assert_equal(r.std.value().shape[1], 2)
    assert_equal(multi.predict(Xt, return_std=False, return_cov=True).covariance.value().shape[2], 2)
    var draws = multi.sample_y(Xt, 3)
    assert_equal(draws.ndim, 3)
    assert_equal(draws.shape[1], 2)
    assert_equal(draws.shape[2], 3)
    # sample_y with an integer seed is deterministic; a borrowed stream advances.
    var a = multi.sample_y(Xt, 2, 5)
    var b = multi.sample_y(Xt, 2, 5)
    for i in range(a.size):
        assert_equal(a.unsafe_get(i), b.unsafe_get(i))
    var stream = RandomState(5)
    var c = multi.sample_y(Xt, 2, stream)
    var d = multi.sample_y(Xt, 2, stream)
    assert_equal(c.unsafe_get(0), a.unsafe_get(0))
    assert_true(c.unsafe_get(0) != d.unsafe_get(0))


def test_column_target_squeezes() raises:
    var X = data(8, 2)
    var y = zeros[DType.float64](Shape(8, 1))
    var flat = target(X)
    for i in range(8):
        y.unsafe_set(i, flat.unsafe_get(i))
    var gpr = GaussianProcessRegressor(RBF(1.0))
    _ = gpr.fit(X, y)
    assert_equal(gpr.predict(X).ndim, 1)
    assert_equal(gpr.predict(X, return_std=False, return_cov=True).covariance.value().ndim, 2)


def test_invalid_inputs_and_parameters() raises:
    var X = data(6, 2)
    var y = target(X)
    var gpr = GaussianProcessRegressor(RBF(1.0))
    with assert_raises(contains="inconsistent numbers of samples"):
        _ = gpr.fit(X, target(data(5, 2)))
    var bad = X.copy()
    bad.unsafe_set(3, Float64("nan"))
    with assert_raises(contains="Input X contains NaN"):
        _ = gpr.fit(bad, y)
    with assert_raises(contains="The number of targets seen in `y`"):
        var t = GaussianProcessRegressor(RBF(1.0), n_targets=2)
        _ = t.fit(X, y)
    with assert_raises(contains="alpha must be a scalar or an array"):
        var alpha: List[Float64] = [1e-3, 1e-3]
        var g = GaussianProcessRegressor(RBF(1.0), alpha=alpha^)
        _ = g.fit(X, y)
    with assert_raises(contains="n_restarts_optimizer"):
        _ = GaussianProcessRegressor(RBF(1.0), n_restarts_optimizer=-1)
    with assert_raises(contains="requires that all bounds are finite"):
        var inf_bounds = GaussianProcessRegressor(RBF(1.0, Bounds(1e-3, Float64.MAX * 2.0)), n_restarts_optimizer=1, random_state=0)
        _ = inf_bounds.fit(X, y)
    with assert_raises(contains="is not returning a positive definite matrix"):
        var dup = zeros[DType.float64](Shape(3, 2))
        var fixed = GaussianProcessRegressor(RBF(1.0, "fixed"), alpha=0.0)
        _ = fixed.fit(dup, target(dup))
    _ = gpr.fit(X, y)
    with assert_raises(contains="features, but GaussianProcessRegressor is expecting 2"):
        _ = gpr.predict(data(3, 3))
    with assert_raises(contains="Gradient can only be evaluated for theta!=None"):
        _ = gpr.log_marginal_likelihood(eval_gradient=True)


def test_optimizer_none_and_lml() raises:
    var X = data(10, 2)
    var y = target(X)
    var fixed = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(0.5), optimizer=None)
    _ = fixed.fit(X, y)
    assert_almost_equal(fixed.kernel_.theta().item(1), fixed.kernel.value().theta().item(1), atol=0.0)
    var lml = fixed.log_marginal_likelihood(fixed.kernel_.theta(), eval_gradient=True)
    assert_almost_equal(lml.value, fixed.log_marginal_likelihood_value_, atol=1e-10)
    # Central differences of the LML against the analytic gradient.
    var theta = fixed.kernel_.theta()
    for c in range(theta.size):
        var up = theta.copy()
        var down = theta.copy()
        up.unsafe_set(c, theta.unsafe_get(c) + 1e-6)
        down.unsafe_set(c, theta.unsafe_get(c) - 1e-6)
        var numeric = (fixed.log_marginal_likelihood(up^).value - fixed.log_marginal_likelihood(down^).value) / 2e-6
        assert_almost_equal(lml.gradient.value().item(c), numeric, atol=1e-5 * (1.0 + abs(numeric)))
    var optimized = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(0.5))
    _ = optimized.fit(X, y)
    assert_true(optimized.log_marginal_likelihood_value_ >= fixed.log_marginal_likelihood_value_)


def test_normalize_constant_target() raises:
    var X = data(6, 2)
    var y = zeros[DType.float64](Shape(6))
    for i in range(6):
        y.unsafe_set(i, 4.0)
    var gpr = GaussianProcessRegressor(RBF(1.0), normalize_y=True)
    _ = gpr.fit(X, y)
    var pred = gpr.predict(X)
    for i in range(6):
        assert_almost_equal(pred.item(i), 4.0, atol=1e-8)


def test_clone_repeat_fit_and_params() raises:
    var X = data(8, 2)
    var gpr = GaussianProcessRegressor(RBF(1.0), random_state=0)
    _ = gpr.fit(X, target(X))
    var fresh = clone(gpr)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_true(fresh.kernel.value() == gpr.kernel.value())
    var first = gpr.alpha_.copy()
    var X2 = data(5, 2, 11)
    _ = gpr.fit(X2, target(X2))
    assert_equal(gpr.alpha_.shape[0], 5)
    assert_equal(first.shape[0], 8)
    var params = gpr.get_params()
    params.n_restarts_optimizer = -3
    with assert_raises(contains="n_restarts_optimizer"):
        _ = gpr.set_params(params)
    assert_equal(gpr.n_restarts_optimizer, 0)  # failed set_params leaves config intact


def test_named_parameter_updates() raises:
    var X = data(7, 2)
    var y = target(X)
    var model = GaussianProcessRegressor(RBF(1.5), alpha=0.01, optimizer=None,
                                         normalize_y=True, random_state=17)
    var before = model.fit(X, y).predict(X)
    var fitted_kernel = model.kernel_.copy()
    _ = model.set_params(alpha=0.2, normalize_y=False, copy_X_train=False,
                         n_restarts_optimizer=0)
    assert_equal(model.alpha[Float64], 0.2)
    assert_false(model.normalize_y)
    assert_false(model.copy_X_train)
    assert_equal(model.random_state.value(), 17)
    assert_false(Bool(model.optimizer))
    assert_true(model.kernel.value() == fitted_kernel)
    assert_true(model.kernel_ == fitted_kernel)
    var after = model.set_params(kernel=None, random_state=None).set_params().predict(X)
    for i in range(7):
        assert_equal(after.item(i), before.item(i))
    with assert_raises():
        _ = model.set_params(alpha=0.7, n_restarts_optimizer=-1)
    assert_equal(model.alpha[Float64], 0.2)
    assert_equal(model.n_restarts_optimizer, 0)
    assert_true(model.__sklearn_is_fitted__())

    _ = model.set_params(kernel=Optional[Kernel](RBF(2.0)),
                         optimizer=Optional[FminLBFGSB](FminLBFGSB()),
                         n_targets=Optional[Int](1), random_state=Optional[Int](0))
    assert_true(model.kernel.value() == Kernel(RBF(2.0)))
    assert_true(Bool(model.optimizer))
    assert_equal(model.n_targets.value(), 1)
    assert_equal(model.random_state.value(), 0)
    _ = model.set_params(kernel=Optional[Kernel](None),
                         optimizer=Optional[FminLBFGSB](None),
                         n_targets=Optional[Int](None), random_state=Optional[Int](None))
    assert_false(Bool(model.kernel))
    assert_false(Bool(model.optimizer))
    assert_false(Bool(model.n_targets))
    assert_false(Bool(model.random_state))
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.alpha[Float64], 0.2)
    var noise: List[Float64] = [0.1]
    _ = model.set_params(alpha=noise.copy()).fit(X, y)
    assert_equal(model.alpha[List[Float64]][0], 0.1)
    assert_true(model.__sklearn_is_fitted__())
    var reference = GaussianProcessRegressor(alpha=0.1, optimizer=None)
    var expected = reference.fit(X, y).predict(X)
    var actual = model.predict(X)
    for i in range(7):
        assert_almost_equal(actual.item(i), expected.item(i), atol=1e-12)


def test_float32_input() raises:
    var X = data(9, 2)
    var X32 = zeros[DType.float32](Shape(9, 2))
    var y32 = zeros[DType.float32](Shape(9))
    var y = target(X)
    for i in range(18):
        X32.unsafe_set(i, X.unsafe_get(i).cast[DType.float32]())
    for i in range(9):
        y32.unsafe_set(i, y.unsafe_get(i).cast[DType.float32]())
    var gpr = GenericGaussianProcessRegressor[DType.float32](RBF(1.0))
    _ = gpr.set_params(normalize_y=True).fit(X32, y32)
    assert_true(gpr.score(X32, y32) > 0.99)


struct KeepInitial(GPROptimizer):
    """Native custom optimizer: returns the start after one evaluation."""

    def __init__(out self):
        pass

    def minimize(
        self, mut objective: LogMarginalLikelihood, initial_theta: Dense64, bounds: Dense64
    ) raises -> OptimizationResult:
        var value = objective.evaluate(initial_theta, True).value
        return OptimizationResult(theta=initial_theta.copy(), value=value, converged=True, message="kept")


def test_custom_native_optimizer() raises:
    var X = data(8, 2)
    var gpr = GenericGaussianProcessRegressor[DType.float64, KeepInitial](RBF(0.3))
    _ = gpr.set_params(optimizer=Optional[KeepInitial](None))
    assert_false(Bool(gpr.optimizer))
    _ = gpr.set_params(optimizer=Optional[KeepInitial](KeepInitial())).fit(X, target(X))
    assert_almost_equal(gpr.kernel_.theta().item(0), gpr.kernel.value().theta().item(0), atol=0.0)
    var reference = GaussianProcessRegressor(RBF(0.3), optimizer=None)
    _ = reference.fit(X, target(X))
    assert_almost_equal(gpr.log_marginal_likelihood_value_, reference.log_marginal_likelihood_value_, atol=1e-10)


def test_score_is_r2() raises:
    var X = data(10, 2)
    var y = target(X)
    var gpr = GaussianProcessRegressor(RBF(1.0) + WhiteKernel(1e-3))
    _ = gpr.fit(X, y)
    assert_true(gpr.score(X, y) > 0.999)
    assert_true(gpr.score(X, y) <= 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_default_fit_optimizes_kernel() raises:
    var X = data(5, 2)
    var y = target(X)
    var model = GaussianProcessRegressor()
    _ = model.fit(X, y)
    assert_equal(model.kernel_.n_dims(), 2)
    var explicit = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(1.0))
    _ = explicit.fit(X, y)
    assert_equal(model.log_marginal_likelihood().value, explicit.log_marginal_likelihood().value)
    var actual = model.predict(X)
    var expected = explicit.predict(X)
    for i in range(actual.size):
        assert_equal(actual.unsafe_get(i), expected.unsafe_get(i))


def test_none_random_state_uses_shared_stream() raises:
    var X = data(5, 2)
    var y = target(X)
    seed_global_random_state(91)
    var borrowed = RandomState(91)
    var model = GaussianProcessRegressor(RBF(1.0), n_restarts_optimizer=2)
    var expected = GaussianProcessRegressor(RBF(1.0), n_restarts_optimizer=2)
    _ = model.fit(X, y)
    _ = expected.fit(X, y, borrowed)
    assert_equal(model.kernel_.theta().item(0), expected.kernel_.theta().item(0))
    with check_random_state() as shared:
        assert_equal(shared[].next_uint32(), borrowed.next_uint32())
    var actual_draws = model.sample_y(X, 2, None)
    var expected_draws = model.sample_y(X, 2, borrowed)
    for i in range(actual_draws.size):
        assert_equal(actual_draws.unsafe_get(i), expected_draws.unsafe_get(i))
    with check_random_state() as shared:
        assert_equal(shared[].next_uint32(), borrowed.next_uint32())


def test_likelihood_clone_kernel_controls_mutation() raises:
    var X = data(6, 2)
    var model = GaussianProcessRegressor(RBF(1.0), optimizer=None)
    _ = model.fit(X, target(X))
    var original = model.kernel_.theta()
    var theta = original.copy()
    theta.unsafe_set(0, 0.4)
    var copied = model.log_marginal_likelihood(theta.copy(), eval_gradient=True)
    assert_equal(model.kernel_.theta().item(0), original.item(0))
    var changed = model.log_marginal_likelihood(theta.copy(), eval_gradient=True, clone_kernel=False)
    assert_almost_equal(model.kernel_.theta().item(0), 0.4)
    assert_equal(copied.value, changed.value)
    assert_equal(copied.gradient.value().item(0), changed.gradient.value().item(0))
    assert_equal(model.log_marginal_likelihood().value, model.log_marginal_likelihood_value_)


def test_sample_y_warns_on_indefinite_posterior() raises:
    # plot_gpr_prior_posterior's fitted RationalQuadratic posterior on 100 grid
    # points has eigenvalues down to -1.8e-8 (sklearn computes the same), so
    # NumPy's default check_valid="warn" warns and still draws; sample_y must too.
    var train = [4.835149195068383, 2.7361612458786118, 4.863421799824422, 3.5740799683718234,
                 3.4886441229863543, 1.080447477901882, 4.881372273881209, 0.031151276022949315,
                 1.2649118119172198, 2.1739576620222287]
    var X_train = zeros[DType.float64](Shape(10, 1))
    var y_train = zeros[DType.float64](Shape(10))
    for i in range(10):
        X_train.unsafe_set(i, train[i])
        var shifted = train[i] - 2.5
        y_train.unsafe_set(i, sin(shifted * shifted))
    var X = zeros[DType.float64](Shape(100, 1))
    for i in range(100):
        X.unsafe_set(i, 5.0 * Float64(i) / 99.0)
    var kernel = ConstantKernel(0.3527724212081871) * RationalQuadratic(
        length_scale=0.2785622248175069, alpha=5387636.441757492
    )
    var gpr = GaussianProcessRegressor(kernel, optimizer=None)
    _ = gpr.fit(X_train, y_train)
    var prediction = gpr.predict(X, return_std=False, return_cov=True)
    var strict = RandomState(0)
    with assert_raises(contains="not_positive_semidefinite"):
        _ = strict.multivariate_normal(prediction.mean, prediction.covariance.value(), 5)
    var draws = gpr.sample_y(X, 5)
    assert_equal(draws.shape[0], 100)
    assert_equal(draws.shape[1], 5)
    var again = gpr.sample_y(X, 5)
    for i in range(draws.size):
        assert_true(isfinite(draws.unsafe_get(i)))
        assert_equal(draws.unsafe_get(i), again.unsafe_get(i))
