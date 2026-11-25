"""Exercise one custom vector kernel through both estimator entry points."""
from numojo import NDArray, Shape, zeros
from std.math import nan
from std.testing import assert_almost_equal, assert_equal, assert_false, assert_raises

from moxie.base import clone
from moxie.gaussian_process import (
    DenseGaussianProcessRegressor,
    FminLBFGSB,
    KernelContract,
    KernelEvaluation,
)
from moxie.gaussian_process.kernels import Bounds, DotProduct, Hyperparameter
from moxie.gaussian_process._plumbing import Dense64
from moxie.kernel_ridge import KernelRidge, TypedGaussianProcessKernel


@fieldwise_init
struct AffineWithNoise(KernelContract):
    comptime Input = Dense64
    comptime Params = Float64

    var offset: Float64

    @staticmethod
    def empty_input() raises -> Dense64:
        return zeros[DType.float64](Shape(0, 0))

    def get_params(self, deep: Bool = True) raises -> Float64:
        return self.offset

    def set_params(mut self, params: Float64) raises:
        self.offset = params

    def _affine(self) -> DotProduct:
        return DotProduct(self.offset, Bounds(0.1, 0.8))

    def hyperparameters(self) raises -> List[Hyperparameter]:
        return self._affine().hyperparameters()

    def theta(self) raises -> Dense64:
        return self._affine().theta()

    def bounds(self) raises -> Dense64:
        return self._affine().bounds()

    def set_theta(mut self, theta: Dense64) raises:
        var affine = self._affine().as_kernel()
        affine.set_theta(theta)
        self.offset = affine.get_params()["sigma_0"].as_float("sigma_0")

    def clone_with_theta(self, theta: Dense64) raises -> Self:
        var kernel = self.copy()
        kernel.set_theta(theta)
        return kernel^

    def evaluate(
        self, X: Dense64, Y: Optional[Dense64] = None, eval_gradient: Bool = False,
    ) raises -> KernelEvaluation:
        var result = self._affine().evaluate(X, Y, eval_gradient)
        if not Y:
            var n = X.shape[0]
            for i in range(n):
                var index = i * n + i
                result.covariance.unsafe_set(index, result.covariance.unsafe_get(index) + 0.125)
        return result^

    def diag(self, X: Dense64) raises -> Dense64:
        var diagonal = self._affine().diag(X)
        for i in range(diagonal.size):
            diagonal.unsafe_set(i, diagonal.unsafe_get(i) + 0.125)
        return diagonal^

    def is_stationary(self) -> Bool:
        return False

    def requires_vector_input(self) -> Bool:
        return True


def array[dtype: DType](shape: Shape, values: List[Float64]) raises -> NDArray[dtype]:
    var result = zeros[dtype](shape)
    for i in range(len(values)):
        result.unsafe_set(i, values[i].cast[dtype]())
    return result^


def emit(case_name: String, key: String, values: Dense64) raises:
    var shape = List[String]()
    var data = List[String]()
    for axis in range(values.ndim):
        shape.append(String(values.shape[axis]))
    for i in range(values.size):
        data.append(String(values.unsafe_get(i)))
    print(case_name, key, ",".join(shape), " ".join(data), sep="|")


def run_case[dtype: DType](name: String) raises:
    comptime GP = DenseGaussianProcessRegressor[AffineWithNoise, dtype]
    comptime Adapter = TypedGaussianProcessKernel[AffineWithNoise]
    var kernel = AffineWithNoise(0.5)
    var X = array[dtype](Shape(4, 2), [0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 4.0, 2.0])
    var y = array[dtype](Shape(4), [1.0, -2.0, 3.0, 1.0])
    var Z = array[dtype](Shape(3, 2), [1.0, 2.0, 3.0, 4.0, 3.0, 0.0])
    var gpr = GP(kernel, alpha=0.5)
    emit(name, "prior_std", gpr.predict(Z, return_std=True).std.value())
    _ = gpr.fit(X, y)
    var posterior = gpr.predict(Z, return_std=True)
    emit(name, "mean", posterior.mean)
    emit(name, "std", posterior.std.value())
    emit(name, "covariance", gpr.predict(Z, return_std=False, return_cov=True).covariance.value())
    emit(name, "theta", gpr.kernel_.theta())
    emit(name, "L", gpr.L_)
    emit(name, "alpha", gpr.alpha_)
    var likelihood = gpr.log_marginal_likelihood(gpr.kernel_.theta(), eval_gradient=True)
    emit(name, "lml", array[DType.float64](Shape(1), [likelihood.value]))
    emit(name, "gradient", likelihood.gradient.value())
    assert_equal(gpr.n_features_in_, 2)
    assert_equal(gpr.X_train_.shape, Shape(4, 2))
    assert_false(clone(gpr).__sklearn_is_fitted__())
    assert_equal(clone(gpr).kernel.value().offset, 0.5)
    with assert_raises(contains="explicit custom kernel"):
        _ = GP()
    with assert_raises(contains="Expected 2D array"):
        _ = gpr.predict(y)
    with assert_raises(contains="expecting 2 features"):
        _ = gpr.predict(zeros[dtype](Shape(3, 1)))
    with assert_raises(contains="inconsistent numbers of samples"):
        _ = gpr.fit(Z, y)
    assert_almost_equal(gpr.predict(Z).item(0), posterior.mean.item(0))

    var ridge = KernelRidge[DType.float64, Adapter](alpha=0.5, kernel=Adapter(kernel^))
    _ = ridge.fit(X, y)
    emit(name, "ridge_dual", ridge.dual_coef_[Dense64])
    emit(name, "ridge_predict", ridge.predict(Z)[Dense64])
    var ridge_clone = ridge.__sklearn_clone__()
    assert_false(ridge_clone.__sklearn_is_fitted__())
    assert_equal(ridge_clone.get_params_record().native_kernel.value().kernel.offset, 0.5)
    with assert_raises():
        _ = ridge_clone.predict(Z)
    X.unsafe_set(0, SIMD[dtype, 1](nan[DType.float64]().cast[dtype]()))
    with assert_raises(contains="NaN"):
        _ = gpr.fit(X, y)
    assert_equal(gpr.X_train_.unsafe_get(0), SIMD[dtype, 1](0.0))
    assert_almost_equal(gpr.predict(Z).item(0), posterior.mean.item(0))
    _ = gpr.set_params(
        kernel=Optional[AffineWithNoise](AffineWithNoise(0.3)),
        optimizer=Optional[FminLBFGSB](None),
    )
    _ = gpr.fit(Z, array[dtype](Shape(3), [1.0, 2.0, 3.0]))
    assert_equal(gpr.X_train_.shape, Shape(3, 2))
    assert_equal(gpr.kernel_.offset, 0.3)

def main() raises:
    run_case[DType.float32]("float32")
    run_case[DType.float64]("float64")
    print("PASS custom dense GP kernel contracts")
