"""Fit a Gaussian process to a noisy sine and predict with uncertainty."""
from numojo import Shape, zeros
from std.math import sin
from moxie.gaussian_process import GaussianProcessRegressor
from moxie.gaussian_process.kernels import ConstantKernel, Kernel, RBF, WhiteKernel


def main() raises:
    var n = 12
    var X = zeros[DType.float64](Shape(n, 1))
    var y = zeros[DType.float64](Shape(n))
    for i in range(n):
        var x = Float64(i) * 0.5
        X.unsafe_set(i, x)
        y.unsafe_set(i, sin(x) + 0.05 * Float64((i * 7) % 5 - 2))
    var kernel: Kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.01)
    var gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=2, random_state=0)
    _ = gpr.fit(X, y)
    print("learned kernel:", gpr.kernel_)
    print("log marginal likelihood:", gpr.log_marginal_likelihood_value_)
    var Xt = zeros[DType.float64](Shape(3, 1))
    Xt.unsafe_set(0, 0.25)
    Xt.unsafe_set(1, 2.6)
    Xt.unsafe_set(2, 7.0)
    var result = gpr.predict(Xt, return_std=True)
    print("mean:", result.mean)
    print("std:", result.std.value())
    print("R^2 on training data:", gpr.score(X, y))
    print("posterior draws (3 x 2):", gpr.sample_y(Xt, 2))
