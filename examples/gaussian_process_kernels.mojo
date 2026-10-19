"""Build, inspect and evaluate composite Gaussian-process kernels."""
from numojo import Shape, zeros
from moxie.gaussian_process.kernels import ConstantKernel, Kernel, Matern, RBF, WhiteKernel


def main() raises:
    var X = zeros[DType.float64](Shape(4, 2))
    for i in range(4):
        X.unsafe_set(2 * i, Float64(i) * 0.5)
        X.unsafe_set(2 * i + 1, Float64(i % 2))
    # sklearn: 1.0 * RBF(length_scale=[1.0, 2.0]) + WhiteKernel(noise_level=0.1)
    var length_scale: List[Float64] = [1.0, 2.0]
    var kernel: Kernel = ConstantKernel(1.0) * RBF(length_scale) + WhiteKernel(0.1)
    print("kernel:", kernel)
    print("theta (log hyperparameters):", kernel.theta())
    for spec in kernel.hyperparameters():
        print("  hyperparameter", spec.name, "elements", spec.n_elements, "fixed", spec.fixed)
    var result = kernel.evaluate(X, eval_gradient=True)
    print("K(X, X):", result.covariance)
    print("gradient shape:", result.gradient.value().shape[0], result.gradient.value().shape[1], result.gradient.value().shape[2])
    print("diag:", kernel.diag(X))
    # General-order Matern uses the native Bessel K_nu.
    var matern: Kernel = Matern(0.8, nu=0.75)
    print("Matern nu=0.75:", matern(X))
    kernel.set_param("k1__k2__length_scale", 3.0)
    print("after set_param:", kernel)
