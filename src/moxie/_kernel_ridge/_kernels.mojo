"""Dense kernel-ridge pairwise kernels over the shared NuMojo storage."""
from ._api import KernelParameters
import numojo as nm
from std.math import sqrt, isfinite, tanh
from std.sys import llvm_intrinsic
from ._validation import validate, epsilon


def validate_kernel(kernel: String) raises:
    if kernel not in [
        String("linear"),
        "poly",
        "polynomial",
        "rbf",
        "laplacian",
        "sigmoid",
        "cosine",
        "chi2",
        "additive_chi2",
        "precomputed",
    ]:
        raise Error("ValueError: unsupported kernel")


def pairwise[
    dtype: DType
](
    X: nm.NDArray[dtype],
    Y: nm.NDArray[dtype],
    kernel: String = "linear",
    gamma: Optional[Float64] = None,
    degree: Float64 = 3,
    coef0: Float64 = 1,
    same_data: Bool = False,
) raises -> nm.NDArray[dtype]:
    comptime assert dtype.is_floating_point()
    validate(X, "X", True)
    validate(Y, "Y", True)
    validate_kernel(kernel)
    if same_data and X.shape[0] != Y.shape[0]:
        raise Error("ValueError: same_data requires matching shapes")
    if kernel == "precomputed":
        if X.shape[1] != Y.shape[0]:
            raise Error(
                "ValueError: precomputed columns must match training rows"
            )
        return X.copy()
    if X.shape[1] != Y.shape[1]:
        raise Error("ValueError: different feature counts")
    var g = 1.0 / Float64(X.shape[1])
    if gamma:
        g = gamma.value()
    if kernel == "chi2" and not gamma:
        raise Error("ValueError: chi2 requires explicit gamma")
    if kernel in [
        String("rbf"),
        "laplacian",
        "poly",
        "polynomial",
        "sigmoid",
        "chi2",
    ]:
        if g < 0 or not isfinite(g):
            raise Error("ValueError: gamma must be finite and nonnegative")
    if kernel == "poly" or kernel == "polynomial":
        if degree < 0 or not isfinite(degree):
            raise Error("ValueError: degree must be finite and nonnegative")
    if kernel in [String("poly"), "polynomial", "sigmoid"] and not isfinite(
        coef0
    ):
        raise Error("ValueError: coef0 must be finite")
    var rows = X.shape[0]
    var other_rows = Y.shape[0]
    var features = X.shape[1]
    var xrs = Int(X.strides[0])
    var xcs = Int(X.strides[1])
    var yrs = Int(Y.strides[0])
    var ycs = Int(Y.strides[1])
    # Measured crossover: at 32x32, approved NuMojo matmul greatly
    # outperforms the inherited scalar dot loop; at 4x4 setup dominates.
    if rows * other_rows >= 256 and kernel in [
        String("linear"),
        "poly",
        "polynomial",
        "sigmoid",
    ]:
        var product = X @ Y.T()
        for i in range(rows * other_rows):
            var value = product.unsafe_get(i)
            if kernel != "linear":
                value *= SIMD[dtype, 1](g)
                value += SIMD[dtype, 1](coef0)
                if kernel == "sigmoid":
                    value = tanh(value)
                else:
                    value = llvm_intrinsic["llvm.pow", SIMD[dtype, 1]](
                        value, SIMD[dtype, 1](degree)
                    )
            if not isfinite(value):
                raise Error("ValueError: nonfinite kernel result")
            product.unsafe_set(i, value)
        return product^
    var result = nm.zeros[dtype](nm.Shape(rows, other_rows))
    for i in range(rows):
        for j in range(other_rows):
            if same_data and j < i:
                result.unsafe_set(
                    i * other_rows + j, result.unsafe_get(j * other_rows + i)
                )
                continue
            var dot = SIMD[dtype, 1](0)
            var nx = SIMD[dtype, 1](0)
            var ny = SIMD[dtype, 1](0)
            var chi = Float64(0)
            var l1 = Float64(0)
            var dx = Float64(0)
            var dy = Float64(0)
            var dp = Float64(0)
            for k in range(features):
                var x = X.unsafe_get(i * xrs + k * xcs)
                var y = Y.unsafe_get(j * yrs + k * ycs)
                dot += x * y
                if kernel == "rbf":
                    dx += Float64(x) * Float64(x)
                    dy += Float64(y) * Float64(y)
                    dp += Float64(x) * Float64(y)
                elif kernel == "cosine":
                    nx += x * x
                    ny += y * y
                elif kernel == "laplacian":
                    l1 += abs(Float64(x) - Float64(y))
                if kernel == "chi2" or kernel == "additive_chi2":
                    if x < 0 or y < 0:
                        raise Error(
                            "ValueError: chi-squared requires nonnegative"
                            " inputs"
                        )
                    if x + y != 0:
                        chi -= Float64(x - y) * Float64(x - y) / Float64(x + y)
            var value = dot
            if kernel == "rbf":
                # sklearn computes Float32 distances in Float64, then casts
                # before the in-place gamma multiplication and exponentiation.
                var distance = SIMD[dtype, 1](
                    max((-2.0 * dp + dx) + dy, Float64(0))
                )
                if same_data and i == j:
                    distance = 0
                value = llvm_intrinsic["llvm.exp", SIMD[dtype, 1]](
                    -SIMD[dtype, 1](g) * distance
                )
            elif kernel == "laplacian":
                value = SIMD[dtype, 1](
                    llvm_intrinsic["llvm.exp", Float64](-g * l1)
                )
            elif kernel == "poly" or kernel == "polynomial":
                var base = SIMD[dtype, 1](g) * dot
                base += SIMD[dtype, 1](coef0)
                value = llvm_intrinsic["llvm.pow", SIMD[dtype, 1]](
                    base, SIMD[dtype, 1](degree)
                )
            elif kernel == "sigmoid":
                var base = SIMD[dtype, 1](g) * dot
                base += SIMD[dtype, 1](coef0)
                value = tanh(base)
            elif kernel == "cosine":
                nx = sqrt(nx)
                ny = sqrt(ny)
                if nx < 10 * epsilon[dtype]():
                    nx = 1
                if ny < 10 * epsilon[dtype]():
                    ny = 1
                value = 0
                for k in range(features):
                    value += (X.unsafe_get(i * xrs + k * xcs) / nx) * (
                        Y.unsafe_get(j * yrs + k * ycs) / ny
                    )
            elif kernel == "chi2":
                value = llvm_intrinsic["llvm.exp", SIMD[dtype, 1]](
                    SIMD[dtype, 1](g) * SIMD[dtype, 1](chi)
                )
            elif kernel == "additive_chi2":
                value = SIMD[dtype, 1](chi)
            if not isfinite(value):
                raise Error("ValueError: nonfinite kernel result")
            result.unsafe_set(i * other_rows + j, value)
    return result^


trait RowKernel(Movable):
    def evaluate[
        dtype: DType
    ](
        self, X: nm.NDArray[dtype], i: Int, Y: nm.NDArray[dtype], j: Int
    ) raises -> SIMD[dtype, 1]:
        ...

    def evaluate_with_params[
        dtype: DType
    ](
        self,
        X: nm.NDArray[dtype],
        i: Int,
        Y: nm.NDArray[dtype],
        j: Int,
        params: Optional[KernelParameters],
    ) raises -> SIMD[dtype, 1]:
        if params and len(params.value()):
            raise Error(
                "TypeError: this callable does not accept kernel_params"
            )
        return self.evaluate(X, i, Y, j)


def pairwise_callable[
    dtype: DType, Kernel: RowKernel
](
    X: nm.NDArray[dtype],
    Y: nm.NDArray[dtype],
    metric: Kernel,
    same_data: Bool = False,
    params: Optional[KernelParameters] = None,
) raises -> nm.NDArray[dtype]:
    """Borrow matrices plus row indices; a typed kernel struct holds parameters.

    No row copies or generic row container. same_data means the caller has
    supplied the same logical array twice, matching sklearn's identity branch.
    """
    comptime assert dtype.is_floating_point()
    validate(X, "X", True)
    validate(Y, "Y", True)
    if X.shape[1] != Y.shape[1] or (same_data and X.shape[0] != Y.shape[0]):
        raise Error("ValueError: incompatible callable input shapes")
    var result = nm.zeros[dtype](nm.Shape(X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i + 1 if same_data else 0, Y.shape[0]):
            var value = metric.evaluate_with_params(X, i, Y, j, params)
            if not isfinite(value):
                raise Error("ValueError: nonfinite callable result")
            result.unsafe_set(i * Y.shape[0] + j, value)
            if same_data:
                result.unsafe_set(j * Y.shape[0] + i, value)
    if same_data:
        for i in range(X.shape[0]):
            var value = metric.evaluate_with_params(X, i, Y, i, params)
            if not isfinite(value):
                raise Error("ValueError: nonfinite callable result")
            result.unsafe_set(i * Y.shape[0] + i, value)
    return result^
