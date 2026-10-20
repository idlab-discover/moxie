"""Dense kernel composition over the accepted NuMojo operations."""
import numojo as nm
from std.math import tanh, isfinite
from std.sys import llvm_intrinsic
from moxie._dense import row_squared_norms
from moxie.svm._contracts import Dense, KernelSpec, validate_matrix


def _exp(x: Float64) -> Float64:
    # Match LIBSVM's platform exponential, including signs of near-zero decisions.
    return llvm_intrinsic["llvm.exp", Float64, has_side_effect=False](x)


def resolve_gamma(x: Dense, mode: String) raises -> Float64:
    validate_matrix(x)
    if mode == "auto":
        return 1.0 / Float64(x.shape[1])
    if mode != "scale":
        raise Error("ValueError: gamma mode must be auto or scale")
    var average = nm.mean[returned_dtype=DType.float64](x)
    var residual = x - average
    var squares = residual * residual
    var variance = nm.sum(squares) / Float64(x.size)
    return 1.0 if variance == 0 else 1.0 / (Float64(x.shape[1]) * variance)


def power(base: Float64, degree: Int) -> Float64:
    var b = base
    var d = degree
    var result = Float64(1)
    while d > 0:
        if d % 2 == 1:
            result *= b
        b *= b
        d //= 2
    return result


def kernel_matrix(x: Dense, train: Dense, spec: KernelSpec) raises -> Dense:
    spec.validate()
    validate_matrix(x)
    validate_matrix(train)
    if spec.kind == "callable":
        raise Error("ValueError: evaluate native callable before engine dispatch")
    if spec.kind == "precomputed":
        if x.shape[1] != train.shape[0]:
            raise Error("ValueError: wrong precomputed column count")
        return x.copy()
    if x.shape[1] != train.shape[1]:
        raise Error("ValueError: incompatible feature count")
    var transposed = train.T()
    var result = x @ transposed
    var xn = row_squared_norms(x)
    var tn = row_squared_norms(train)
    for i in range(x.shape[0]):
        for j in range(train.shape[0]):
            var at = i * train.shape[0] + j
            var value = result.unsafe_get(at)
            if spec.kind == "poly":
                value = power(spec.gamma * value + spec.coef0, spec.degree)
            elif spec.kind == "rbf":
                value = _exp(-spec.gamma * (xn.unsafe_get(i) + tn.unsafe_get(j) - 2 * value))
            elif spec.kind == "sigmoid":
                value = tanh(spec.gamma * value + spec.coef0)
            if not isfinite(value):
                raise Error("ValueError: nonfinite kernel value")
            result.unsafe_set(at, value)
    return result^


def prediction_kernel_matrix(x: Dense, train: Dense, spec: KernelSpec) raises -> Dense:
    """LIBSVM prediction evaluates RBF from differences, avoiding cancellation."""
    if spec.kind != "rbf":
        return kernel_matrix(x, train, spec)
    spec.validate()
    validate_matrix(x)
    validate_matrix(train, x.shape[1])
    var result = nm.zeros[DType.float64](nm.Shape(x.shape[0], train.shape[0]))
    var residual = nm.zeros[DType.float64](nm.Shape(train.shape[0], train.shape[1]))
    for i in range(x.shape[0]):
        for j in range(train.shape[0]):
            for feature in range(train.shape[1]):
                residual.unsafe_set(j * train.shape[1] + feature, x.unsafe_get(i * Int(x.strides[0]) + feature * Int(x.strides[1])) - train.unsafe_get(j * Int(train.strides[0]) + feature * Int(train.strides[1])))
        var squared = row_squared_norms(residual)
        for j in range(train.shape[0]):
            result.unsafe_set(i * train.shape[0] + j, _exp(-spec.gamma * squared.unsafe_get(j)))
    return result^
