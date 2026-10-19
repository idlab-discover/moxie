"""Differential fixtures for precomputed kernels and explicit-Y gradients."""
from numojo import NDArray, Shape, zeros
from std.sys import argv
from moxie.gaussian_process._plumbing import Dense64
from moxie.gaussian_process.kernels import (
    CompoundKernel, ConstantKernel, Kernel, PairwiseKernel, RBF, to_float64,
)


def catalog() raises -> List[Kernel]:
    var pre = Kernel(PairwiseKernel(metric="precomputed"))
    var fixed = Kernel(PairwiseKernel(metric="precomputed", gamma_bounds="fixed"))
    var rbf = Kernel(PairwiseKernel(metric="rbf", gamma=0.4))
    var rbf_fixed = Kernel(PairwiseKernel(metric="rbf", gamma_bounds="fixed"))
    return [pre.copy(), fixed.copy(), pre + pre, pre * pre, pre ** 2.0,
            pre + fixed, pre * fixed, fixed ** 2.0,
            CompoundKernel([pre.copy(), pre.copy()]),
            CompoundKernel([fixed.copy(), fixed.copy()]),
            rbf.copy(), rbf_fixed.copy(), rbf + rbf, rbf * rbf,
            CompoundKernel([rbf.copy(), rbf.copy()]), pre + ConstantKernel(), pre + RBF()]


def parse(tokens: List[String], mut pos: Int) raises -> Dense64:
    var rows = Int(tokens[pos])
    var cols = Int(tokens[pos + 1])
    pos += 2
    var out = zeros[DType.float64](Shape(rows, cols))
    for i in range(rows * cols):
        out.unsafe_set(i, Float64(tokens[pos]))
        pos += 1
    return out^


def emit(key: String, A: Dense64) raises:
    var line = key + "|"
    for i in range(A.ndim):
        if i:
            line += ","
        line += String(A.shape[i])
    line += "|"
    for i in range(A.size):
        if i:
            line += " "
        line += String(A.unsafe_get(i))
    print(line)


def run_case[dtype: DType](name: String, has_y: Bool, X: Dense64, Y: Dense64) raises:
    var Xn = zeros[dtype](X.shape)
    var Yn = zeros[dtype](Y.shape)
    for i in range(X.size):
        Xn.unsafe_set(i, X.unsafe_get(i).cast[dtype]())
    for i in range(Y.size):
        Yn.unsafe_set(i, Y.unsafe_get(i).cast[dtype]())
    var Xd = to_float64(Xn)
    var Yd: Optional[Dense64] = to_float64(Yn) if has_y else None
    var kernels = catalog()
    for i in range(len(kernels)):
        ref kernel = kernels[i]
        var key = name + "|" + String(i)
        try:
            var result = kernel.evaluate(Xd, Yd.copy(), eval_gradient=True)
            emit(key + "|K", result.covariance)
            emit(key + "|G", result.gradient.value())
        except:
            print(key + "|gradient_error")
        try:
            var optional_y: Optional[NDArray[dtype]] = Yn.copy() if has_y else None
            if i == 8 or i == 9 or i == 14:
                emit(key + "|call", kernel.evaluate(Xd, Yd.copy()).covariance)
            else:
                emit(key + "|call", kernel(Xn, optional_y^))
        except:
            print(key + "|call_error")
        try:
            emit(key + "|diag", kernel.diag_of(Xn))
        except:
            print(key + "|diag_error")


def main() raises:
    var args = argv()
    var contents = open(String(args[1]), "r").read()
    var tokens = List[String]()
    for token in contents.split():
        tokens.append(String(token))
    var pos = 0
    while pos < len(tokens):
        var name = tokens[pos]
        var has_y = tokens[pos + 1] == "yes"
        var is32 = tokens[pos + 2] == "f32"
        pos += 3
        var X = parse(tokens, pos)
        var Y = parse(tokens, pos)
        if is32:
            run_case[DType.float32](name, has_y, X, Y)
        else:
            run_case[DType.float64](name, has_y, X, Y)
