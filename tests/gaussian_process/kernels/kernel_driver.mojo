"""Native side of the sklearn kernel parity check (Python is fixture tooling only).

Usage: kernel_driver <dataset-file>. Every catalog kernel is evaluated on every
dataset; results are printed as `R|kernel|dataset|key|shape|values` records.
The catalog must match `KERNELS` in tests/parity.py entry for entry.
"""
from numojo import NDArray, Shape, zeros
from std.collections import Dict
from std.sys import argv

from moxie.gaussian_process.kernels import (
    Bounds,
    CompoundKernel,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Exponentiation,
    Kernel,
    Matern,
    PairwiseKernel,
    Product,
    RBF,
    RationalQuadratic,
    Sum,
    WhiteKernel,
    to_float64,
)
from moxie.gaussian_process._plumbing import Dense64

from std.math import inf

comptime INF = inf[DType.float64]()


def pairwise(metric: String, gamma: Float64 = 0.4) -> Kernel:
    return PairwiseKernel(gamma=gamma, metric=metric)


def catalog() raises -> List[Kernel]:
    var poly_kwargs = Dict[String, Float64]()
    poly_kwargs["degree"] = 2.0
    poly_kwargs["coef0"] = 0.5
    var sigmoid_kwargs = Dict[String, Float64]()
    sigmoid_kwargs["coef0"] = -0.2
    sigmoid_kwargs["degree"] = 7.0  # filtered out, as sklearn filter_params
    var aniso: List[Float64] = [0.5, 2.0, 1.2]
    var ks = List[Kernel]()
    ks.append(ConstantKernel(2.5))
    ks.append(ConstantKernel(2.5, "fixed"))
    ks.append(WhiteKernel(0.3))
    ks.append(RBF(1.3))
    ks.append(RBF(aniso))
    ks.append(RBF(1.3, "fixed"))
    ks.append(Matern(0.9, nu=0.5))
    ks.append(Matern(0.9, nu=1.5))
    ks.append(Matern(0.9, nu=2.5))
    ks.append(Matern(0.9, nu=INF))
    ks.append(Matern(0.9, nu=0.3))
    ks.append(Matern(1.4, nu=3.7))
    ks.append(Matern(aniso, nu=0.5))
    ks.append(Matern(aniso, nu=1.5))
    ks.append(Matern(aniso, nu=2.5))
    ks.append(Matern(aniso, nu=INF))
    ks.append(Matern(aniso, nu=1.1))
    ks.append(RationalQuadratic(0.8, alpha=2.5))
    ks.append(RationalQuadratic(0.8, alpha=2.5, alpha_bounds="fixed"))
    ks.append(ExpSineSquared(1.1, periodicity=2.3))
    ks.append(ExpSineSquared(1.1, periodicity=2.3, periodicity_bounds="fixed"))
    ks.append(DotProduct(0.7))
    ks.append(pairwise("linear"))
    ks.append(pairwise("rbf"))
    ks.append(pairwise("laplacian"))
    ks.append(PairwiseKernel(gamma=0.4, metric="poly", pairwise_kernels_kwargs=poly_kwargs.copy()))
    ks.append(pairwise("polynomial", 0.2))
    ks.append(PairwiseKernel(gamma=0.3, metric="sigmoid", pairwise_kernels_kwargs=sigmoid_kwargs.copy()))
    ks.append(pairwise("cosine"))
    ks.append(pairwise("chi2"))
    ks.append(pairwise("additive_chi2"))
    ks.append(ConstantKernel(2.0) * RBF(1.5) + WhiteKernel(0.1))
    ks.append((RBF(0.7) + DotProduct(0.3)) ** 2.0)
    ks.append(RationalQuadratic(0.8, alpha=1.5) * ExpSineSquared(1.1, periodicity=2.3))
    ks.append(Exponentiation(Matern(aniso, nu=2.5), 1.7))
    ks.append(ConstantKernel(0.5, "fixed") + RBF(1.2) * 3.0)
    ks.append(
        ConstantKernel(1.3) * (RBF(aniso) + Matern(1.1, nu=0.5)) * ExpSineSquared(0.9, 3.0)
        + WhiteKernel(0.05, Bounds(1e-3, 1.0))
    )
    ks.append(Sum(Product(ConstantKernel(0.7), Matern(1.4, nu=3.7)), WhiteKernel(0.2)))
    ks.append(1.5 + RBF(0.8))
    ks.append(2.0 * Matern(0.6, Bounds(1e-2, 10.0), nu=1.5))
    ks.append(CompoundKernel([RBF(1.0), Matern(2.0, nu=2.5)]))
    ks.append(CompoundKernel([ConstantKernel(2.0), WhiteKernel(0.4), DotProduct(1.1)]))
    return ks^


def emit(kernel: Int, dataset: String, key: String, a: Dense64) raises:
    var line = String("R|") + String(kernel) + "|" + dataset + "|" + key + "|"
    for i in range(a.ndim):
        if i > 0:
            line += ","
        line += String(a.shape[i])
    line += "|"
    for i in range(a.size):
        if i > 0:
            line += " "
        line += String(a.unsafe_get(i))
    print(line)


def emit_text(kernel: Int, dataset: String, key: String, text: String):
    print("R|" + String(kernel) + "|" + dataset + "|" + key + "||" + text)


def parse_matrix(tokens: List[String], mut pos: Int) raises -> Dense64:
    var rows = Int(tokens[pos])
    var cols = Int(tokens[pos + 1])
    pos += 2
    var out = zeros[DType.float64](Shape(rows, cols))
    for i in range(rows * cols):
        out.unsafe_set(i, Float64(tokens[pos]))
        pos += 1
    return out^


def narrow(X: Dense64) raises -> NDArray[DType.float32]:
    var out = zeros[DType.float32](Shape(X.shape[0], X.shape[1]))
    for i in range(X.size):
        out.unsafe_set(i, X.unsafe_get(i).cast[DType.float32]())
    return out^


def run_dataset(name: String, float32: Bool, X: Dense64, Y: Dense64) raises:
    var kernels = catalog()
    for i in range(len(kernels)):
        try:
            report(i, kernels[i], name, float32, X, Y)
        except e:
            emit_text(i, name, "error", String(e))


def report(i: Int, k: Kernel, name: String, float32: Bool, X: Dense64, Y: Dense64) raises:
    emit_text(i, name, "repr", String(k))
    emit_text(i, name, "stationary", String(k.is_stationary()))
    emit_text(i, name, "vector_input", String(k.requires_vector_input()))
    var names = String()
    for spec in k.hyperparameters():
        names += spec.name + ":" + String(spec.n_elements) + ":" + String(spec.fixed) + ";"
    emit_text(i, name, "hyperparameters", names)
    var keys = String()
    for key in k.get_params().keys():
        keys += key + ";"
    emit_text(i, name, "params", keys)
    emit(i, name, "theta", k.theta())
    emit(i, name, "bounds", k.bounds())
    var Xk = X.copy()
    var Yk = Y.copy()
    if float32:
        Xk = to_float64(narrow(X))
        Yk = to_float64(narrow(Y))
    var full = k.evaluate(Xk, eval_gradient=True)
    emit(i, name, "K", full.covariance)
    emit(i, name, "gradient", full.gradient.value())
    emit(i, name, "KXY", k.evaluate(Xk, Yk.copy()).covariance)
    emit(i, name, "diag", k.diag(Xk))
    var shifted = k.theta()
    for j in range(shifted.size):
        shifted.unsafe_set(j, shifted.unsafe_get(j) + 0.1)
    emit(i, name, "K_shifted", k.clone_with_theta(shifted).evaluate(Xk).covariance)
    if float32 and k.name() != "CompoundKernel":
        emit(i, name, "K_call32", k(narrow(X)))


def main() raises:
    var args = argv()
    var text = open(String(args[1]), "r").read()
    var tokens = List[String]()
    for token in text.split():
        tokens.append(String(token))
    var pos = 0
    while pos < len(tokens):
        var name = tokens[pos]
        var float32 = tokens[pos + 1] == "f32"
        pos += 2
        var X = parse_matrix(tokens, pos)
        var Y = parse_matrix(tokens, pos)
        run_dataset(name, float32, X, Y)
