"""Native side of GPR parity. Usage: gpr_driver <cases-file>.

Each case line: `name case_id dtype X y Xt` (matrices as `rows cols values...`).
Estimator configurations are selected by case_id and must match CASES in
tests/parity.py. Output records: `R|name|key|shape|values`.
"""
from numojo import NDArray, Shape, zeros
from std.math import inf
from std.sys import argv

from moxie.gaussian_process import GenericGaussianProcessRegressor
from moxie.gaussian_process._plumbing import Dense64
from moxie.gaussian_process.kernels import (
    Bounds,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)


def emit(name: String, key: String, a: Dense64) raises:
    var line = String("R|") + name + "|" + key + "|"
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


def emit_scalar(name: String, key: String, value: Float64) raises:
    var a = zeros[DType.float64](Shape(1))
    a.unsafe_set(0, value)
    emit(name, key, a)


def parse(tokens: List[String], mut pos: Int) raises -> Dense64:
    var rows = Int(tokens[pos])
    var cols = Int(tokens[pos + 1])
    pos += 2
    var out = zeros[DType.float64](Shape(rows, cols)) if cols > 0 else zeros[DType.float64](Shape(rows))
    for i in range(rows * max(cols, 1)):
        out.unsafe_set(i, Float64(tokens[pos]))
        pos += 1
    return out^


def convert[dtype: DType](A: Dense64) raises -> NDArray[dtype]:
    var out = zeros[dtype](A.shape)
    for i in range(A.size):
        out.unsafe_set(i, A.unsafe_get(i).cast[dtype]())
    return out^


def configure[dtype: DType](case_id: Int, n: Int) raises -> GenericGaussianProcessRegressor[dtype]:
    comptime GPR = GenericGaussianProcessRegressor[dtype]
    var aniso: List[Float64] = [1.0, 2.0]
    if case_id == 0:
        return GPR()
    if case_id == 1:
        return GPR(ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1), normalize_y=True, n_restarts_optimizer=3, random_state=0)
    if case_id == 2:
        var alpha = List[Float64]()
        for i in range(n):
            alpha.append(1e-3 * Float64(1 + i % 3))
        var model = GPR(ConstantKernel(1.0) * Matern(aniso, nu=2.5))
        _ = model.set_params(alpha=alpha^)
        return model^
    if case_id == 3:
        return GPR(ExpSineSquared(1.0, 3.0, periodicity_bounds=Bounds(1.0, 10.0)) + WhiteKernel(1e-2), n_restarts_optimizer=2, random_state=3)
    if case_id == 4:
        return GPR(DotProduct(1.0) + WhiteKernel(0.5), normalize_y=True)
    if case_id == 5:
        var model = GPR(RBF(0.7), optimizer=None)
        _ = model.set_params(alpha=1e-2)
        return model^
    if case_id == 6:
        return GPR(ConstantKernel(2.0) * Matern(1.0, nu=0.8) + WhiteKernel(1e-2))
    if case_id == 7:
        return GPR(RationalQuadratic(1.0, 1.0), normalize_y=True)
    if case_id == 8:
        return GPR(ConstantKernel(1.0, "fixed") * RBF(1.0, Bounds(1e-2, 1e2)), n_targets=2)
    raise Error("unknown case " + String(case_id))


def run_case[dtype: DType](name: String, case_id: Int, X: Dense64, y: Dense64, Xt: Dense64) raises:
    var Xn = convert[dtype](X)
    var yn = convert[dtype](y)
    var Xtn = convert[dtype](Xt)
    var gpr = configure[dtype](case_id, X.shape[0])
    var prior = gpr.predict(Xtn, return_std=True)
    emit(name, "prior_mean", prior.mean)
    emit(name, "prior_std", prior.std.value())
    _ = gpr.fit(Xn, yn)
    emit(name, "theta", gpr.kernel_.theta())
    emit_scalar(name, "lml", gpr.log_marginal_likelihood_value_)
    emit(name, "alpha_", gpr.alpha_)
    emit(name, "L_", gpr.L_)
    var with_std = gpr.predict(Xtn, return_std=True)
    emit(name, "mean", with_std.mean)
    emit(name, "std", with_std.std.value())
    emit(name, "cov", gpr.predict(Xtn, return_std=False, return_cov=True).covariance.value())
    var shifted = gpr.kernel_.theta()
    for i in range(shifted.size):
        shifted.unsafe_set(i, shifted.unsafe_get(i) + 0.1)
    var lml = gpr.log_marginal_likelihood(shifted^, eval_gradient=True)
    emit_scalar(name, "lml_shifted", lml.value)
    emit(name, "lml_gradient", lml.gradient.value())
    emit(name, "samples", gpr.sample_y(Xtn, 3, 0))
    emit_scalar(name, "score", gpr.score(Xn, yn))
    emit_scalar(name, "warnings", Float64(len(gpr.convergence_warnings_)))
    var inplace_theta = gpr.kernel_.theta()
    for i in range(inplace_theta.size):
        inplace_theta.unsafe_set(i, inplace_theta.unsafe_get(i) + 0.1)
    emit_scalar(name, "lml_inplace", gpr.log_marginal_likelihood(inplace_theta^, clone_kernel=False).value)
    emit(name, "theta_inplace", gpr.kernel_.theta())
    emit_scalar(name, "lml_cached", gpr.log_marginal_likelihood().value)


def main() raises:
    var args = argv()
    var text = open(String(args[1]), "r").read()
    var tokens = List[String]()
    for token in text.split():
        tokens.append(String(token))
    var pos = 0
    while pos < len(tokens):
        var name = tokens[pos]
        var case_id = Int(tokens[pos + 1])
        var kind = tokens[pos + 2]
        pos += 3
        var X = parse(tokens, pos)
        var y = parse(tokens, pos)
        var Xt = parse(tokens, pos)
        try:
            if kind == "f32":
                run_case[DType.float32](name, case_id, X, y, Xt)
            else:
                run_case[DType.float64](name, case_id, X, y, Xt)
        except e:
            print("R|" + name + "|error||" + String(e))
