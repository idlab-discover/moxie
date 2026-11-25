"""Driver: f32|f64 X.txt W0.txt held_out.txt k(-1=None) algorithm whiten fun alpha max_iter tol solver seed use_w_init mode(fit|fit_transform|function)."""
from std.sys import argv
from numojo import NDArray
from moxie.decomposition import FastICA, fastica
from moxie.decomposition._fastica import WhitenOption
from decomposition_testing import emit, emit_scalar, read_matrix


def evaluate[dtype: DType]() raises:
    var args = argv()
    var x = read_matrix[dtype](String(args[2]))
    var k: Optional[Int] = None
    if Int(args[5]) >= 0: k = Int(args[5])
    var whiten = WhitenOption(String(args[7]))
    if String(args[7]) == "false": whiten = WhitenOption(False)
    var fun_args: Optional[Dict[String, Float64]] = None
    if Float64(String(args[9])) != 1.0:
        var d = Dict[String, Float64]()
        d["alpha"] = Float64(String(args[9]))
        fun_args = d^
    var w_init: Optional[NDArray[DType.float64]] = None
    if Int(args[14]) == 1: w_init = read_matrix[DType.float64](String(args[3]))
    var mode = String(args[15])
    if mode == "function":
        var result = fastica(x, k, algorithm=String(args[6]), whiten=whiten, fun=String(args[8]), fun_args=fun_args,
            max_iter=Int(args[10]), tol=Float64(String(args[11])), w_init=w_init, whiten_solver=String(args[12]),
            random_state=Int(args[13]))
        if result.K: emit("K", result.K.value())
        emit("W", result.W)
        emit("sources", result.S.value())
        if result.X_mean: emit("mean", result.X_mean.value())
        emit_scalar("n_iter", Float64(result.n_iter))
        return
    var model = FastICA[dtype](k, algorithm=String(args[6]), whiten=whiten, fun=String(args[8]), fun_args=fun_args,
        max_iter=Int(args[10]), tol=Float64(String(args[11])), w_init=w_init, whiten_solver=String(args[12]),
        random_state=Int(args[13]))
    if mode == "fit_transform":
        emit("sources", model.fit_transform(x))
    else:
        model.fit(x)
    emit("components", model.components_.value())
    emit("mixing", model.mixing_.value())
    if model.mean_: emit("mean", model.mean_.value())
    if model.whitening_: emit("whitening", model.whitening_.value())
    emit_scalar("n_iter", Float64(model.n_iter_))
    var projected = model.transform(x)
    emit("transform", projected)
    emit("inverse", model.inverse_transform(projected))
    var held_out = read_matrix[dtype](String(args[4]))
    var held_projection = model.transform(held_out)
    emit("held_out", held_projection)
    emit("held_inverse", model.inverse_transform(held_projection))


def main() raises:
    if argv()[1] == "f32": evaluate[DType.float32]()
    else: evaluate[DType.float64]()
