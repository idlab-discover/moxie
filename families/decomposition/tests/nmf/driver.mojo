"""Driver: f32|f64 X W0 H0 estimator(nmf|minibatch|function|partial) k(-1 auto,-2 None) init(none|...) solver beta
tol max_iter seed alpha_W alpha_H(-1 same) l1_ratio shuffle batch_size forget fresh update_H."""
from std.sys import argv
from numojo import NDArray
from moxie.decomposition import NMF, MiniBatchNMF, non_negative_factorization
from moxie.decomposition._nmf import NMFComponents, BetaLoss, AlphaH
from decomposition_testing import emit, emit_scalar, read_matrix


def beta_of(text: String) raises -> BetaLoss:
    if text == "frobenius" or text == "kullback-leibler" or text == "itakura-saito": return BetaLoss(text)
    return BetaLoss(Float64(text))


def evaluate[dtype: DType]() raises:
    var args = argv()
    var x = read_matrix[dtype](String(args[2]))
    var w0 = read_matrix[dtype](String(args[3]))
    var h0 = read_matrix[dtype](String(args[4]))
    var kind = String(args[5])
    var k = NMFComponents(Int(args[6]))
    if Int(args[6]) == -1: k = NMFComponents(String("auto"))
    elif Int(args[6]) == -2: k = NMFComponents(NoneType())
    var init: Optional[String] = None
    if String(args[7]) != "none": init = String(args[7])
    var solver = String(args[8])
    var beta = beta_of(String(args[9]))
    var tol = Float64(String(args[10]))
    var max_iter = Int(args[11])
    var seed = Int(args[12])
    var alpha_w = Float64(String(args[13]))
    var alpha_h = AlphaH(Float64(String(args[14])))
    if Float64(String(args[14])) < 0: alpha_h = AlphaH(String("same"))
    var l1 = Float64(String(args[15]))
    var shuffle = Int(args[16]) == 1
    var batch = Int(args[17])
    var forget = Float64(String(args[18]))
    var fresh = Int(args[19]) == 1
    var update_h = Int(args[20]) == 1
    var custom = init and init.value() == "custom"
    var w_given: Optional[NDArray[dtype]] = None
    var h_given: Optional[NDArray[dtype]] = None
    if custom or not update_h:
        w_given = w0.copy()
        h_given = h0.copy()
    if kind == "function":
        var r = non_negative_factorization(x, w_given, h_given, k, init=init, update_H=update_h, solver=solver,
            beta_loss=beta, tol=tol, max_iter=max_iter, alpha_W=alpha_w, alpha_H=alpha_h, l1_ratio=l1,
            random_state=seed, shuffle=shuffle)
        emit("W", r.W)
        emit("H", r.H)
        emit_scalar("n_iter", Float64(r.n_iter))
    elif kind == "nmf":
        var model = NMF[dtype](k, init=init, solver=solver, beta_loss=beta, tol=tol, max_iter=max_iter,
            random_state=seed, alpha_W=alpha_w, alpha_H=alpha_h, l1_ratio=l1, shuffle=shuffle)
        emit("W", model.fit_transform(x, w_given, h_given))
        emit("H", model.components_.value())
        emit_scalar("n_iter", Float64(model.n_iter_))
        emit_scalar("err", model.reconstruction_err_)
        emit("transform", model.transform(x))
    else:
        var model = MiniBatchNMF[dtype](k, init=init, batch_size=batch, beta_loss=beta, tol=tol, max_iter=max_iter,
            alpha_W=alpha_w, alpha_H=alpha_h, l1_ratio=l1, forget_factor=forget, fresh_restarts=fresh,
            random_state=seed)
        if kind == "partial":
            # Two online steps on the full X (the oracle mirrors this exactly).
            model.partial_fit(x, w_given, h_given)
            model.partial_fit(x)
            emit("H", model.components_.value())
            emit_scalar("n_steps", Float64(model.n_steps_))
        else:
            emit("W", model.fit_transform(x, w_given, h_given))
            emit("H", model.components_.value())
            emit_scalar("n_iter", Float64(model.n_iter_))
            emit_scalar("n_steps", Float64(model.n_steps_))
            emit_scalar("err", model.reconstruction_err_)
        emit("transform", model.transform(x))


def main() raises:
    if argv()[1] == "f32": evaluate[DType.float32]()
    else: evaluate[DType.float64]()
