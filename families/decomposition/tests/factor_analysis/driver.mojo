"""Driver: f32|f64 X.txt psi.txt held_out.txt k(-1=None) svd_method rotation(none|varimax|quartimax) max_iter tol iterated_power seed use_psi(0|1)."""
from std.sys import argv
from moxie.decomposition import FactorAnalysis
from decomposition_testing import emit, emit_scalar, read_matrix


def evaluate[dtype: DType]() raises:
    var args = argv()
    var x = read_matrix[dtype](String(args[2]))
    var held_out = read_matrix[dtype](String(args[4]))
    var k: Optional[Int] = None
    if Int(args[5]) >= 0: k = Int(args[5])
    var rotation: Optional[String] = None
    if String(args[7]) != "none": rotation = String(args[7])
    var init: Optional[List[Float64]] = None
    if Int(args[12]) == 1:
        var psi = read_matrix[DType.float64](String(args[3]))
        var values = List[Float64]()
        for i in range(psi.size): values.append(Float64(psi.unsafe_get(i)))
        init = values^
    var model = FactorAnalysis[dtype](k, svd_method=String(args[6]), rotation=rotation,
        max_iter=Int(args[8]), tol=Float64(String(args[9])), iterated_power=Int(args[10]),
        random_state=Int(args[11]), noise_variance_init=init)
    var scores = model.fit_transform(x)
    emit("transform", scores)
    emit("held_out", model.transform(held_out))
    emit("components", model.components_.value())
    emit("noise", model.noise_variance_.value())
    emit("mean", model.mean_.value())
    emit_scalar("n_iter", Float64(model.n_iter_))
    for i in range(len(model.loglike_)): emit_scalar("loglike", model.loglike_[i])
    emit("covariance", model.get_covariance())
    emit("precision", model.get_precision())
    emit("score_samples", model.score_samples(x))
    emit_scalar("score", model.score(x))


def main() raises:
    if argv()[1] == "f32": evaluate[DType.float32]()
    else: evaluate[DType.float64]()
