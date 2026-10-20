"""Native differential driver: driver f32|f64 X.txt k algorithm n_iter n_oversamples normalizer seed tol."""
from std.sys import argv
from moxie.decomposition import TruncatedSVD
from decomposition_testing import emit, read_matrix


def evaluate[dtype: DType]() raises:
    var args = argv()
    var x = read_matrix[dtype](String(args[2]))
    var model = TruncatedSVD[dtype](Int(args[3]), algorithm=String(args[4]), n_iter=Int(args[5]),
        n_oversamples=Int(args[6]), power_iteration_normalizer=String(args[7]),
        random_state=Int(args[8]), tol=Float64(String(args[9])))
    var scores = model.fit_transform(x)
    emit("fit_transform", scores)
    emit("components", model.components_.value())
    emit("variance", model.explained_variance_.value())
    emit("ratio", model.explained_variance_ratio_.value())
    emit("singular", model.singular_values_.value())
    var projected = model.transform(x)
    emit("transform", projected)
    emit("inverse", model.inverse_transform(projected))


def main() raises:
    if argv()[1] == "f32": evaluate[DType.float32]()
    else: evaluate[DType.float64]()
