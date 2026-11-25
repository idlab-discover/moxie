"""Drive dense spectral entry points for pinned scikit-learn parity."""
from std.math import abs
from std.sys import argv
from numojo import NDArray, Shape, zeros
from moxie.manifold import SpectralEmbedding, spectral_embedding
from moxie.neighbors._distance import at
from trustworthiness_driver import read_matrix


def _custom_affinity[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    var n = x.shape[0]
    var graph = zeros[dtype](Shape(n, n))
    for row in range(n):
        for col in range(n):
            var weight = 1 / (1 + abs(at(x, row, 0) - at(x, col, 0)))
            if row < col:
                weight += 0.125
            graph.unsafe_set(row * n + col, SIMD[dtype, 1](weight))
    return graph^


def _write_matrix[dtype: DType](x: NDArray[dtype]) raises:
    print(x.shape[0], x.shape[1], String(dtype))
    for i in range(x.size):
        print(Float64(x.unsafe_get(i)))


def _compare[dtype: DType]() raises:
    var args = argv()
    var operation = String(args[1])
    var x = read_matrix[dtype](String(args[3]))
    var n_components = Int(String(args[4]))
    if operation == "function":
        _write_matrix(spectral_embedding(
            x, n_components=n_components, eigen_solver="dense", random_state=17,
            eigen_tol=1e-7, norm_laplacian=args[5] == "True", drop_first=args[6] == "True",
        ))
        return

    var gamma: Optional[Float64] = None
    if args[6] != "None":
        gamma = Float64(String(args[6]))
    var model = SpectralEmbedding[dtype](
        n_components, affinity=String(args[5]), gamma=gamma,
        random_state=17, eigen_solver="dense",
    )
    if args[5] == "callable":
        _ = model.set_params(affinity_function=_custom_affinity[dtype])
    var embedding = model.fit_transform(x)
    print(model.n_features_in_, len(model.fit_warnings))
    for warning in model.fit_warnings:
        print(warning.category)
        print(warning.message)
    if model.gamma_:
        print(model.gamma_.value())
    else:
        print("None")
    _write_matrix(model.affinity_matrix_.value())
    _write_matrix(embedding)


def main() raises:
    if argv()[2] == "float32":
        _compare[DType.float32]()
    else:
        _compare[DType.float64]()
