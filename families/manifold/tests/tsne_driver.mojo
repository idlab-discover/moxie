"""Native t-SNE entry points used by the pinned sklearn comparison."""
from std.sys import argv
from numojo import NDArray, Shape, zeros
from moxie.manifold import TSNE, TSNEMetricParams
from moxie.manifold._t_sne import (
    _gradient_descent, _joint_probabilities, _kl_divergence, _neighbor_probabilities, _squared_distances,
)
from moxie.manifold._barnes_hut_tsne import _barnes_hut_objective

from trustworthiness_driver import read_matrix


def write_array[dtype: DType](array: NDArray[dtype]):
    for i in range(array.size):
        print(Float64(array.unsafe_get(i)))


def fit[input_dtype: DType, output_dtype: DType]() raises:
    var args = argv()
    var X = read_matrix[input_dtype](String(args[4]))
    var init = String(args[5])
    var components = Int(String(args[6]))
    var metric = String(args[7])
    var minimum_gradient = Float64(String(args[11]))
    var perplexity = Float64(String(args[14]))
    var method = "exact"
    var angle: Float64 = 0.5
    if len(args) > 16:
        method = String(args[15])
        angle = Float64(String(args[16]))
    var model = TSNE[input_dtype, output_dtype](
        components, method=method, angle=angle, perplexity=perplexity, metric=metric,
        max_iter=Int(String(args[8])), early_exaggeration=Float64(String(args[9])),
        min_grad_norm=minimum_gradient, n_iter_without_progress=Int(String(args[12])),
        random_state=Int(String(args[13])),
    )
    if not (init == "pca" or init == "random"):
        model.init = read_matrix[output_dtype](init)
    else:
        model.init = init
    if args[10] != "auto":
        model.learning_rate = Float64(String(args[10]))
    if args[1] == "init":
        write_array(model._initial_coordinates(X))
        return
    var Y = model.fit_transform(X)
    print(model.n_iter_, model.kl_divergence_, model.learning_rate_, model.n_features_in_)
    print(Y.shape[0], Y.shape[1], String(output_dtype))
    write_array(Y)


def pairwise[input_dtype: DType]() raises:
    var args = argv()
    var X = read_matrix[input_dtype](String(args[3]))
    var metric = String(args[4])
    var params: Optional[TSNEMetricParams[input_dtype]] = None
    if len(args) > 5:
        params = TSNEMetricParams[input_dtype](p=Float64(String(args[5])))
    write_array(_squared_distances(X, metric, params))


def objective[dtype: DType]() raises:
    var args = argv()
    var Y = read_matrix[dtype](String(args[3]))
    var P = _joint_probabilities(
        read_matrix[DType.float32](String(args[4])), Float64(String(args[5])),
    )
    var weights = zeros[DType.float64](Shape(P.size))
    var gradient = zeros[DType.float64](Y.shape)
    var dof = Float64(max(Y.shape[1] - 1, 1))
    print(_kl_divergence(Y, P, weights, gradient, dof, True))
    for i in range(gradient.size):
        print(Float64(gradient.unsafe_get(i).cast[dtype]() * SIMD[dtype, 1](2 * (dof + 1) / dof)))
    write_array(P)


def barnes_hut_objective[dtype: DType]() raises:
    var args = argv()
    var X = read_matrix[dtype](String(args[3]))
    var Y = read_matrix[DType.float32](String(args[4]))
    var angle = Float32(Float64(String(args[7])))
    var P = _neighbor_probabilities(X, String(args[5]), None, Float64(String(args[6])))
    var gradient = zeros[DType.float64](Y.shape)
    print(_barnes_hut_objective(Y, P, 1, gradient, angle, True))
    var dof = Float32(max(Y.shape[1] - 1, 1))
    for i in range(gradient.size):
        print(Float32(gradient.unsafe_get(i)) * (2 * (dof + 1) / dof))
    for pair in P:
        print(pair[0], pair[1], pair[2])


def descent() raises:
    var args = argv()
    var Y = read_matrix[DType.float64](String(args[2]))
    var n = Y.shape[0]
    # Zero probabilities give a constant loss and a nonzero gradient.
    var P = zeros[DType.float64](Shape(n * (n - 1) // 2))
    var weights = zeros[DType.float64](Shape(P.size))
    var gradient = zeros[DType.float64](Y.shape)
    var loss, iteration = _gradient_descent(Y, P, weights, gradient, 1, 0, 150, -1, 0.8, 2.0, 0, 0)
    print(loss, iteration)
    write_array(Y)


def main() raises:
    var args = argv()
    if args[1] == "descent":
        descent()
    elif args[1] == "distances":
        if args[2] == "float32":
            pairwise[DType.float32]()
        else:
            pairwise[DType.float64]()
    elif args[1] == "barnes_hut_objective":
        if args[2] == "float32":
            barnes_hut_objective[DType.float32]()
        else:
            barnes_hut_objective[DType.float64]()
    elif args[1] == "objective":
        if args[2] == "float32":
            objective[DType.float32]()
        else:
            objective[DType.float64]()
    elif args[2] == "float32":
        fit[DType.float32, DType.float32]()
    elif args[3] == "float32":
        fit[DType.float64, DType.float32]()
    else:
        fit[DType.float64, DType.float64]()
