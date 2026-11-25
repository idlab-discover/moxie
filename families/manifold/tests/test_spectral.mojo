from std.math import abs
from std.testing import assert_equal, assert_raises, assert_true
from numojo import NDArray, NDArrayStrides, Shape, zeros
from moxie.base import clone
from moxie.manifold import SpectralEmbedding, SpectralEmbeddingParams, spectral_embedding


def _check_close[dtype: DType](left: NDArray[dtype], right: NDArray[dtype]) raises:
    assert_equal(left.shape[0], right.shape[0])
    assert_equal(left.shape[1], right.shape[1])
    for i in range(left.size):
        assert_true(abs(Float64(left.unsafe_get(i)) - Float64(right.unsafe_get(i))) < 1e-6)


def _weighted_graph[dtype: DType]() raises -> NDArray[dtype]:
    var graph = zeros[dtype](Shape(4, 4))
    for i in range(4):
        graph.unsafe_set(i * 4 + i, 11)
    for i in range(3):
        graph.unsafe_set(i * 4 + i + 1, SIMD[dtype, 1](i + 1))
        graph.unsafe_set((i + 1) * 4 + i, SIMD[dtype, 1](i + 1))
    return graph^


def _lifecycle[dtype: DType]() raises:
    var graph = _weighted_graph[dtype]()
    var model = SpectralEmbedding[dtype](affinity="precomputed", eigen_solver="dense")
    assert_true(model.pairwise_input())
    with assert_raises(contains="AttributeError"):
        _ = model.get_n_features_in()

    var result = model.fit_transform(graph)
    var expected = spectral_embedding(graph, n_components=2, eigen_solver="dense")
    _check_close(result, expected)
    assert_equal(model.get_n_features_in(), 4)
    assert_equal(len(model.fit_warnings), 0)
    assert_true(not model.gamma_)
    result.unsafe_set(0, 99)
    graph.unsafe_set(0, 99)
    _check_close(model.embedding_.value(), expected)
    assert_equal(Float64(model.affinity_matrix_.value().unsafe_get(0)), 11.0)

    var fresh = clone(model)
    assert_true(not fresh.__sklearn_is_fitted__())
    assert_true(not fresh.affinity_matrix_)
    assert_true(fresh.pairwise_input())

    # Failed fits preserve the prior snapshot and warnings.
    _ = model.set_params(n_components=0)
    with assert_raises(contains="n_components"):
        _ = model.fit(graph)
    _check_close(model.embedding_.value(), expected)
    assert_equal(len(model.fit_warnings), 0)

    _ = model.set_params(n_components=1, affinity="rbf", gamma=0.0)
    var features = zeros[dtype](Shape(3, 2))
    features.unsafe_set(4, 3)
    _ = model.fit(features, y=features.copy())
    assert_true(not model.pairwise_input())
    assert_equal(model.get_n_features_in(), 2)
    assert_equal(model.gamma_.value(), 0.0)
    assert_equal(model.embedding_.value().shape[1], 1)
    for i in range(9):
        assert_equal(Float64(model.affinity_matrix_.value().unsafe_get(i)), 1.0)
    var params: SpectralEmbeddingParams = model.get_params()
    params.gamma = None
    _ = model.set_params(params)
    _ = model.fit(features)
    assert_equal(model.gamma_.value(), 0.5)


def _strides_and_warnings() raises:
    var graph = _weighted_graph[DType.float64]()
    var reversed = graph.view_with_layout(Shape(4, 4), NDArrayStrides(-4, -1), 15)
    var model = SpectralEmbedding(affinity="precomputed", eigen_solver="dense")
    var strided = model.fit_transform(reversed)
    _check_close(strided, model.fit_transform(reversed.contiguous()))

    graph.unsafe_set(1, 0)
    graph.unsafe_set(4, 0)
    graph.unsafe_set(6, 1)
    _ = model.fit(graph)
    assert_equal(len(model.fit_warnings), 2)
    assert_equal(model.fit_warnings[0].category, "UserWarning")
    assert_true("not symmetric" in model.fit_warnings[0].message)
    assert_true("not fully connected" in model.fit_warnings[1].message)
    assert_equal(Float64(model.affinity_matrix_.value().unsafe_get(6)), 1.0)
    var copied = model.copy()
    _ = model.fit(_weighted_graph[DType.float64]())
    assert_equal(len(model.fit_warnings), 0)
    assert_equal(len(copied.fit_warnings), 2)


def _invalid_and_unsupported() raises:
    var graph = _weighted_graph[DType.float64]()
    var default_model = SpectralEmbedding()
    assert_equal(default_model.get_params().affinity, "nearest_neighbors")
    assert_true(not default_model.get_params().eigen_solver)
    assert_true(not default_model.get_params().eigen_tol)
    with assert_raises(contains="requires sparse graphs"):
        _ = default_model.fit(graph)
    with assert_raises(contains="selects ARPACK"):
        _ = spectral_embedding(graph, n_components=2)
    var excluded: List[String] = ["arpack", "lobpcg", "amg"]
    for solver in excluded:
        with assert_raises(contains="NotImplementedError"):
            _ = spectral_embedding(graph, n_components=2, eigen_solver=solver.copy())
    with assert_raises(contains="eigen_tol"):
        _ = spectral_embedding(graph, n_components=2, eigen_solver="dense", eigen_tol=-1.0)
    with assert_raises(contains="random_state"):
        _ = spectral_embedding(graph, n_components=2, eigen_solver="dense", random_state=-1)
    with assert_raises(contains="n_components"):
        _ = spectral_embedding(graph, n_components=0, eigen_solver="dense")
    assert_equal(
        spectral_embedding(graph, n_components=9, eigen_solver="dense").shape[1],
        3,
    )
    assert_equal(
        spectral_embedding(zeros[DType.float64](Shape(1, 1)), eigen_solver="dense").shape[1],
        0,
    )
    assert_equal(
        spectral_embedding(graph, n_components=4, drop_first=False, eigen_solver="dense").shape[1],
        4,
    )
    var invalid_shape = zeros[DType.float64](Shape(4, 3))
    with assert_raises(contains="square"):
        _ = spectral_embedding(invalid_shape, n_components=2, eigen_solver="dense")
    graph.unsafe_set(1, -1)
    with assert_raises(contains="nonnegative"):
        _ = spectral_embedding(graph, n_components=2, eigen_solver="dense")
    graph.unsafe_set(1, Float64("nan"))
    with assert_raises(contains="finite"):
        _ = spectral_embedding(graph, n_components=2, eigen_solver="dense")


def main() raises:
    _lifecycle[DType.float32]()
    _lifecycle[DType.float64]()
    _strides_and_warnings()
    _invalid_and_unsupported()
    print("PASS: dense spectral contracts")
