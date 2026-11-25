from std.math import abs, isfinite, nan
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from numojo import NDArray, NDArrayStrides, Shape, zeros
from moxie._random import seed_global_random_state
from moxie.base import clone
from moxie.manifold import TSNE, TSNEMetricParams
from moxie.neighbors._distance import at


def points[dtype: DType]() raises -> NDArray[dtype]:
    var X = zeros[dtype](Shape(6, 3))
    for i in range(6):
        X.unsafe_set(i * 3, SIMD[dtype, 1](i))
        X.unsafe_set(i * 3 + 1, SIMD[dtype, 1](i % 3))
        X.unsafe_set(i * 3 + 2, SIMD[dtype, 1](i * (i % 2)))
    return X^


def repeat_and_clone[dtype: DType](method: String) raises:
    var X = points[dtype]()
    var model = TSNE[dtype](perplexity=2, method=method, init="random", random_state=17,
                           max_iter=260, learning_rate=2.0)
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = model.get_feature_names_out()
    with assert_raises(contains="AttributeError"):
        _ = model.get_n_features_in()
    var Y = model.fit_transform(X)
    var expected: NDArray[DType.float32] = Y.copy()
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.get_n_features_in(), 3)
    assert_equal(model.n_iter_, 259)
    assert_equal(model.learning_rate_, 2.0)
    assert_true(isfinite(model.kl_divergence_))
    assert_equal(model.get_feature_names_out()[1], "tsne1")
    var wrong: List[String] = ["wrong"]
    with assert_raises(contains="input_features length"):
        _ = model.get_feature_names_out(wrong^)

    Y.unsafe_set(0, 10000)
    assert_equal(model.embedding_.value().unsafe_get(0), expected.unsafe_get(0))
    _ = model.fit(X, y=X.copy())
    for i in range(expected.size):
        assert_equal(model.embedding_.value().unsafe_get(i), expected.unsafe_get(i))

    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.get_params().max_iter, 260)
    _ = fresh.set_params(n_components=1).fit(X)
    assert_equal(fresh.embedding_.value().shape[1], 1)
    assert_equal(model.embedding_.value().shape[1], 2)

    var backwards = X.view_with_layout(Shape(6, 3), NDArrayStrides(-3, 1), 15)
    var packed = zeros[dtype](Shape(6, 3))
    for i in range(6):
        for j in range(3):
            packed.unsafe_set(i * 3 + j, X.unsafe_get((5 - i) * 3 + j))
    var strided_result = model.fit_transform(backwards)
    var packed_result = model.fit_transform(packed)
    for i in range(strided_result.size):
        assert_equal(strided_result.unsafe_get(i), packed_result.unsafe_get(i))


def explicit_configuration() raises:
    var X = points[DType.float64]()
    var storage = zeros[DType.float64](Shape(6, 3))
    storage.unsafe_set(0, 0.01)
    var init = storage.view_with_layout(Shape(6, 2), NDArrayStrides(-3, 1), 15)
    var model = TSNE[DType.float64, DType.float64](
        perplexity=2, method="exact", init=init^, max_iter=260, learning_rate=2.0,
    )
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    storage.unsafe_set(0, 8.0)
    var params = model.get_params()
    params.init[NDArray[DType.float64]].unsafe_set(0, 10.0)
    # Both construction and get_params detach their array configuration.
    assert_equal(model.init[NDArray[DType.float64]].unsafe_get(-15), 0.01)
    assert_equal(fresh.init[NDArray[DType.float64]].unsafe_get(0), 0.0)
    params.metric = "minkowski"
    params.metric_params = TSNEMetricParams[DType.float64](p=1.5)
    _ = model.set_params(params)
    params.init[NDArray[DType.float64]].unsafe_set(0, 20.0)
    assert_equal(model.init[NDArray[DType.float64]].unsafe_get(0), 10.0)
    var result: NDArray[DType.float64] = model.fit_transform(X)
    assert_equal(result.shape[0], 6)
    _ = model.set_params(n_components=3)
    with assert_raises(contains="init shape"):
        _ = model.fit(X)
    assert_equal(model.embedding_.value().unsafe_get(0), result.unsafe_get(0))


def fit_validation() raises:
    var X = points[DType.float64]()
    var model = TSNE()
    assert_equal(model.method, "barnes_hut")
    with assert_raises(contains="perplexity"):
        _ = model.fit(X)
    _ = model.set_params(perplexity=2.0, n_components=4)
    with assert_raises(contains="n_components must be at most 3"):
        _ = model.fit(X)
    _ = model.set_params(n_components=2, max_iter=251)
    assert_true(isfinite(model.fit(X).kl_divergence_))
    _ = model.set_params(method="exact")
    _ = model.set_params(perplexity=2.0, max_iter=249)
    with assert_raises(contains="max_iter"):
        _ = model.fit(X)
    _ = model.set_params(max_iter=260, metric="precomputed")
    with assert_raises(contains='init="pca"'):
        _ = model.fit(X)
    model.init = "random"
    with assert_raises(contains="square pairwise matrix"):
        _ = model.fit(X)
    var distances = zeros[DType.float64](Shape(6, 6))
    distances.unsafe_set(1, -1)
    with assert_raises(contains="nonnegative"):
        _ = model.fit(distances)
    for i in range(6):
        for j in range(6):
            distances.unsafe_set(i * 6 + j, Float64(abs(i - j)))
    distances.unsafe_set(1, 2)
    _ = model.fit(distances)
    assert_equal(model.n_features_in_, 6)
    assert_equal(distances.unsafe_get(1), 2)
    _ = model.set_params(metric="unknown")
    with assert_raises(contains="unsupported dense TSNE metric"):
        _ = model.fit(X)
    _ = model.set_params(metric="euclidean")
    X.unsafe_set(0, nan[DType.float64]())
    with assert_raises(contains="finite"):
        _ = model.fit(X)


def barnes_hut_bounds() raises:
    var X = points[DType.float64]()
    var init = zeros[DType.float64](Shape(6, 2))
    for i in range(6):
        # Adjacent representable Float32 values near 1024 are farther apart
        # than the duplicate threshold and must split into separate leaves.
        init.unsafe_set(i * 2, 1024 + Float64(i) * 0.0001220703125)
    var model = TSNE[DType.float64, DType.float64](
        init=init.copy(), perplexity=2, max_iter=251, learning_rate=0.5, angle=0.0,
    )
    var embedding: NDArray[DType.float64] = model.fit_transform(X)
    assert_true(isfinite(model.kl_divergence_))
    assert_equal(embedding.shape[0], 6)
    assert_true(embedding.unsafe_get(0) != embedding.unsafe_get(2))

    _ = model.set_params(angle=1.1)
    with assert_raises(contains="angle must be in"):
        _ = model.fit(X)
    assert_equal(model.embedding_.value().unsafe_get(0), embedding.unsafe_get(0))
    init.unsafe_set(0, 1e50)
    var params = model.get_params()
    params.angle = 0.5
    params.init = init^
    _ = model.set_params(params)
    with assert_raises(contains="coordinates must fit in Float32"):
        _ = model.fit(X)


def global_stream() raises:
    var X = points[DType.float64]()
    var seeded = TSNE(method="exact", init="random", perplexity=2, random_state=17,
                      max_iter=260, learning_rate=2.0)
    var default_stream = TSNE(method="exact", init="random", perplexity=2,
                              max_iter=260, learning_rate=2.0)
    seed_global_random_state(17)
    var expected = seeded.fit_transform(X)
    var first = default_stream.fit_transform(X)
    var second = default_stream.fit_transform(X)
    for i in range(first.size):
        assert_equal(first.unsafe_get(i), expected.unsafe_get(i))
    assert_true(first.unsafe_get(0) != second.unsafe_get(0))
    default_stream.set_output(transform="default")
    with assert_raises(contains="native array output"):
        default_stream.set_output(transform="pandas")
    with assert_raises(contains="metadata routing"):
        default_stream.get_metadata_routing()


def first_feature_distance(
    X: NDArray[DType.float64], i: Int, Y: NDArray[DType.float64], j: Int,
) raises -> Float64:
    return abs(at(X, i, 0) - at(Y, j, 0))


def invalid_distance(
    X: NDArray[DType.float64], i: Int, Y: NDArray[DType.float64], j: Int,
) raises -> Float64:
    return -1


def callable_metric_contract(method: String) raises:
    var X = points[DType.float64]()
    var view = X.view_with_layout(Shape(6, 1), NDArrayStrides(3, 1), 0)
    var model = TSNE(
        method=method, metric="callable", init="random", perplexity=2,
        random_state=17, max_iter=260, learning_rate=2.0,
    )
    with assert_raises(contains="requires metric_params.distance"):
        _ = model.fit(X)
    var params = model.get_params()
    params.metric_params = TSNEMetricParams[DType.float64](distance=first_feature_distance)
    var expected = model.set_params(params).fit_transform(X)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    var viewed = fresh.fit_transform(view)
    for i in range(expected.size):
        assert_equal(viewed.unsafe_get(i), expected.unsafe_get(i))
    _ = model.set_params(metric="euclidean")
    with assert_raises(contains="metric_params do not match"):
        _ = model.fit(X)
    params.metric_params = TSNEMetricParams[DType.float64](distance=invalid_distance)
    _ = model.set_params(params)
    with assert_raises(contains="finite nonnegative distance"):
        _ = model.fit(X)
    assert_equal(model.embedding_.value().unsafe_get(0), expected.unsafe_get(0))


def main() raises:
    for method in ["exact", "barnes_hut"]:
        repeat_and_clone[DType.float64](method)
        repeat_and_clone[DType.float32](method)
        callable_metric_contract(method)
    explicit_configuration()
    fit_validation()
    barnes_hut_bounds()
    global_stream()
    print("PASS: native exact and Barnes–Hut TSNE contracts")
