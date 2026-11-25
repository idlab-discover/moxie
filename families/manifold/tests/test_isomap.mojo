from numojo import NDArrayStrides, Shape, zeros
from std.math import abs
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie import clone
from moxie.manifold import Isomap


def contracts[dtype: DType]() raises:
    var X = zeros[dtype](Shape(8, 2))
    for i in range(8):
        X.unsafe_set(i * 2, Scalar[dtype](i))
    var model = Isomap[dtype](n_neighbors=2)
    var params = model.get_params()
    assert_equal(params.path_method, "auto")
    assert_equal(params.metric, "minkowski")
    assert_equal(params.p, 2)
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(X)
    with assert_raises(contains="NotFittedError"):
        _ = model.reconstruction_error()
    with assert_raises(contains="NotFittedError"):
        _ = model.get_feature_names_out()
    with assert_raises(contains="AttributeError"):
        _ = model.get_n_features_in()
    with assert_raises(contains="native array output"):
        model.set_output(transform="pandas")
    model.set_output(transform="default")

    var reversed_rows = X.view_with_layout(Shape(8, 2), NDArrayStrides(-2, 1), 14)
    var Y = model.fit_transform(reversed_rows)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(len(model.fit_warnings), 0)
    assert_equal(model.get_n_features_in(), 2)
    assert_equal(Y.shape[0], 8)
    assert_equal(Y.shape[1], 2)
    assert_equal(model.nbrs_.value().n_samples_fit_, 8)
    assert_equal(model.kernel_pca_.value().n_features_in_, 8)
    for i in range(8):
        for j in range(8):
            assert_equal(
                model.dist_matrix_.value().unsafe_get(i * 8 + j), Scalar[dtype](abs(i - j))
            )
    var error: Scalar[dtype] = model.reconstruction_error()
    assert_true(error < 0.005)
    var input_names: List[String] = ["a", "b"]
    var names = model.get_feature_names_out(input_names^)
    assert_equal(names[0], "isomap0")
    assert_equal(names[1], "isomap1")
    with assert_raises(contains="input_features length"):
        var wrong_names: List[String] = ["a"]
        _ = model.get_feature_names_out(wrong_names^)

    var original = Y.unsafe_get(0)
    Y.unsafe_set(0, 100)
    assert_equal(model.embedding_.value().unsafe_get(0), original)
    var projected = model.transform(reversed_rows)
    X.unsafe_set(0, 100)
    var queries = zeros[dtype](Shape(8, 2))
    for i in range(8):
        queries.unsafe_set(i * 2, Scalar[dtype](7 - i))
    var replay = model.transform(queries)
    for i in range(replay.size):
        assert_equal(replay.unsafe_get(i), projected.unsafe_get(i))

    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_false(Bool(fresh.embedding_))
    assert_false(Bool(fresh.nbrs_))
    assert_false(Bool(fresh.kernel_pca_))
    assert_false(Bool(fresh.dist_matrix_))
    assert_equal(fresh.get_params().n_neighbors.value(), 2)
    _ = model.set_params(path_method="FW", n_components=1)
    var before_refit = model.transform(queries)
    assert_equal(before_refit.shape[1], 2)
    Y = model.fit_transform(queries)
    assert_equal(Y.shape[1], 1)
    for i in range(8):
        assert_true(abs(Y.unsafe_get(i) - projected.unsafe_get(i * 2)) < 1e-5)

    params = model.get_params()
    params.n_neighbors = None
    params.radius = 1.25
    _ = model.set_params(params)
    Y = model.fit_transform(queries)
    assert_equal(Y.shape[1], 1)
    var outlier = zeros[dtype](Shape(1, 2))
    outlier.unsafe_set(0, 100)
    with assert_raises(contains="no neighbors within radius"):
        _ = model.transform(outlier)
    with assert_raises(contains="query columns"):
        _ = model.transform(zeros[dtype](Shape(1, 3)))

    var invalid = Isomap[dtype](radius=1.0)
    with assert_raises(contains="Both n_neighbors and radius"):
        _ = invalid.fit(X)
    with assert_raises(contains="path_method"):
        _ = invalid.set_params(radius=Optional[Float64](None), path_method="unknown").fit(X)
    with assert_raises(contains="nonfinite"):
        _ = model.fit(queries / Scalar[dtype](0))
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.embedding_.value().shape[1], 1)

    var precomputed = Isomap[dtype](metric="precomputed", n_neighbors=2)
    with assert_raises(contains="square"):
        _ = precomputed.fit(X)
    var D = zeros[dtype](Shape(8, 8))
    D.unsafe_set(1, -1)
    with assert_raises(contains="nonnegative"):
        _ = precomputed.fit(D)


def warnings_lifecycle() raises:
    var X = zeros[DType.float64](Shape(4, 1))
    X.unsafe_set(1, 1)
    X.unsafe_set(2, 10)
    X.unsafe_set(3, 11)
    var model = Isomap(n_neighbors=1)
    _ = model.fit(X)
    assert_equal(len(model.fit_warnings), 1)
    assert_equal(model.fit_warnings[0].category, "UserWarning")
    var fresh = clone(model)
    assert_equal(len(fresh.fit_warnings), 0)
    with assert_raises(contains="nonfinite"):
        _ = model.fit(X / 0.0)
    assert_equal(len(model.fit_warnings), 1)
    var params = model.get_params()
    params.n_neighbors = 3
    _ = model.set_params(params).fit(X)
    assert_equal(len(model.fit_warnings), 0)


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    warnings_lifecycle()
    print("PASS: Isomap state, ownership, strides, validation, clone and refit for Float32/Float64")
