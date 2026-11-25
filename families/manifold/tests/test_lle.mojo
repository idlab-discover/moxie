from numojo import array, NDArray, NDArrayStrides, Shape, zeros
from std.math import nan
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from moxie import clone
from moxie.manifold import LocallyLinearEmbedding, LocallyLinearEmbeddingResult, locally_linear_embedding


def compare(first: NDArray[DType.float64], second: NDArray[DType.float64]) raises:
    assert_equal(first.shape[0], second.shape[0])
    assert_equal(first.shape[1], second.shape[1])
    for i in range(first.size):
        assert_equal(first.unsafe_get(i), second.unsafe_get(i))


def check[dtype: DType]() raises:
    var X = array[dtype]([1, 1, 2, 3, 5, 2, 6, 1, 8, 3, 9, 5, 12, 4, 15, 3], [8, 2])
    var model = LocallyLinearEmbedding[dtype]()
    var defaults = model.get_params()
    assert_equal(defaults.n_neighbors, 5)
    assert_equal(defaults.n_components, 2)
    assert_equal(defaults.eigen_solver, "auto")
    assert_equal(defaults.method, "standard")
    assert_equal(defaults.reg, 0.001)
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = model.transform(X)
    with assert_raises(contains="AttributeError"):
        _ = model.get_n_features_in()
    with assert_raises(contains="NotFittedError"):
        _ = model.get_feature_names_out()
    with assert_raises(contains="native array output"):
        model.set_output(transform="pandas")
    model.set_output(transform="default")

    var first: NDArray[DType.float64] = model.fit_transform(X)
    assert_equal(model.n_features_in_, 2)
    assert_equal(model.nbrs_.value().n_samples_fit_, 8)
    assert_equal(first.shape[0], 8)
    assert_equal(first.shape[1], 2)
    assert_equal(model.get_feature_names_out()[1], "locallylinearembedding1")
    with assert_raises(contains="input_features length"):
        var wrong_names: List[String] = ["x"]
        _ = model.get_feature_names_out(wrong_names^)
    compare(first, model.fit_transform(X))

    var fresh = clone(model)
    assert_false(Bool(fresh.embedding_))
    assert_false(Bool(fresh.nbrs_))
    assert_equal(fresh.get_params().n_neighbors, 5)
    first.unsafe_set(0, 100)
    assert_true(model.embedding_.value().unsafe_get(0) != 100)

    var before = model.transform(X)
    var query = X.copy()
    X.unsafe_set(0, -100)
    compare(before, model.transform(query))

    var reverse = query.view_with_layout(Shape(8, 2), NDArrayStrides(-2, 1), 14)
    var compact = reverse.contiguous()
    var strided = model.fit_transform(reverse)
    compare(strided, model.fit_transform(compact))
    compare(model.transform(reverse), model.transform(compact))
    var returned: LocallyLinearEmbeddingResult = locally_linear_embedding(
        reverse, n_neighbors=5, n_components=2, eigen_solver="dense",
    )
    compare(returned.embedding, strided)
    assert_equal(returned.reconstruction_error, model.reconstruction_error_)

    with assert_raises(contains="output dimension"):
        _ = model.set_params(n_components=3).fit(query)
    assert_equal(model.embedding_.value().shape[1], 2)
    assert_equal(model.transform(query).shape[1], 2)
    _ = model.set_params(n_components=1).fit(query)
    assert_equal(model.embedding_.value().shape[1], 1)
    assert_equal(len(model.get_feature_names_out()), 1)
    with assert_raises(contains="query columns"):
        _ = model.transform(zeros[dtype](Shape(1, 3)))
    with assert_raises(contains="expected rank two"):
        _ = model.fit(zeros[dtype](Shape(2)))
    with assert_raises(contains="expected nonempty matrix"):
        _ = model.fit(zeros[dtype](Shape(0, 2)))
    query.unsafe_set(0, nan[dtype]())
    with assert_raises(contains="expected finite entries"):
        _ = model.fit(query)
    assert_equal(model.embedding_.value().shape[1], 1)

    var invalid = LocallyLinearEmbedding[dtype](n_neighbors=8)
    with assert_raises(contains="n_neighbors < n_samples"):
        _ = invalid.fit(X)
    with assert_raises(contains="Hessian LLE requires"):
        _ = invalid.set_params(n_neighbors=5, method="hessian").fit(X)
    with assert_raises(contains="n_neighbors >= n_components"):
        _ = invalid.set_params(n_neighbors=1, method="modified").fit(X)
    with assert_raises(contains="eigen_solver='arpack'"):
        _ = invalid.set_params(eigen_solver="arpack").fit(X)
    with assert_raises(contains="nonnegative"):
        _ = invalid.set_params(reg=-1.0).fit(X)
    with assert_raises(contains="random_state must be"):
        _ = invalid.set_params(reg=0.001, random_state=Optional[Int](-1)).fit(X)
    var changed = invalid.get_params()
    changed.random_state = None
    changed.n_jobs = 0
    changed.eigen_solver = "auto"
    changed.method = "standard"
    _ = invalid.set_params(changed)
    with assert_raises(contains="n_jobs cannot be zero"):
        _ = invalid.fit(X)

    var duplicates = zeros[dtype](Shape(4, 1))
    var duplicate_fit = model.set_params(n_neighbors=2).fit_transform(duplicates)
    assert_true(duplicate_fit.shape[0] == 4)
    assert_equal(model.n_features_in_, 1)
    with assert_raises(contains="covariance factorization failed"):
        _ = model.set_params(reg=0.0).fit(duplicates)
    assert_equal(model.n_features_in_, 1)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: dense LLE estimator and function contracts")
