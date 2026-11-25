from numojo import array, NDArrayStrides, Shape, zeros
from std.testing import assert_almost_equal, assert_equal, assert_false, assert_raises, assert_true
from moxie.base import clone
from moxie.manifold import MDS, NormalizedStress, smacof


def check[dtype: DType]() raises:
    var distances = array[dtype](
        [0, 5, 3, 4, 5, 0, 2, 2, 3, 2, 0, 1, 4, 2, 1, 0], [4, 4],
    )
    var initial = array[DType.float64](
        [-0.266, -0.539, 0.451, 0.252, 0.016, -0.238, -0.200, 0.524], [4, 2],
    )
    var result = smacof(distances, init=initial, max_iter=1)
    assert_false(Bool(result.n_iter))
    assert_almost_equal(result.embedding.unsafe_get(0), -1.414243185149859, atol=1e-12)
    assert_almost_equal(result.embedding.unsafe_get(7), 1.4313619523809293, atol=1e-12)

    var model = MDS[dtype](dissimilarity="precomputed", metric=False, n_init=3)
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises(contains="AttributeError: MDS has no n_features_in_"):
        _ = model.get_n_features_in()
    var embedded = model.fit_transform(distances, init=initial)
    assert_equal(embedded.shape[1], 2)
    assert_equal(model.n_features_in_, 4)
    assert_true(model.n_iter_ > 0)
    assert_equal(len(model.fit_warnings), 4)
    assert_equal(model.fit_warnings[0].category, "FutureWarning")
    assert_equal(model.fit_warnings[3].category, "UserWarning")
    var fitted_copy = model.copy()
    fitted_copy.fit_warnings.clear()
    fitted_copy.embedding_.value().unsafe_set(0, 123)
    assert_equal(len(model.fit_warnings), 4)
    assert_equal(model.embedding_.value().unsafe_get(0), embedded.unsafe_get(0))
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_false(Bool(fresh.embedding_))
    assert_equal(len(fresh.fit_warnings), 0)
    assert_equal(fresh.get_params().n_init, 3)

    _ = model.set_params(metric="precomputed", dissimilarity="deprecated",
                         init="random", n_init=1, max_iter=3, random_state=17,
                         normalized_stress=NormalizedStress(False))
    var first = model.fit_transform(distances)
    var second = model.fit_transform(distances)
    assert_equal(model.n_iter_, 3)
    assert_equal(len(model.fit_warnings), 0)
    assert_equal(first.unsafe_get(0), second.unsafe_get(0))
    second.unsafe_set(0, 1234)
    assert_equal(model.embedding_.value().unsafe_get(0), first.unsafe_get(0))
    assert_equal(model.dissimilarity_matrix_.value().unsafe_get(1), 5)
    distances.unsafe_set(1, 200)
    assert_equal(model.dissimilarity_matrix_.value().unsafe_get(1), 5)
    with assert_raises(contains="symmetric"):
        _ = model.fit(distances)
    assert_equal(model.embedding_.value().unsafe_get(0), first.unsafe_get(0))

    var contiguous = array[dtype]([0, 2, 2, 0], [2, 2])
    var padded = array[dtype]([10, 0, 2, 20, 2, 0], [2, 3])
    var view = padded.view_with_layout(Shape(2, 2), NDArrayStrides(3, 1), 1)
    var backing = array[dtype]([90, 0, 2, 2, 0], [1, 5])
    var contiguous_view = backing.view_with_layout(Shape(2, 2), NDArrayStrides(2, 1), 1)
    var same = model.fit_transform(contiguous)
    var strided = model.fit_transform(view)
    var offset = model.fit_transform(contiguous_view)
    assert_equal(model.get_n_features_in(), 2)
    for i in range(same.size):
        assert_equal(same.unsafe_get(i), strided.unsafe_get(i))
        assert_equal(same.unsafe_get(i), offset.unsafe_get(i))
    var params = model.get_params()
    params.random_state = None
    _ = model.set_params(params)
    assert_false(Bool(model.get_params().random_state))

    var wrong_shape = zeros[dtype](Shape(2, 3))
    with assert_raises(contains="square"):
        _ = smacof(wrong_shape)
    with assert_raises(contains="init matrix"):
        _ = smacof(zeros[dtype](Shape(2, 2)), init=initial)
    with assert_raises(contains="disparity normalization"):
        _ = smacof(zeros[dtype](Shape(2, 2)), metric=False)
    with assert_raises(contains="n_jobs"):
        _ = smacof(zeros[dtype](Shape(2, 2)), n_jobs=0)
    with assert_raises(contains="normalized_stress"):
        _ = smacof(zeros[dtype](Shape(2, 2)), normalized_stress="invalid")
    with assert_raises(contains="provided both"):
        _ = model.set_params(metric="cosine", dissimilarity="precomputed").fit(distances)


def check_inferred_metrics[dtype: DType]() raises:
    var contiguous = array[dtype]([0, 0, 2, 0, 0, 1, 3, 3], [4, 2])
    var backing = array[dtype]([90, 0, 0, 91, 2, 0, 92, 0, 1, 93, 3, 3], [4, 3])
    var strided = backing.view_with_layout(Shape(4, 2), NDArrayStrides(3, 1), 1)
    for metric in ["seuclidean", "mahalanobis"]:
        var model = MDS[dtype](metric=metric, init="random", max_iter=2, random_state=0)
        var expected = model.fit_transform(contiguous)
        var actual = model.fit_transform(strided)
        for i in range(expected.size):
            assert_equal(actual.unsafe_get(i), expected.unsafe_get(i))


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    check_inferred_metrics[DType.float32]()
    check_inferred_metrics[DType.float64]()
    print("PASS: MDS and smacof dense contracts")
