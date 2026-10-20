from numojo import Shape, zeros
from numojo import NDArrayStrides
from std.testing import assert_equal, assert_raises, assert_true
from std.math import abs
from moxie.decomposition import IncrementalPCA
from moxie_numax import NumaxIncrementalPCABackend


def check[dtype: DType]() raises:
    var model = IncrementalPCA[dtype, NumaxIncrementalPCABackend[4, 3, 2]](2, batch_size=4)
    var x = zeros[dtype](Shape(9, 3))
    for i in range(9):
        for j in range(3): x.unsafe_set(i * 3 + j, Scalar[dtype](i * i + (i + 2) % (j + 2)))
    with assert_raises(): _ = model.transform(x)
    var z = model.fit_transform(x)
    assert_equal(model.n_samples_seen_, 9)
    assert_equal(model.n_components_, 2)
    assert_equal(z.shape[0], 9)
    _ = model.inverse_transform(z)
    _ = model.get_covariance()
    _ = model.get_precision()
    var cloned = model.__sklearn_clone__()
    assert_true(not cloned.__sklearn_is_fitted__())
    var single = zeros[dtype](Shape(1, 3))
    model.partial_fit(single, check_input=False)
    assert_equal(model.n_samples_seen_, 10)
    model.fit(x)
    assert_equal(model.n_samples_seen_, 9)
    with assert_raises(): model.partial_fit(zeros[dtype](Shape(2, 4)))
    assert_equal(model.n_samples_seen_, 9)
    var params = model.get_params()
    params.n_components = 1
    model.set_params(params)
    with assert_raises(): model.partial_fit(single)
    params.n_components = 2
    model.set_params(params)
    model.partial_fit(single)
    assert_equal(model.n_samples_seen_, 10)
    var inplace = model.__sklearn_clone__()
    params.copy_input = False
    inplace.set_params(params)
    with assert_raises(): inplace.fit(x)
    inplace.fit_inplace(x)
    assert_equal(inplace.n_samples_seen_, 9)
    var names = model.get_feature_names_out()
    assert_equal(names[0], "incrementalpca0")
    model.set_output(transform="default")


def edges() raises:
    var model = IncrementalPCA[DType.float64, NumaxIncrementalPCABackend[4, 3, 2]](2, batch_size=4)
    var x = zeros[DType.float64](Shape(8, 3))
    for i in range(8):
        for j in range(3): x.unsafe_set(i * 3 + j, Float64(i * i + (i + j) % 3))
    var original = x.copy()
    model.fit(x)
    for i in range(x.size): assert_equal(x.unsafe_get(i), original.unsafe_get(i))
    var total = model.n_samples_seen_
    with assert_raises(): model.partial_fit(zeros[DType.float64](Shape(0, 3)))
    with assert_raises(): model.partial_fit(zeros[DType.float64](Shape(20, 3)))
    with assert_raises(): _ = model.inverse_transform(zeros[DType.float64](Shape(3, 1)))
    assert_equal(model.n_samples_seen_, total)
    var invalid = IncrementalPCA(n_components=0)
    with assert_raises(): invalid.fit(x)
    var invalid_batch = IncrementalPCA(batch_size=0)
    with assert_raises(): invalid_batch.fit(x)
    var first = model.__sklearn_clone__()
    with assert_raises(): first.partial_fit(zeros[DType.float64](Shape(1, 3)))
    var strided = x.view_with_layout(Shape(8, 3), NDArrayStrides(-3, 1), 21)
    var streaming = model.__sklearn_clone__()
    streaming.partial_fit(strided)
    for j in range(3):
        assert_true(abs(streaming.mean_.value().unsafe_get(j) - model.mean_.value().unsafe_get(j)) < 1e-12)
        assert_true(abs(streaming.var_.value().unsafe_get(j) - model.var_.value().unsafe_get(j)) < 1e-10)
    var before = streaming.mean_.value().unsafe_get(0)
    x.unsafe_set(0, 999)
    assert_equal(streaming.mean_.value().unsafe_get(0), before)
    var inplace = model.__sklearn_clone__()
    inplace.set_params(copy=False)
    var centered = original.copy()
    inplace.partial_fit_inplace(centered)
    for j in range(3):
        var sum: Float64 = 0
        for i in range(8): sum += centered.unsafe_get(i * 3 + j)
        assert_true(abs(sum) < 1e-10)
    var default_rank = IncrementalPCA()
    default_rank.partial_fit(original)
    assert_equal(default_rank.n_components_, 3)
    assert_equal(default_rank.n_samples_seen_, 8)
    default_rank.partial_fit(zeros[DType.float64](Shape(1, 3)))
    assert_equal(default_rank.n_samples_seen_, 9)


def main() raises:
    edges()
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: IncrementalPCA lifecycle contracts")
