"""LatentDirichletAllocation lifecycle, ownership, invalid settings and dtype contracts."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.math import abs
from std.utils.numerics import inf
from numojo import NDArray, Shape, zeros, NDArrayStrides
from moxie.base import clone
from moxie.decomposition import LatentDirichletAllocation


def counts[dtype: DType](n: Int, p: Int) raises -> NDArray[dtype]:
    var x = zeros[dtype](Shape(n, p))
    for i in range(n):
        for j in range(p): x.unsafe_set(i * p + j, Scalar[dtype](Float64((i * 3 + j * j) % 5) if (i + j) % 3 != 0 else 0.0))
    return x^


def rejects[dtype: DType](var model: LatentDirichletAllocation[dtype], x: NDArray[dtype]) raises:
    with assert_raises(): model.fit(x)


def check[dtype: DType]() raises:
    var x = counts[dtype](10, 8)
    var model = LatentDirichletAllocation[dtype](3, random_state=0, max_iter=5)
    with assert_raises(): _ = model.transform(x)
    with assert_raises(): _ = model.score(x)
    with assert_raises(): _ = model.perplexity(x)
    var z = model.fit_transform(x)
    assert_equal(z.shape[0], 10)
    assert_equal(z.shape[1], 3)
    for i in range(10):
        var total: Float64 = 0
        for t in range(3): total += Float64(z.unsafe_get(i * 3 + t))
        assert_true(abs(total - 1) < 1e-4)
    assert_equal(model.n_iter_, 5)
    assert_equal(model.n_batch_iter_, 6)
    assert_true(abs(model.doc_topic_prior_ - 1.0 / 3.0) < 1e-12)
    assert_equal(model.get_feature_names_out()[2], "latentdirichletallocation2")
    assert_true(model.perplexity(x) > 0)
    var first = model.components_.value().unsafe_get(0)
    model.fit(x)
    assert_equal(model.components_.value().unsafe_get(0), first)
    x.unsafe_set(1, 40)
    assert_equal(model.components_.value().unsafe_get(0), first)
    x = counts[dtype](10, 8)
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    copy.set_params(learning_method="online", batch_size=4)
    copy.fit(x.view_with_layout(Shape(10, 8), NDArrayStrides(-8, 1), 72))
    var online = LatentDirichletAllocation[dtype](2, random_state=1)
    online.partial_fit(x)
    online.partial_fit(x)
    assert_equal(online.n_batch_iter_, 3)
    with assert_raises(): online.partial_fit(zeros[dtype](Shape(2, 5)))
    with assert_raises(): _ = online.transform(zeros[dtype](Shape(2, 5)))
    var negative = x.copy()
    negative.unsafe_set(0, -1)
    rejects(LatentDirichletAllocation[dtype](2), negative)
    var nonfinite = x.copy()
    nonfinite.unsafe_set(0, inf[dtype]())
    rejects(LatentDirichletAllocation[dtype](2), nonfinite)
    rejects(LatentDirichletAllocation[dtype](2), zeros[dtype](Shape(0, 8)))
    rejects(LatentDirichletAllocation[dtype](2, n_jobs=2), x)
    rejects(LatentDirichletAllocation[dtype](2, n_jobs=0), x)
    var sequential = LatentDirichletAllocation[dtype](2, n_jobs=1, random_state=0)
    sequential.fit(x)
    var params = sequential.get_params()
    params.n_components = 4
    sequential.set_params(params)
    sequential.fit(counts[dtype](5, 3))
    assert_equal(sequential.components_.value().shape[0], 4)
    assert_equal(sequential.n_features_in_, 3)
    rejects(LatentDirichletAllocation[dtype](0), x)
    rejects(LatentDirichletAllocation[dtype](2, learning_method="gibbs"), x)
    rejects(LatentDirichletAllocation[dtype](2, doc_topic_prior=Optional[Float64](2.0)), x)
    rejects(LatentDirichletAllocation[dtype](2, learning_offset=0.5), x)
    rejects(LatentDirichletAllocation[dtype](2, batch_size=0), x)
    rejects(LatentDirichletAllocation[dtype](2, max_iter=-1), x)
    rejects(LatentDirichletAllocation[dtype](2, total_samples=0.0), x)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: LatentDirichletAllocation contracts in Float32 and Float64")
