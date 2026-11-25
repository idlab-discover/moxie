"""Random projections inside moxie.pipeline on the dense NDArray trait path.

sklearn's canonical use is make_pipeline(SparseRandomProjection(), clf). The
pipeline and union results must equal the explicit two-stage computation, keep
the selected dtype and shapes, and survive clone/refit and held-out data.
"""
from numojo import NDArray, Shape, zeros
from std.testing import assert_equal, assert_false, assert_raises
from moxie.base import clone
from moxie.linear_model import RidgeClassifier
from moxie.pipeline import make_pipeline, make_union
from moxie.random_projection import GaussianRandomProjection, SparseRandomProjection


def data[dtype: DType](rows: Int, offset: Int) raises -> NDArray[dtype]:
    var x = zeros[dtype](Shape(rows, 40))
    for i in range(rows):
        for j in range(40):
            var k = i + offset
            x.unsafe_set(i * 40 + j, Scalar[dtype](Float64((k * 13 + j * 7) % 17) / 8 + Float64(k % 2) * Float64(j % 3)))
    return x^


def labels(rows: Int, offset: Int) raises -> NDArray[DType.int64]:
    var y = zeros[DType.int64](Shape(rows))
    for i in range(rows):
        y.unsafe_set(i, Int64((i + offset) % 2))
    return y^


def same[dtype: DType](a: NDArray[dtype], b: NDArray[dtype]) raises:
    assert_equal(a.ndim, b.ndim)
    for d in range(a.ndim):
        assert_equal(a.shape[d], b.shape[d])
    for k in range(a.size):
        assert_equal(a.unsafe_get(k), b.unsafe_get(k))


def projection_classifier[dtype: DType]() raises:
    var x = data[dtype](30, 0)
    var y = labels(30, 0)
    var held_out = data[dtype](9, 30)
    var pipe = make_pipeline(SparseRandomProjection[dtype](n_components=12, random_state=0), RidgeClassifier[dtype]())
    assert_equal(pipe.transformer_name, "sparserandomprojection")
    with assert_raises():
        _ = pipe.n_features_in_()
    _ = pipe.fit(x, y)
    assert_equal(pipe.n_features_in_(), x.shape[1])
    var projection = SparseRandomProjection[dtype](n_components=12, random_state=0)
    var features: NDArray[dtype] = projection.fit_transform(x)
    assert_equal(features.shape[0], 30)
    assert_equal(features.shape[1], 12)
    var model = RidgeClassifier[dtype]()
    _ = model.fit(features, y)
    same(pipe.predict(x), model.predict(features))
    same(pipe.predict(held_out), model.predict(projection.transform(held_out)))
    assert_equal(pipe.score(x, y), model.score(features, y))
    var fresh = clone(pipe)
    assert_false(fresh.transformer.__sklearn_is_fitted__())
    _ = fresh.fit(x, y)
    same(fresh.predict(held_out), pipe.predict(held_out))
    var params = pipe.get_params()
    params.transformer.n_components = 6
    _ = pipe.set_params(params).fit(x, y)
    assert_equal(pipe.transformer.n_components_, 6)
    assert_equal(fresh.transformer.n_components_, 12)
    _ = projection.set_params(params.transformer)
    _ = model.fit(projection.fit_transform(x), y)
    same(pipe.predict(held_out), model.predict(projection.transform(held_out)))
    var gaussian = make_pipeline(GaussianRandomProjection[dtype](n_components=8, random_state=1), RidgeClassifier[dtype]())
    _ = gaussian.fit(x, y)
    var dense = GaussianRandomProjection[dtype](n_components=8, random_state=1)
    var reference = RidgeClassifier[dtype]()
    _ = reference.fit(dense.fit_transform(x), y)
    same(gaussian.predict(held_out), reference.predict(dense.transform(held_out)))


def projection_union[dtype: DType]() raises:
    var x = data[dtype](30, 0)
    var held_out = data[dtype](9, 30)
    var union = make_union[dtype=dtype](
        GaussianRandomProjection[dtype](n_components=5, random_state=2),
        SparseRandomProjection[dtype](n_components=4, density=0.3, random_state=3),
    )
    var combined: NDArray[dtype] = union.fit_transform(x)
    assert_equal(combined.shape[0], 30)
    assert_equal(combined.shape[1], 9)
    var g = GaussianRandomProjection[dtype](n_components=5, random_state=2)
    var s = SparseRandomProjection[dtype](n_components=4, density=0.3, random_state=3)
    _ = g.fit(x)
    _ = s.fit(x)
    var out = union.transform(held_out)
    var left = g.transform(held_out)
    var right = s.transform(held_out)
    assert_equal(out.shape[1], 9)
    for i in range(9):
        for j in range(5):
            assert_equal(out.unsafe_get(i * 9 + j), left.unsafe_get(i * 5 + j))
        for j in range(4):
            assert_equal(out.unsafe_get(i * 9 + 5 + j), right.unsafe_get(i * 4 + j))
    var names = union.get_feature_names_out()
    assert_equal(len(names), 9)
    assert_equal(names[0], "gaussianrandomprojection__gaussianrandomprojection0")
    assert_equal(names[5], "sparserandomprojection__sparserandomprojection0")
    var fresh = clone(union)
    same(fresh.fit_transform(x), combined)


def main() raises:
    projection_classifier[DType.float32]()
    projection_classifier[DType.float64]()
    projection_union[DType.float32]()
    projection_union[DType.float64]()
    print("PASS: random projections in Pipeline and FeatureUnion: dtypes, shapes, held-out, clone, refit")
