"""Properties from sklearn/tests/test_random_projection.py at 4358542a.

Ports test_random_projection_embedding_quality and
test_random_projection_numerical_consistency, and checks the shared
feature-metadata traits used by generic composition code.
"""
from std.testing import assert_equal, assert_true, assert_raises, TestSuite
from moxie._feature_metadata import FeatureNamesOut, InputFeatureCount
from moxie._random import RandomState
from moxie.random_projection import (
    ProjectionMatrix,
    GaussianRandomProjection,
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)


def squared_distances(m: ProjectionMatrix[DType.float64]) raises -> List[Float64]:
    var dense = m.to_dense()
    var out = List[Float64]()
    for i in range(dense.rows):
        for j in range(dense.rows):
            var total = Float64(0)
            for k in range(dense.cols):
                var d = dense.data[i * dense.cols + k] - dense.data[j * dense.cols + k]
                total += d * d
            out.append(total)
    return out^


def assert_jl_contract(
    original: List[Float64], projected: ProjectionMatrix[DType.float64], eps: Float64
) raises:
    var after = squared_distances(projected)
    var checked = 0
    for k in range(len(original)):
        if original[k] != 0:
            var ratio = after[k] / original[k]
            assert_true(ratio < 1 + eps, String("distance ratio too large: ", ratio))
            assert_true(1 - eps < ratio, String("distance ratio too small: ", ratio))
            checked += 1
    assert_equal(checked, 8 * 7)


def test_embedding_quality() raises:
    # Same shape and density as sklearn's make_sparse_random_data(8, 5000, 15000).
    var rng = RandomState(0)
    var X = ProjectionMatrix(8, 5000)
    for _ in range(15000):
        var row = rng.randint(8)
        var col = rng.randint(5000)
        X.data[row * 5000 + col] += rng.normal()
    var eps = 0.2
    var original = squared_distances(X)
    var gaussian = GaussianRandomProjection(eps=eps, random_state=0)
    var sparse = SparseRandomProjection(eps=eps, random_state=0)
    for csr in [False, True]:
        var data = X.to_csr() if csr else X.copy()
        assert_jl_contract(original, gaussian.fit_transform(data), eps)
        assert_jl_contract(original, sparse.fit_transform(data), eps)
        assert_equal(gaussian.n_components_, johnson_lindenstrauss_min_dim(8, eps=eps))
        assert_equal(sparse.n_components_, gaussian.n_components_)


def test_float32_float64_consistency() raises:
    # sklearn draws X = RandomState(42).rand(25, 3000) and requires atol=1e-5;
    # Float32 products of 3000 terms keep about 1e-6 absolute accuracy here.
    var rng = RandomState(42)
    var values64 = List[Float64]()
    var values32 = List[Float32]()
    for _ in range(25 * 3000):
        var value = rng.random_sample()
        values64.append(value)
        values32.append(Float32(value))
    var X64 = ProjectionMatrix(25, 3000, values64^)
    var X32 = ProjectionMatrix[DType.float32](25, 3000, values32^)
    var g32 = GaussianRandomProjection[DType.float32](random_state=0)
    var g64 = GaussianRandomProjection(random_state=0)
    compare(g32.fit_transform(X32), g64.fit_transform(X64), 1e-5)
    compare(g32.components_, g64.components_, 1e-7)
    var s32 = SparseRandomProjection[DType.float32](random_state=0)
    var s64 = SparseRandomProjection(random_state=0)
    compare(s32.fit_transform(X32), s64.fit_transform(X64), 1e-5)
    compare(s32.components_, s64.components_, 1e-7)
    assert_true(s32.components_.is_sparse)


def compare(
    a: ProjectionMatrix[DType.float32], b: ProjectionMatrix[DType.float64], atol: Float64
) raises:
    var x = a.to_dense()
    var y = b.to_dense()
    assert_equal(x.rows, y.rows)
    assert_equal(x.cols, y.cols)
    for k in range(len(y.data)):
        assert_true(abs(Float64(x.data[k]) - y.data[k]) <= atol)


def names_and_counts[E: FeatureNamesOut & InputFeatureCount](model: E) raises -> Int:
    return len(model.get_feature_names_out()) * 1000 + model.get_n_features_in()


def test_shared_feature_metadata_traits() raises:
    var X = ProjectionMatrix(3, 5, List[Float64]([
        1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0,
    ]))
    var gaussian = GaussianRandomProjection(2, random_state=0)
    var sparse = SparseRandomProjection[DType.float64](4, random_state=0)
    # sklearn's n_features_in_ attribute does not exist before fit.
    with assert_raises(contains="AttributeError"):
        _ = gaussian.get_n_features_in()
    with assert_raises(contains="AttributeError"):
        _ = sparse.get_n_features_in()
    with assert_raises():
        _ = names_and_counts(gaussian)
    _ = gaussian.fit(X)
    _ = sparse.fit(X.to_csr())
    assert_equal(names_and_counts(gaussian), 2005)
    assert_equal(names_and_counts(sparse), 4005)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
