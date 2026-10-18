"""Public imports, fluent contracts, storage and shared stream integration."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone, UnsupervisedEstimator, TransformerMixin
from moxie.random_projection import (
    GaussianRandomProjection, SparseRandomProjection, johnson_lindenstrauss_min_dim,
    ProjectionMatrix, GaussianRandomProjectionParams, SparseRandomProjectionParams,
    ComponentCount, ProjectionDensity, RandomProjector,
)
from moxie._random import RandomState, check_random_state, seed_global_random_state, entropy_state


def same[dtype: DType](a: ProjectionMatrix[dtype], b: ProjectionMatrix[dtype]) raises:
    assert_equal(a.rows, b.rows)
    assert_equal(a.cols, b.cols)
    var x = a.to_dense()
    var y = b.to_dense()
    for i in range(len(x.data)):
        assert_equal(x.data[i], y.data[i])


def chain[E: UnsupervisedEstimator & TransformerMixin](mut model: E, params: E.Params, X: E.Input) raises -> E.Transformed where conforms_to(E.Target, Deinitable):
    return model.set_params(params).fit(X).transform(X)


def exercise[dtype: DType]() raises:
    var values: List[Scalar[dtype]] = [1.0, 2.0, 0.0, 4.0, 0.0, 3.0, 2.0, 0.0, 4.0, 0.0, 1.0, 2.0]
    var X = ProjectionMatrix[dtype](3, 4, values^)
    var csr = X.to_csr()
    var gaussian = GaussianRandomProjection[dtype](2, random_state=42)
    var Y = chain(gaussian, gaussian.get_params(), X)
    same(Y, gaussian.transform(csr))
    var inverse = gaussian.inverse_transform(Y)
    assert_equal(inverse.cols, 4)
    assert_false(inverse.is_sparse)
    _ = gaussian.set_params(compute_inverse_components=True).fit(csr)
    same(inverse, gaussian.inverse_transform(Y))
    var sparse = SparseRandomProjection[dtype](2, density=0.5, random_state=42)
    var S = chain(sparse, sparse.get_params(), csr)
    assert_true(S.is_sparse)
    same(S, sparse.transform(X))
    var duplicate_values: List[Scalar[dtype]] = [1, 2, 3]
    var duplicate = ProjectionMatrix[dtype].from_csr(
        1, 4, duplicate_values^, List[Int]([1, 1, 3]), List[Int]([0, 3])
    )
    same(gaussian.transform(duplicate), gaussian.transform(duplicate.to_dense()))
    same(sparse.transform(duplicate), sparse.transform(duplicate.to_dense()))
    _ = sparse.set_params(dense_output=True)
    assert_false(sparse.transform(csr).is_sparse)
    var reconstruction = sparse.inverse_transform(S)
    assert_false(reconstruction.is_sparse)
    assert_equal(reconstruction.rows, 3)
    assert_equal(reconstruction.cols, 4)
    var before = sparse.components_.copy()
    _ = sparse.set_params(compute_inverse_components=True).fit(X)
    same(reconstruction, sparse.inverse_transform(S))
    with assert_raises():
        _ = sparse.fit(ProjectionMatrix[dtype](0, 4))
    same(before, sparse.components_)
    assert_true(Bool(sparse.inverse_components_))
    var cache_copy = sparse.copy()
    cache_copy.inverse_components_.value().data[0] += 10
    same(reconstruction, sparse.inverse_transform(S))
    _ = sparse.set_params(compute_inverse_components=False).fit(X)
    assert_false(Bool(sparse.inverse_components_))
    var copied = sparse.copy()
    copied.components_.data[0] += 10
    same(sparse.components_, before)
    var cloned = clone(sparse)
    assert_false(cloned.__sklearn_is_fitted__())
    assert_equal(sparse.get_feature_names_out(), List[String](["sparserandomprojection0", "sparserandomprojection1"]))
    var gp: GaussianRandomProjectionParams = gaussian.get_params()
    var sp: SparseRandomProjectionParams = sparse.get_params()
    assert_equal(gp.n_components[Int], 2)
    assert_equal(sp.density[Float64], 0.5)


def streams() raises:
    var X = ProjectionMatrix(2, 7)
    var expected = RandomState(42)
    var advancing = RandomState(42)
    var g = GaussianRandomProjection(3)
    var s = SparseRandomProjection(3, density=0.5)
    seed_global_random_state(42)
    for _ in range(2):
        _ = g.fit(X)
        var manual = GaussianRandomProjection(3)
        _ = manual.fit(X, rng=expected)
        same(g.components_, manual.components_)
        _ = s.fit(X)
        var manual_sparse = SparseRandomProjection(3, density=0.5)
        _ = manual_sparse.fit(X, rng=expected)
        same(s.components_, manual_sparse.components_)
    with check_random_state() as state:
        assert_equal(state[].next_uint32(), expected.next_uint32())
    var before = advancing.copy()
    with assert_raises():
        _ = s.fit(ProjectionMatrix(0, 7), rng=advancing)
    assert_equal(advancing.next_uint32(), before.next_uint32())
    _ = s.set_params(density=ProjectionDensity(0.0))
    before = advancing.copy()
    with assert_raises():
        _ = s.fit(X, rng=advancing)
    assert_equal(advancing.next_uint32(), before.next_uint32())
    _ = s.set_params(density=ProjectionDensity(0.5), random_state=42)
    with assert_raises():
        _ = s.fit(X, rng=advancing)
    _ = s.fit(X)
    var seeded = s.components_.copy()
    _ = s.fit(X)
    same(seeded, s.components_)
    var independent = entropy_state()
    _ = g.fit(X, rng=independent)
    assert_equal(g.components_.rows, 3)


def main() raises:
    assert_equal(johnson_lindenstrauss_min_dim(1e6, eps=0.5), 663)
    exercise[DType.float32]()
    exercise[DType.float64]()
    streams()
    print("PASS: public projection imports, dtypes, storage, fluent calls and shared streams")
