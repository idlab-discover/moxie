"""Native state, validation, storage, and numerical contracts."""
from std.math import isfinite
from std.testing import (
    assert_equal,
    assert_false,
    assert_true,
    assert_raises,
    TestSuite,
)
from moxie.base import clone
from moxie.random_projection import (
    ProjectionMatrix,
    RandomProjector,
    GaussianRandomProjection,
    SparseRandomProjection,
    ComponentCount,
    ProjectionDensity,
    johnson_lindenstrauss_min_dim,
    _gaussian_random_matrix,
    _sparse_random_matrix,
)
from moxie._projection.linalg import pseudoinverse
from moxie._projection.matrix import project


def close(
    actual: ProjectionMatrix[DType.float64],
    expected: ProjectionMatrix[DType.float64],
    tolerance: Float64 = 1e-10,
) raises:
    assert_equal(actual.rows, expected.rows)
    assert_equal(actual.cols, expected.cols)
    var left = actual.to_dense()
    var right = expected.to_dense()
    for i in range(len(left.data)):
        assert_true(isfinite(left.data[i]))
        assert_true(
            abs(left.data[i] - right.data[i])
            <= tolerance * (1 + abs(right.data[i]))
        )


def input_data() raises -> ProjectionMatrix[DType.float64]:
    return ProjectionMatrix(
        3,
        4,
        List[Float64](
            [1.0, 2.0, 0.0, 4.0, 0.0, 3.0, 2.0, 0.0, 4.0, 0.0, 1.0, 2.0]
        ),
    )


def fit_generic[
    E: RandomProjector
](mut model: E, X: E.Input) raises -> E.Transformed where conforms_to(
    E.Target, Deinitable
):
    return model.fit_transform(X)


def test_gaussian_lifecycle() raises:
    var X = input_data()
    var model = GaussianRandomProjection(n_components=2, random_state=42)
    assert_false(model.__sklearn_is_fitted__())
    with assert_raises():
        _ = model.transform(X)
    with assert_raises():
        _ = model.inverse_transform(X)
    with assert_raises():
        _ = model.get_feature_names_out()
    var transformed = fit_generic(model, X)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.n_components_, 2)
    assert_equal(model.n_features_in_, 4)
    assert_false(model.components_.is_sparse)
    assert_false(transformed.is_sparse)
    var original = model.components_.copy()
    _ = model.fit(X, Optional[List[Float64]]([100.0]))  # y is ignored
    close(model.components_, original, 0)
    close(model.transform(X.to_csr()), transformed)
    close(model.transform(model.inverse_transform(transformed)), transformed)
    var cloned = clone(model)
    assert_false(cloned.__sklearn_is_fitted__())
    assert_equal(cloned.n_components[Int], 2)
    assert_equal(cloned.random_state.value(), 42)
    close(cloned.fit_transform(X), transformed, 0)
    var copied = model.copy()
    copied.components_.data[0] = 900
    close(model.components_, original, 0)
    var params = model.get_params()
    params.n_components = 3
    assert_equal(model.n_components[Int], 2)
    _ = model.set_params(params)
    close(model.transform(X), transformed)  # configuration doesn't erase fit
    _ = model.fit(X)
    assert_equal(model.n_components_, 3)
    _ = model.set_params(
        n_components=ComponentCount(2), compute_inverse_components=True
    )
    _ = model.fit(X)
    assert_true(Bool(model.inverse_components_))
    close(model.transform(model.inverse_transform(transformed)), transformed)
    _ = model.set_params(compute_inverse_components=False)
    _ = model.fit(X)
    assert_false(Bool(model.inverse_components_))
    _ = model.set_params(compute_inverse_components=True)
    with assert_raises():
        _ = model.inverse_transform(transformed)
    assert_equal(
        model.get_feature_names_out(),
        List[String](
            ["gaussianrandomprojection0", "gaussianrandomprojection1"]
        ),
    )
    with assert_raises():
        _ = model.get_feature_names_out(Optional[List[String]](["one"]))


def test_sparse_lifecycle_and_output() raises:
    var X = input_data()
    var model = SparseRandomProjection(
        n_components=2, density=0.5, random_state=42
    )
    with assert_raises():
        _ = model.transform(X)
    var dense = fit_generic(model, X)
    assert_true(model.components_.is_sparse)
    assert_false(dense.is_sparse)
    assert_equal(model.density_, 0.5)
    var sparse = model.transform(X.to_csr())
    assert_true(sparse.is_sparse)
    close(sparse, dense)
    close(model.transform(model.inverse_transform(sparse)), dense)
    _ = model.set_params(dense_output=True)
    assert_false(model.transform(X.to_csr()).is_sparse)
    close(model.transform(X.to_csr()), dense)
    var cloned = clone(model)
    assert_false(cloned.__sklearn_is_fitted__())
    close(cloned.fit_transform(X), dense, 0)
    _ = model.set_params(
        density=ProjectionDensity(1.0),
        dense_output=False,
        compute_inverse_components=True,
    )
    _ = model.fit(X.to_csr())
    assert_false(model.components_.is_sparse)
    assert_false(model.transform(X.to_csr()).is_sparse)
    assert_true(Bool(model.inverse_components_))
    var all_dense = model.transform(X)
    close(model.transform(model.inverse_transform(all_dense)), all_dense)
    _ = model.set_params(density=ProjectionDensity(String("auto")))
    _ = model.fit(X)
    assert_equal(model.density_, 0.5)
    assert_equal(
        model.get_feature_names_out(),
        List[String](["sparserandomprojection0", "sparserandomprojection1"]),
    )
    var params = model.get_params(False)
    params.random_state = None
    _ = model.set_params(params)
    assert_false(Bool(model.random_state))


def test_dimensions_and_jl() raises:
    assert_equal(johnson_lindenstrauss_min_dim(1e6, eps=0.5), 663)
    assert_equal(johnson_lindenstrauss_min_dim(100, eps=1e-5), 368416070986)
    assert_equal(johnson_lindenstrauss_min_dim(1), 0)
    assert_equal(
        johnson_lindenstrauss_min_dim(1e6, eps=List[Float64]([0.5, 0.1, 0.01])),
        List[Int]([663, 11841, 1112658]),
    )
    assert_equal(
        johnson_lindenstrauss_min_dim(List[Float64]([1e4, 1e5, 1e6])),
        List[Int]([7894, 9868, 11841]),
    )
    assert_equal(
        johnson_lindenstrauss_min_dim(
            List[Float64]([1e6]), eps=List[Float64]([0.5, 0.1])
        ),
        List[Int]([663, 11841]),
    )
    with assert_raises():
        _ = johnson_lindenstrauss_min_dim(100, eps=0.0)
    with assert_raises():
        _ = johnson_lindenstrauss_min_dim(100, eps=1.0)
    with assert_raises():
        _ = johnson_lindenstrauss_min_dim(0)
    with assert_raises():
        _ = johnson_lindenstrauss_min_dim(100, eps=1e-100)
    with assert_raises():
        _ = johnson_lindenstrauss_min_dim(
            List[Float64]([2.0, 3.0]), eps=List[Float64]([0.1, 0.2, 0.3])
        )
    with assert_raises():
        _ = johnson_lindenstrauss_min_dim(
            List[Float64](), eps=List[Float64]([2.0])
        )
    assert_equal(
        len(
            johnson_lindenstrauss_min_dim(
                List[Float64](), eps=List[Float64]([0.5])
            )
        ),
        0,
    )
    var auto = GaussianRandomProjection(eps=0.5, random_state=0)
    var wide = ProjectionMatrix(2, 40)
    _ = auto.fit(wide)
    assert_equal(auto.n_components[String], "auto")
    assert_equal(auto.n_components_, 33)
    assert_equal(len(auto.get_feature_names_out()), 33)
    var sparse_auto = SparseRandomProjection(eps=0.5, random_state=0)
    _ = sparse_auto.fit(wide.to_csr())
    assert_equal(sparse_auto.n_components_, 33)
    with assert_raises():
        _ = auto.fit(ProjectionMatrix(1, 40))
    with assert_raises():
        _ = auto.fit(ProjectionMatrix(2, 2))
    assert_equal(auto.n_components_, 33)  # failed fit leaves old state intact
    var expansion = GaussianRandomProjection(7, eps=2.0, random_state=0)
    _ = expansion.fit(ProjectionMatrix(2, 3))
    assert_equal(expansion.components_.rows, 7)
    assert_equal(expansion.components_.cols, 3)


def test_invalid_parameters_and_inputs() raises:
    var X = input_data()
    var model = GaussianRandomProjection(2, random_state=0)
    _ = model.fit(X)
    var original = model.components_.copy()
    with assert_raises():
        _ = model.fit(ProjectionMatrix(0, 4))
    with assert_raises():
        _ = model.transform(ProjectionMatrix(2, 3))
    with assert_raises():
        _ = model.transform(ProjectionMatrix(0, 4))
    with assert_raises():
        _ = model.inverse_transform(ProjectionMatrix(2, 4))
    X.data[0] = Float64("nan")
    with assert_raises():
        _ = model.fit(X)
    close(model.components_, original, 0)
    X = input_data()
    for count in List[Int]([0, -1]):
        _ = model.set_params(n_components=ComponentCount(count))
        with assert_raises():
            _ = model.fit(X)
    _ = model.set_params(n_components=ComponentCount(String("bad")))
    with assert_raises():
        _ = model.fit(X)
    _ = model.set_params(n_components=ComponentCount(2))
    for bad_seed in List[Int]([-1, 4294967296]):
        _ = model.set_params(random_state=bad_seed)
        with assert_raises():
            _ = model.fit(X)
    _ = model.set_params(random_state=0, eps=Float64("nan"))
    with assert_raises():
        _ = model.fit(X)
    var sparse = SparseRandomProjection(2, random_state=0)
    for bad_density in List[Float64]([0.0, -1.0, 1.1, Float64("nan")]):
        _ = sparse.set_params(density=ProjectionDensity(bad_density))
        with assert_raises():
            _ = sparse.fit(X)
    with assert_raises():
        _ = _gaussian_random_matrix(0, 2)
    with assert_raises():
        _ = _sparse_random_matrix(2, 0)


def test_csr_validation_and_duplicates() raises:
    var csr = ProjectionMatrix.from_csr(
        1,
        3,
        List[Float64]([1.0, 2.0, 3.0]),
        List[Int]([2, 0, 2]),
        List[Int]([0, 3]),
    )
    var dense = ProjectionMatrix(1, 3, List[Float64]([2.0, 0.0, 4.0]))
    close(csr, dense, 0)
    var model = GaussianRandomProjection(2, random_state=0)
    _ = model.fit(csr)
    close(model.transform(csr), model.transform(dense))
    with assert_raises():
        _ = ProjectionMatrix.from_csr(
            1, 3, List[Float64]([1.0]), List[Int]([3]), List[Int]([0, 1])
        )
    with assert_raises():
        _ = ProjectionMatrix.from_csr(
            2, 3, List[Float64]([1.0]), List[Int]([0]), List[Int]([0, 2, 1])
        )
    with assert_raises():
        _ = ProjectionMatrix(2, 3, List[Float64]([1.0]))
    csr.indptr[1] = 100
    with assert_raises():
        _ = model.transform(csr)


def test_rank_deficient_inverse_and_distribution() raises:
    var zero = ProjectionMatrix(2, 5)
    close(pseudoinverse(zero), ProjectionMatrix(5, 2), 0)
    var rank_one = ProjectionMatrix(
        2, 3, List[Float64]([1.0, 2.0, 3.0, 2.0, 4.0, 6.0])
    )
    var inverse = pseudoinverse(rank_one)
    # A A+ A = A (project multiplies by its second argument's transpose).
    var aa = project(rank_one, inverse.transpose())
    close(project(aa, rank_one.transpose()), rank_one)
    var gaussian = _gaussian_random_matrix(100, 1000, 0)
    var mean = Float64(0)
    var second = Float64(0)
    for value in gaussian.data:
        mean += value / Float64(len(gaussian.data))
        second += value * value / Float64(len(gaussian.data))
    assert_true(abs(mean) < 0.0015)
    assert_true(abs(second - 0.01) < 0.0003)
    var sparse = _sparse_random_matrix(100, 1000, 0.3, 0)
    assert_true(abs(Float64(len(sparse.data)) / 100000.0 - 0.3) < 0.01)
    var positive = 0
    for value in sparse.data:
        if value > 0:
            positive += 1
    assert_true(abs(Float64(positive) / Float64(len(sparse.data)) - 0.5) < 0.02)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
