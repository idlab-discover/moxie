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
    ComponentCount,
    johnson_lindenstrauss_min_dim,
    _gaussian_random_matrix,
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
    _ = model.fit(X.to_csr())
    close(model.inverse_transform(transformed.to_csr()), model.inverse_transform(transformed))
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
    with assert_raises():
        _ = _gaussian_random_matrix(0, 2)


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_shared_rng_and_dense_boundaries() raises:
    from moxie._random import RandomState, entropy_state
    var rng = RandomState(42)
    var expected = rng.copy()
    var model = GaussianRandomProjection(1)
    var X = ProjectionMatrix(2, 3)
    for _ in range(3):
        _ = model.fit(X, rng=rng)
        for j in range(3):
            assert_equal(model.components_.data[j], expected.standard_normal())
    var before = rng.copy()
    with assert_raises():
        _ = model.fit(ProjectionMatrix(0, 3), rng=rng)
    assert_equal(rng.next_uint32(), before.next_uint32())
    var context = entropy_state()
    _ = model.fit(X, rng=context)
    _ = model.fit(X)
    assert_equal(model.components_.rows, 1)
    var params = model.get_params()
    params.random_state = 42
    _ = model.set_params(params)
    with assert_raises():
        _ = model.fit(X, rng=rng)
    params.random_state = None
    _ = model.set_params(params)
    _ = model.fit(X, rng=rng)
    model.components_ = model.components_.to_csr()
    _ = model.transform(X)
    _ = model.inverse_transform(ProjectionMatrix(2, 1))


def test_inverse_cache_ownership() raises:
    var X = input_data()
    var cached = GaussianRandomProjection(2, random_state=42, compute_inverse_components=True)
    var Y = cached.fit_transform(X)
    var original = cached.inverse_transform(Y)
    cached.components_.data[0] += 10
    close(cached.inverse_transform(Y), original, 0)
    _ = cached.set_params(compute_inverse_components=False)
    var changed = cached.inverse_transform(Y)
    assert_true(abs(changed.data[0] - original.data[0]) > 1e-5)
    var copied = cached.copy()
    copied.components_.data[0] = 0
    assert_true(cached.components_.data[0] != 0)
    _ = cached.set_params(compute_inverse_components=True)
    _ = cached.fit(ProjectionMatrix(2, 7))
    assert_equal(cached.inverse_components_.value().rows, 7)
    assert_equal(cached.n_features_in_, 7)
