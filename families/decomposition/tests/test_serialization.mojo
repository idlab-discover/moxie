"""Dense decomposition persistence contracts."""

from std.math import isinf, isnan
from std.sys import argv
from std.testing import assert_equal, assert_raises, assert_true
from numojo import NDArray, NDArrayStrides, Shape, zeros
from moxie.base import clone
from moxie.decomposition import FactorAnalysis, PCA, IncrementalPCA, TruncatedSVD
from moxie.decomposition._pca import ComponentCount
from moxie.decomposition_serialization import (
    FactorAnalysisSchema, PCASchema, IncrementalPCASchema, TruncatedSVDSchema,
)
from moxie.serialization import dumps, loads, save_model, load_model


def same_array[dtype: DType](left: NDArray[dtype], right: NDArray[dtype]) raises:
    assert_equal(left.ndim, right.ndim)
    for axis in range(left.ndim):
        assert_equal(left.shape[axis], right.shape[axis])
    for i in range(left.size):
        assert_equal(left.unsafe_get(i), right.unsafe_get(i))


def training_data[dtype: DType]() raises -> NDArray[dtype]:
    var X = zeros[dtype](Shape(4, 3))
    var values = [4, 1, 0, -3, 2, 6, 9, 2, 3, 4, 7, 2]
    for i in range(len(values)):
        X.unsafe_set(i, Scalar[dtype](values[i]))
    return X^


def test_pca[dtype: DType](path: String) raises:
    comptime Schema = PCASchema[dtype]
    var X = training_data[dtype]()
    var fresh = PCA[dtype](0.8, copy=False, whiten=True, svd_solver="full",
        tol=0.25, iterated_power=3, n_oversamples=4, power_iteration_normalizer="QR", random_state=42)
    save_model[Schema](fresh, path)
    var unfitted = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(fresh)), dumps(Schema.encode(unfitted)))
    with assert_raises(contains="NotFittedError: PCA"):
        _ = unfitted.transform(X)

    var model = PCA[dtype](2, whiten=True, svd_solver="auto")
    _ = model.fit(X)
    _ = model.set_params(svd_solver="randomized", n_components=ComponentCount(String("mle")))
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(restored)))
    assert_equal(restored.svd_solver, String("randomized"))
    assert_equal(restored._fit_svd_solver, String("full"))
    assert_equal(restored.n_components_, 2)
    assert_true(not clone(restored).__sklearn_is_fitted__())
    var projected = restored.transform(X)
    same_array(model.transform(X), projected)
    same_array(model.inverse_transform(projected), restored.inverse_transform(projected))
    same_array(model.get_covariance(), restored.get_covariance())
    same_array(model.get_precision(), restored.get_precision())
    same_array(model.score_samples(X), restored.score_samples(X))
    var owned = Schema.decode(Schema.encode(model))
    model.mean_.value().unsafe_set(0, 100)
    same_array(owned.transform(X), projected)

    var rank_zero = PCA[dtype](0)
    _ = rank_zero.fit(X)
    var empty = Schema.decode(loads(dumps(Schema.encode(rank_zero))))
    assert_true(empty.__sklearn_is_fitted__())
    assert_equal(empty.n_components_, 0)
    assert_equal(empty.components_.value().shape[0], 0)

    var constant = PCA[dtype](1)
    _ = constant.fit(zeros[dtype](Shape(3, 2)))
    var degenerate = Schema.decode(Schema.encode(constant))
    assert_true(isnan(degenerate.explained_variance_ratio_.value().unsafe_get(0)))

    var mean = restored.mean_.value().copy()
    var strided = mean.view_with_layout(Shape(3), NDArrayStrides(-1), 2)
    restored.mean_ = strided^
    var linear = Schema.decode(Schema.encode(restored))
    for i in range(3):
        assert_equal(linear.mean_.value().unsafe_get(i), mean.unsafe_get(2 - i))

    var archive = Schema.encode(restored)
    archive.schema_version = 2
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = Schema.decode(archive)
    archive.schema_version = 1
    archive.payloads[0].shape = [3, 2]
    with assert_raises(contains="invalid shape for components_"):
        _ = Schema.decode(archive)
    restored.singular_values_.value().unsafe_set(0, -1)
    with assert_raises(contains="invalid value in singular_values_"):
        _ = Schema.encode(restored)


def test_incremental_pca[dtype: DType](path: String) raises:
    comptime Schema = IncrementalPCASchema[dtype]
    var X = training_data[dtype]()
    var shifted = X.copy()
    for i in range(shifted.size):
        shifted.unsafe_set(i, X.unsafe_get(i) + 11)

    var model = IncrementalPCA[dtype](2, whiten=True, batch_size=3)
    save_model[Schema](model, path)
    var unfitted = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(unfitted)))
    with assert_raises(contains="NotFittedError: IncrementalPCA"):
        _ = unfitted.transform(X)

    _ = model.partial_fit(X)
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(restored.n_samples_seen_, 4)
    assert_equal(restored.batch_size_, 0)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(restored)))
    same_array(model.transform(shifted), restored.transform(shifted))
    _ = model.partial_fit(shifted)
    _ = restored.partial_fit(shifted)
    assert_equal(restored.n_samples_seen_, 8)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(restored)))
    same_array(model.transform(X), restored.transform(X))

    _ = model.fit(X)
    _ = model.set_params(batch_size=Optional[Int](7))
    var fitted = Schema.decode(Schema.encode(model))
    assert_equal(fitted.batch_size_, 3)
    assert_equal(fitted.batch_size.value(), 7)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(fitted)))

    var archive = Schema.encode(model)
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = PCASchema[dtype].decode(archive)
    archive.payloads[2].name = "foreign_array"
    with assert_raises(contains="missing array var_"):
        _ = Schema.decode(archive)

    var single = IncrementalPCA[dtype](1)
    _ = single.partial_fit(zeros[dtype](Shape(1, 2)))
    var single_restored = Schema.decode(Schema.encode(single))
    assert_true(isnan(single_restored.explained_variance_.value().unsafe_get(0)))
    _ = single_restored.partial_fit(X.view_with_layout(Shape(4, 2), NDArrayStrides(3, 1), 0))
    assert_equal(single_restored.n_samples_seen_, 5)


def test_truncated_svd[dtype: DType](path: String) raises:
    comptime Schema = TruncatedSVDSchema[dtype]
    var X = training_data[dtype]()
    var model = TruncatedSVD[dtype](2, algorithm="arpack", n_iter=4,
        n_oversamples=3, power_iteration_normalizer="QR", random_state=7, tol=0.01)
    save_model[Schema](model, path)
    var unfitted = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(unfitted)))
    with assert_raises(contains="NotFittedError: This TruncatedSVD"):
        _ = unfitted.transform(X)
    assert_equal(unfitted.algorithm, String("arpack"))
    assert_equal(unfitted.n_iter, 4)
    assert_equal(unfitted.n_oversamples, 3)
    assert_equal(unfitted.power_iteration_normalizer, String("QR"))
    assert_equal(unfitted.random_state.value(), 7)
    assert_equal(unfitted.tol, 0.01)

    _ = model.fit(X)
    _ = model.set_params(n_components=1, algorithm="randomized")
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(restored)))
    assert_equal(restored.n_components, 1)
    assert_equal(restored.components_.value().shape[0], 2)
    assert_equal(restored.algorithm, String("randomized"))
    assert_equal(restored.get_feature_names_out(), model.get_feature_names_out())
    assert_true(not clone(restored).__sklearn_is_fitted__())
    var projected = model.transform(X)
    same_array(projected, restored.transform(X))
    same_array(model.inverse_transform(projected), restored.inverse_transform(projected))
    model.components_.value().unsafe_set(0, 100)
    same_array(projected, restored.transform(X))

    var archive = Schema.encode(restored)
    archive.schema_version = 2
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = Schema.decode(archive)
    archive.schema_version = 1
    archive.payloads[1].name = "foreign_array"
    with assert_raises(contains="missing array singular_values_"):
        _ = Schema.decode(archive)

    var constant = TruncatedSVD[dtype](2, random_state=7)
    _ = constant.fit(zeros[dtype](Shape(1, 3)))
    var degenerate = Schema.decode(Schema.encode(constant))
    assert_equal(degenerate.components_.value().shape[0], 1)
    assert_true(isnan(degenerate.explained_variance_ratio_.value().unsafe_get(0)))


def test_factor_analysis[dtype: DType](path: String) raises:
    comptime Schema = FactorAnalysisSchema[dtype]
    var X = training_data[dtype]()
    var init = List[Float64](length=3, fill=1.25)
    var model = FactorAnalysis[dtype](1, tol=0.001, max_iter=8, noise_variance_init=init^,
        svd_method="lapack", iterated_power=2, rotation=String("varimax"), random_state=None)
    save_model[Schema](model, path)
    var unfitted = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(unfitted)))
    with assert_raises(contains="NotFittedError: This FactorAnalysis"):
        _ = unfitted.transform(X)
    assert_equal(unfitted.noise_variance_init.value(), List[Float64](length=3, fill=1.25))
    assert_equal(unfitted.svd_method, String("lapack"))
    assert_equal(unfitted.iterated_power, 2)
    assert_equal(unfitted.rotation.value(), String("varimax"))
    assert_true(not unfitted.random_state)
    assert_equal(unfitted.tol, 0.001)

    _ = model.fit(X)
    _ = model.set_params(copy=False, max_iter=1, n_components=Optional[Optional[Int]](Optional[Int](0)))
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(dumps(Schema.encode(model)), dumps(Schema.encode(restored)))
    assert_equal(restored.n_components.value(), 0)
    assert_equal(restored.components_.value().shape[0], 1)
    assert_equal(restored.max_iter, 1)
    assert_true(not restored.copy)
    assert_true(restored.n_iter_ > restored.max_iter)
    assert_equal(restored.get_feature_names_out(), model.get_feature_names_out())
    assert_true(not clone(restored).__sklearn_is_fitted__())
    same_array(model.transform(X), restored.transform(X))
    same_array(model.get_covariance(), restored.get_covariance())
    same_array(model.get_precision(), restored.get_precision())
    same_array(model.score_samples(X), restored.score_samples(X))
    assert_equal(model.score(X), restored.score(X))

    var archive = Schema.encode(restored)
    archive.schema_version = 2
    with assert_raises(contains="unsupported estimator type or state schema"):
        _ = Schema.decode(archive)
    archive.schema_version = 1
    archive.payloads[0].shape = [3, 1]
    with assert_raises(contains="invalid shape for components_"):
        _ = Schema.decode(archive)
    restored.noise_variance_.value().unsafe_set(0, 0)
    with assert_raises(contains="invalid value in noise_variance_"):
        _ = Schema.encode(restored)
    restored.noise_variance_.value().unsafe_set(0, 1)
    restored.loglike_.append(0)
    with assert_raises(contains="invalid FactorAnalysis likelihood history length"):
        _ = Schema.encode(restored)

    var no_factors = FactorAnalysis[dtype](0)
    _ = no_factors.fit(X)
    var zero_rank = Schema.decode(Schema.encode(no_factors))
    assert_equal(zero_rank.components_.value().shape[0], 0)
    same_array(no_factors.score_samples(X), zero_rank.score_samples(X))
    var singular = FactorAnalysis[dtype](1, svd_method="lapack", max_iter=1)
    _ = singular.fit(zeros[dtype](Shape(3, 2)))
    var degenerate = Schema.decode(Schema.encode(singular))
    assert_true(isinf(degenerate.loglike_[0]) and degenerate.loglike_[0] > 0)


def main() raises:
    var path = String(argv()[1])
    test_pca[DType.float32](path)
    test_pca[DType.float64](path)
    test_incremental_pca[DType.float32](path)
    test_incremental_pca[DType.float64](path)
    test_truncated_svd[DType.float32](path)
    test_truncated_svd[DType.float64](path)
    test_factor_analysis[DType.float32](path)
    test_factor_analysis[DType.float64](path)
    var model = PCA[DType.float32]()
    save_model[PCASchema[DType.float32]](model, path)
    with assert_raises(contains="projection dtype mismatch"):
        _ = load_model[PCASchema[DType.float64]](path)
    var factor = FactorAnalysis[DType.float32]()
    var archive = FactorAnalysisSchema[DType.float32].encode(factor)
    with assert_raises(contains="projection dtype mismatch"):
        _ = FactorAnalysisSchema[DType.float64].decode(archive)
    print("PASS: PCA, IncrementalPCA, TruncatedSVD and FactorAnalysis serialization contracts in Float32 and Float64")
