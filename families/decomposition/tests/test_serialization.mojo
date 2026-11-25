"""Runtime PCA and IncrementalPCA persistence contracts."""

from std.math import isnan
from std.sys import argv
from std.testing import assert_equal, assert_raises, assert_true
from numojo import NDArray, NDArrayStrides, Shape, zeros
from moxie.base import clone
from moxie.decomposition import PCA, IncrementalPCA
from moxie.decomposition._pca import ComponentCount
from moxie.decomposition_serialization import PCASchema, IncrementalPCASchema
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


def main() raises:
    var path = String(argv()[1])
    test_pca[DType.float32](path)
    test_pca[DType.float64](path)
    test_incremental_pca[DType.float32](path)
    test_incremental_pca[DType.float64](path)
    var model = PCA[DType.float32]()
    save_model[PCASchema[DType.float32]](model, path)
    with assert_raises(contains="projection dtype mismatch"):
        _ = load_model[PCASchema[DType.float64]](path)
    print("PASS: runtime PCA and IncrementalPCA serialization contracts in Float32 and Float64")
