"""File round trips, malformed archives, and restored inference for pinned parity."""
from moxie.covariance import (
    EmpiricalCovariance, EmpiricalCovarianceSchema,
    ShrunkCovariance, ShrunkCovarianceSchema,
    LedoitWolf, LedoitWolfSchema, OAS, OASSchema,
)
from moxie.serialization import ModelSchema, dumps, load_model, save_model
from numojo import NDArray, NDArrayStrides, Shape, zeros
from parity_io import Tokens, emit, emit_scalar
from std.math import nan
from std.sys import argv
from std.testing import assert_equal, assert_false, assert_raises, assert_true


def roundtrip[Schema: ModelSchema](
    model: Schema.Model, path: String,
) raises -> Schema.Model where conforms_to(Schema.Model, Deinitable):
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_true(dumps(Schema.encode(model)) == dumps(Schema.encode(restored)))
    return restored^


def inference[dtype: DType](
    name: String, score: Float64, expected_score: Float64,
    distances: NDArray[dtype], expected: NDArray[dtype], sample_count: Int,
) raises:
    assert_equal(score, expected_score)
    emit_scalar(name + "_score", score)
    assert_equal(distances.ndim, 1)
    assert_equal(distances.size, sample_count)
    for i in range(distances.size):
        assert_equal(distances.unsafe_get(i), expected.unsafe_get(i))
    emit(name + "_mahalanobis", distances)


def invalid_archives(X: NDArray[DType.float32]) raises:
    comptime Schema = EmpiricalCovarianceSchema[DType.float32]
    var model = EmpiricalCovariance[DType.float32]()
    _ = model.fit(X)

    var archive = Schema.encode(model)
    with assert_raises():
        _ = ShrunkCovarianceSchema[DType.float32].decode(archive)
    with assert_raises():
        _ = EmpiricalCovarianceSchema[DType.float64].decode(archive)

    archive.schema_version += 1
    with assert_raises():
        _ = Schema.decode(archive)
    archive.schema_version -= 1
    archive.manifest.append(0)
    with assert_raises():
        _ = Schema.decode(archive)
    _ = archive.manifest.pop()

    var saved_name = archive.payloads[0].name.copy()
    archive.payloads[0].name = "unknown"
    with assert_raises():
        _ = Schema.decode(archive)
    archive.payloads[0].name = saved_name^
    var saved_shape = archive.payloads[0].shape.copy()
    archive.payloads[0].shape = [1, X.shape[1]]
    with assert_raises():
        _ = Schema.decode(archive)
    archive.payloads[0].shape = saved_shape^
    var saved_dtype = archive.payloads[0].dtype
    archive.payloads[0].dtype = 3  # Int32 has the same item width.
    with assert_raises():
        _ = Schema.decode(archive)
    archive.payloads[0].dtype = saved_dtype
    _ = Schema.decode(archive)

    var invalid_location = zeros[DType.float32](Shape(X.shape[1]))
    invalid_location.unsafe_set(0, nan[DType.float32]())
    model._model.state.location_ = invalid_location^
    with assert_raises():
        _ = Schema.encode(model)
    _ = model.fit(X)
    var widened = model._model._covariance64.value().copy()
    widened.unsafe_set(0, widened.unsafe_get(0) + 1)
    model._model._covariance64 = widened^
    with assert_raises():
        _ = Schema.encode(model)

    var adaptive = OAS[DType.float32]()
    _ = adaptive.fit(X)
    adaptive._shrinkage = None
    with assert_raises():
        _ = OASSchema[DType.float32].encode(adaptive)


def strided_state(X: NDArray[DType.float32]) raises:
    comptime Schema = EmpiricalCovarianceSchema[DType.float32]
    var model = EmpiricalCovariance[DType.float32](store_precision=False)
    _ = model.fit(X)
    var features = X.shape[1]
    var location = model.location_()
    model._model.state.location_ = location.view_with_layout(
        Shape(features), NDArrayStrides(-1), features - 1,
    )
    var restored = Schema.decode(Schema.encode(model)).location_()
    for i in range(features):
        assert_equal(restored.item(i), location.item(features - 1 - i))


def check_models[dtype: DType](X: NDArray[dtype], probe: NDArray[dtype], path: String) raises:
    var empirical = EmpiricalCovariance[dtype](store_precision=False)
    var empirical_empty = roundtrip[EmpiricalCovarianceSchema[dtype]](empirical, path)
    assert_false(empirical_empty.__sklearn_is_fitted__())
    assert_false(empirical_empty.store_precision)
    with assert_raises():
        _ = empirical_empty.score(probe)
    _ = empirical.fit(X)
    _ = empirical.set_params(store_precision=True, assume_centered=True)
    var empirical_restored = roundtrip[EmpiricalCovarianceSchema[dtype]](empirical, path)
    assert_true(empirical_restored.store_precision)
    assert_true(empirical_restored.assume_centered)
    assert_false(Bool(empirical_restored.precision_()))
    assert_equal(empirical_restored.n_features_in_(), X.shape[1])
    inference(
        "empirical", empirical_restored.score(probe), empirical.score(probe),
        empirical_restored.mahalanobis(probe), empirical.mahalanobis(probe), probe.shape[0],
    )

    var fixed = ShrunkCovariance[dtype](shrinkage=0.3)
    var fixed_empty = roundtrip[ShrunkCovarianceSchema[dtype]](fixed, path)
    assert_false(fixed_empty.__sklearn_is_fitted__())
    assert_equal(fixed_empty.shrinkage, 0.3)
    _ = fixed.fit(X)
    _ = fixed.set_params(store_precision=False, shrinkage=0.8)
    var fixed_restored = roundtrip[ShrunkCovarianceSchema[dtype]](fixed, path)
    assert_false(fixed_restored.store_precision)
    assert_equal(fixed_restored.shrinkage, 0.8)
    assert_true(Bool(fixed_restored.precision_()))
    inference(
        "fixed", fixed_restored.score(probe), fixed.score(probe),
        fixed_restored.mahalanobis(probe), fixed.mahalanobis(probe), probe.shape[0],
    )

    var lw = LedoitWolf[dtype](block_size=2, assume_centered=True)
    var lw_empty = roundtrip[LedoitWolfSchema[dtype]](lw, path)
    assert_false(lw_empty.__sklearn_is_fitted__())
    assert_true(lw_empty.assume_centered)
    assert_equal(lw_empty.block_size, 2)
    _ = lw.fit(X)
    _ = lw.set_params(block_size=7, assume_centered=False)
    var lw_restored = roundtrip[LedoitWolfSchema[dtype]](lw, path)
    assert_false(lw_restored.assume_centered)
    assert_equal(lw_restored.block_size, 7)
    assert_equal(lw_restored.shrinkage_(), lw.shrinkage_())
    emit_scalar("lw_shrinkage", lw_restored.shrinkage_())
    inference(
        "lw", lw_restored.score(probe), lw.score(probe),
        lw_restored.mahalanobis(probe), lw.mahalanobis(probe), probe.shape[0],
    )

    var oas = OAS[dtype](store_precision=False)
    var oas_empty = roundtrip[OASSchema[dtype]](oas, path)
    assert_false(oas_empty.__sklearn_is_fitted__())
    assert_false(oas_empty.store_precision)
    with assert_raises():
        _ = oas_empty.shrinkage_()
    _ = oas.fit(X)
    var oas_restored = roundtrip[OASSchema[dtype]](oas, path)
    assert_equal(oas_restored.shrinkage_(), oas.shrinkage_())
    emit_scalar("oas_shrinkage", oas_restored.shrinkage_())
    inference(
        "oas", oas_restored.score(probe), oas.score(probe),
        oas_restored.mahalanobis(probe), oas.mahalanobis(probe), probe.shape[0],
    )

    # A loaded estimator can replace its learned state through ordinary fit.
    _ = oas_restored.fit(probe)
    var refit = oas.__sklearn_clone__()
    _ = refit.fit(probe)
    assert_equal(oas_restored.score(X), refit.score(X))


def run_case[dtype: DType](mut tokens: Tokens, path: String) raises:
    var X = tokens.matrix[dtype]()
    var probe = tokens.matrix[dtype]()
    check_models(X, probe, path)
    comptime if dtype == DType.float32:
        strided_state(rebind[NDArray[DType.float32]](X))
        invalid_archives(rebind[NDArray[DType.float32]](X^))


def main() raises:
    var fixture = String(argv()[1])
    var tokens = Tokens(fixture)
    var dtype = tokens.integer()
    var path = fixture + ".moxie"
    if dtype == 32:
        run_case[DType.float32](tokens, path)
    elif dtype == 64:
        run_case[DType.float64](tokens, path)
    else:
        raise Error("expected Float32 or Float64 fixture")
