from std.testing import assert_equal, assert_raises, assert_true
from numojo import NDArray, NDArrayStrides, Shape, zeros
from moxie.base import clone
from moxie.cross_decomposition import (
    CCA,
    CCASchema,
    PLSCanonical,
    PLSCanonicalSchema,
    PLSRegression,
    PLSRegressionSchema,
    PLSSVD,
    PLSSVDSchema,
)
from moxie.cross_decomposition._contracts import Matrix, PairedScores
from moxie.cross_decomposition._pls import _PLS
from moxie.cross_decomposition._serialization import _PLSSchema
from moxie.serialization import (
    ModelArchive,
    dumps,
    load,
    load_model,
    loads,
    save,
    save_model,
)


def inputs[dtype: DType]() raises -> NDArray[dtype]:
    var x = zeros[dtype](Shape(6, 3))
    for i in range(x.size):
        x.unsafe_set(i, Scalar[dtype]((i * i + 3 * i) % 19 - 9))
    return x^


def targets[dtype: DType](single: Bool) raises -> NDArray[dtype]:
    var y = zeros[dtype](Shape(6) if single else Shape(6, 2))
    for i in range(y.size):
        y.unsafe_set(i, Scalar[dtype]((i * i + 2 * i) % 13 - 6))
    return y^


def same_array(actual: Matrix, expected: Matrix) raises:
    assert_equal(actual.ndim, expected.ndim)
    for axis in range(expected.ndim):
        assert_equal(actual.shape[axis], expected.shape[axis])
    for i in range(expected.size):
        assert_equal(actual.unsafe_get(i), expected.unsafe_get(i))


def same_scores(actual: PairedScores, expected: PairedScores) raises:
    same_array(actual.x_scores, expected.x_scores)
    same_array(actual.y_scores, expected.y_scores)


@no_inline
def file_round_trip(archive: ModelArchive, path: String) raises -> ModelArchive:
    save(archive, path)
    return load(path)


@no_inline
def same_archive(actual: ModelArchive, expected: ModelArchive) raises:
    assert_equal(actual.type_id, expected.type_id)
    assert_equal(actual.schema_version, expected.schema_version)
    assert_equal(actual.manifest, expected.manifest)
    assert_equal(len(actual.payloads), len(expected.payloads))
    for i in range(len(expected.payloads)):
        assert_equal(actual.payloads[i].name, expected.payloads[i].name)
        assert_equal(actual.payloads[i].dtype, expected.payloads[i].dtype)
        assert_equal(actual.payloads[i].shape, expected.payloads[i].shape)
        assert_equal(actual.payloads[i].data, expected.payloads[i].data)


@no_inline
def pls_round_trip[flavor: Int, dtype: DType](
    var model: _PLS[flavor, dtype], single: Bool
) raises:
    comptime Schema = _PLSSchema[flavor, dtype]
    var path = "builds/ticket23-" + model.name() + "-" + String(dtype) + ".moxie"
    var x = inputs[dtype]()
    var y = targets[dtype](single)
    var unfitted_archive = Schema.encode(model)
    var fresh = Schema.decode(file_round_trip(unfitted_archive, path))
    assert_true(not fresh.__sklearn_is_fitted__())
    same_archive(Schema.encode(fresh), unfitted_archive)
    with assert_raises():
        _ = fresh.predict(x)

    _ = model.fit(x, y)
    var prediction = model.predict(x)
    var projected = model.transform(x, y)
    assert_equal(prediction.ndim, y.ndim)
    _ = model.set_params(n_components=-1, scale=False, max_iter=0, tol=-1.0, copy=False)
    var fitted_archive = Schema.encode(model)
    var restored = Schema.decode(file_round_trip(fitted_archive, path))
    assert_equal(restored.n_components, -1)
    assert_equal(restored.max_iter, 0)
    assert_equal(restored.tol, -1.0)
    assert_true(not restored.scale and not restored.copy_input)
    assert_equal(restored.get_n_features_in(), 3)
    assert_equal(restored.get_feature_names_out(), model.get_feature_names_out())
    same_archive(Schema.encode(restored), fitted_archive)
    assert_equal(restored.n_iter_, model.n_iter_)
    same_array(restored.predict(x), prediction)
    same_scores(restored.transform(x, y), projected)
    assert_true(not clone(restored).__sklearn_is_fitted__())
    restored.intercept_.unsafe_set(0, 987.0)
    same_array(model.predict(x), prediction)


@no_inline
def plssvd_round_trip[dtype: DType]() raises:
    comptime Schema = PLSSVDSchema[dtype]
    var path = "builds/ticket23-PLSSVD-" + String(dtype) + ".moxie"
    var model = PLSSVD[dtype](2)
    var x = inputs[dtype]()
    var y = targets[dtype](False)
    var unfitted_archive = Schema.encode(model)
    var fresh = Schema.decode(file_round_trip(unfitted_archive, path))
    assert_true(not fresh.__sklearn_is_fitted__())
    same_archive(Schema.encode(fresh), unfitted_archive)
    with assert_raises():
        _ = fresh.transform(x)

    _ = model.fit(x, y)
    var scores = model.transform(x, y)
    _ = model.set_params(n_components=-1, scale=False, copy=False)
    var fitted_archive = Schema.encode(model)
    var restored = Schema.decode(file_round_trip(fitted_archive, path))
    assert_equal(restored.n_components, -1)
    assert_true(not restored.scale and not restored.copy_input)
    assert_equal(restored.get_n_features_in(), 3)
    assert_equal(restored.get_feature_names_out(), model.get_feature_names_out())
    same_archive(Schema.encode(restored), fitted_archive)
    same_scores(restored.transform(x, y), scores)
    same_array(restored.transform(x), scores.x_scores)
    assert_true(not clone(restored).__sklearn_is_fitted__())
    restored.x_weights_.unsafe_set(0, 987.0)
    same_scores(model.transform(x, y), scores)


@no_inline
def strided_state() raises:
    var model = PLSSVD(2)
    _ = model.fit(inputs[DType.float64](), targets[DType.float64](False))
    var expected_weights = model.x_weights_.copy()
    var backing = zeros[DType.float64](Shape(12))
    for i in range(6):
        backing.unsafe_set(2 * (5 - i), model.x_weights_.unsafe_get(i))
    var expected = dumps(PLSSVDSchema.encode(model))
    model.x_weights_ = backing.view_with_layout(Shape(3, 2), NDArrayStrides(-4, -2), 10)
    assert_equal(dumps(PLSSVDSchema.encode(model)), expected)
    var restored = PLSSVDSchema.decode(loads(expected))
    same_array(restored.x_weights_, expected_weights)


@no_inline
def constant_residual() raises:
    var x = inputs[DType.float64]()
    var y = zeros[DType.float64](Shape(6))
    var model = PLSRegression(2)
    _ = model.fit(x, y)
    var path = String("builds/ticket23-constant-residual.moxie")
    save_model[PLSRegressionSchema[]](model, path)
    var restored = load_model[PLSRegressionSchema[]](path)
    assert_equal(len(restored.diagnostics_), 1)
    assert_equal(restored.diagnostics_, model.diagnostics_)
    assert_equal(len(restored.fit_warnings), 1)
    assert_equal(restored.fit_warnings[0].category, "UserWarning")
    assert_equal(restored.fit_warnings[0].message, "y residual is constant at iteration 0")
    assert_equal(len(restored.n_iter_), 0)
    same_array(restored.predict(x), model.predict(x))


@no_inline
def invalid_state() raises:
    var model = PLSCanonical(2, algorithm="svd")
    _ = model.fit(inputs[DType.float64](), targets[DType.float64](False))
    var archive = PLSCanonicalSchema.encode(model)
    with assert_raises():
        _ = CCASchema.decode(archive)
    with assert_raises():
        _ = PLSCanonicalSchema[DType.float32].decode(archive)

    var invalid = archive.copy()
    invalid.schema_version += 1
    with assert_raises():
        _ = PLSCanonicalSchema.decode(invalid)
    invalid = archive.copy()
    invalid.manifest.append(0)
    with assert_raises():
        _ = PLSCanonicalSchema.decode(invalid)
    invalid = archive.copy()
    invalid.payloads[0].name = "unknown"
    with assert_raises():
        _ = PLSCanonicalSchema.decode(invalid)
    invalid = archive.copy()
    invalid.payloads[0].shape = [2, 3]
    with assert_raises():
        _ = PLSCanonicalSchema.decode(invalid)
    invalid = archive.copy()
    invalid.payloads[0].dtype = 4  # Int64 is the same width as the required Float64.
    with assert_raises():
        _ = PLSCanonicalSchema.decode(invalid)

    var svd = PLSSVD(2)
    _ = svd.fit(inputs[DType.float64](), targets[DType.float64](False))
    invalid = PLSSVDSchema.encode(svd)
    assert_equal(invalid.payloads[4].name, "_x_std")
    for i in range(8):
        invalid.payloads[4].data[i] = 0
    with assert_raises():
        _ = PLSSVDSchema.decode(invalid)
    var fresh = PLSSVDSchema.encode(PLSSVD())
    fresh.payloads.append(invalid.payloads[0].copy())
    with assert_raises():
        _ = PLSSVDSchema.decode(fresh)
    svd.x_weights_ = zeros[DType.float64](Shape(6))
    with assert_raises():
        _ = PLSSVDSchema.encode(svd)


def main() raises:
    pls_round_trip(PLSRegression[DType.float32](2), True)
    pls_round_trip(PLSCanonical(2, algorithm="nipals"), False)
    pls_round_trip(PLSCanonical(2, algorithm="svd"), False)
    pls_round_trip(CCA(2), False)
    plssvd_round_trip[DType.float32]()
    plssvd_round_trip[DType.float64]()
    strided_state()
    constant_residual()
    invalid_state()
    print("PASS: cross-decomposition model schemas")
