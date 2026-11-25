from std.testing import assert_equal, assert_raises, assert_true
from numojo import NDArray, NDArrayStrides, Shape, zeros
from moxie.base import clone
from moxie.compose import ColumnTable
from moxie.cross_decomposition import (
    CCA,
    CCASchema,
    PLSCanonical,
    PLSCanonicalSchema,
    PLSMetadataRequest,
    PLSRegression,
    PLSRegressionSchema,
    PLSSVD,
    PLSSVDSchema,
    PairedTableScores,
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


def split_columns[dtype: DType](
    x: NDArray[dtype], names: List[String],
) raises -> ColumnTable[dtype]:
    var rows = x.shape[0]
    var base = zeros[dtype](Shape(rows, 1))
    var first = zeros[DType.int64](Shape(rows))
    var last = zeros[DType.float64](Shape(rows))
    for row in range(rows):
        first.unsafe_set(row, Int64(x.unsafe_get(row * 3)))
        base.unsafe_set(row, x.unsafe_get(row * 3 + 1))
        last.unsafe_set(row, Float64(x.unsafe_get(row * 3 + 2)))
    var table = ColumnTable(base, [names[1].copy()])
    table.add_numeric(names[0], first, position=0)
    table.add_numeric(names[2], last)
    return table^


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
    assert_equal(unfitted_archive.schema_version, 1)
    var fresh = Schema.decode(file_round_trip(unfitted_archive, path))
    assert_true(not fresh.__sklearn_is_fitted__())
    assert_true(fresh.get_metadata_routing().score.value().isa[NoneType]())
    same_archive(Schema.encode(fresh), unfitted_archive)
    with assert_raises():
        _ = fresh.predict(x)

    _ = model.set_score_request(sample_weight=String("validation_weight")).fit(x, y)
    var prediction = model.predict(x)
    var projected = model.transform(x, y)
    assert_equal(prediction.ndim, y.ndim)
    _ = model.set_params(n_components=-1, scale=False, max_iter=0, tol=-1.0, copy=False)
    var fitted_archive = Schema.encode(model)
    assert_equal(fitted_archive.schema_version, 2)
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
    var cloned = clone(restored)
    assert_true(not cloned.__sklearn_is_fitted__())
    assert_equal(cloned.get_metadata_routing().score.requested_name().or_else(""), "validation_weight")
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
def score_requests() raises:
    comptime Schema = PLSRegressionSchema[]
    var model = PLSRegression()
    var metadata = {"sample_weight": 7, "validation_weight": 8}
    with assert_raises():
        _ = model.get_metadata_routing().score.route(metadata)

    _ = model.set_score_request(sample_weight=False)
    var declined = Schema.decode(loads(dumps(Schema.encode(model))))
    assert_true(not declined.get_metadata_routing().score.route(metadata))
    _ = model.set_score_request(sample_weight=True).set_score_request()
    var accepted = Schema.decode(loads(dumps(Schema.encode(model))))
    assert_equal(accepted.get_metadata_routing().score.route(metadata).or_else(-1), 7)

    _ = model.set_score_request(sample_weight=String("validation_weight"))
    with assert_raises():
        _ = model.set_score_request(sample_weight=String("invalid alias"))
    assert_equal(model.get_metadata_routing().score.route(metadata).or_else(-1), 8)
    var requests = model.get_metadata_routing()
    assert_true(not requests.fit.requested_name())
    var snapshot: PLSMetadataRequest = requests
    var score_only = PLSMetadataRequest(requests.score.copy())
    var invalid = Schema.encode(model)
    invalid.manifest[len(invalid.manifest) - 1] = 32
    with assert_raises():
        _ = Schema.decode(invalid)

    _ = model.set_score_request(sample_weight=None)
    assert_equal(Schema.encode(model).schema_version, 1)
    assert_equal(snapshot.score.route(metadata).or_else(-1), 8)
    assert_equal(score_only.score.route(metadata).or_else(-1), 8)
    with assert_raises():
        _ = model.get_metadata_routing().score.route(metadata)


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
    invalid.schema_version = 99
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


def encode_values(array: Matrix) raises -> String:
    var values = List[String](capacity=array.size)
    for i in range(array.size):
        values.append(String(array.unsafe_get(i)))
    return ",".join(values)


def emit_named_case(
    estimator: String, names: List[String], scores: PairedTableScores
) raises:
    var schema = scores.x_scores.schema()
    for dtype in schema.dtypes:
        assert_equal(dtype, "float64")
    print(
        "named|" + estimator + "|" + ",".join(names) + "|" + ",".join(schema.names)
        + "|" + encode_values(scores.x_scores.values()) + "|" + encode_values(scores.y_scores)
    )


def same_table_scores(
    actual: PairedTableScores, expected: PairedScores, components: List[String]
) raises:
    assert_equal(actual.x_scores.schema().names, components)
    same_array(actual.x_scores.values(), expected.x_scores)
    same_array(actual.y_scores, expected.y_scores)


@no_inline
def named_pls[flavor: Int, dtype: DType](var model: _PLS[flavor, dtype], single: Bool) raises:
    comptime Schema = _PLSSchema[flavor, dtype]
    var x = inputs[dtype]()
    var y = targets[dtype](single)
    var names: List[String] = ["mass", "âge", "taux"]
    var table = ColumnTable(x, names)
    var reordered: List[String] = ["âge", "mass", "taux"]
    var wrong = ColumnTable(x, reordered)
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(table)
    with assert_raises(contains="NotFittedError"):
        _ = model.transform_table(x)
    var positional = clone(model)
    var scores = model.fit_transform(table, y)
    var components = model.get_feature_names_out(names.copy())
    var mixed = split_columns(x, names)
    same_table_scores(model.fit_transform_table(mixed, y), scores, components)
    same_array(model.transform(mixed), scores.x_scores)
    same_array(model.predict(mixed), model.predict(x))
    same_table_scores(positional.fit_transform_table(x, y), scores, components)
    same_table_scores(model.fit_transform_table(table, y), scores, components)
    same_table_scores(model.transform_table(x, y), scores, components)
    assert_equal(model.transform_table(x).schema().names, components)
    same_array(model.transform_table(table).values(), scores.x_scores)
    assert_equal(model.feature_names_in_.value(), names)
    same_scores(scores, model.transform(table, y))
    same_array(model.transform(table), scores.x_scores)
    same_array(model.predict(table), model.predict(x))
    var weights = zeros[DType.float64](Shape(6))
    for i in range(6):
        weights.unsafe_set(i, Float64(i + 1))
    assert_equal(model.score(table, y, weights.copy()), model.score(x, y, weights^))
    assert_equal(model.score(mixed, y), model.score(x, y))
    with assert_raises(contains="The feature names should match those that were passed during fit."):
        _ = model.predict(wrong)
    with assert_raises(contains="The feature names should match those that were passed during fit."):
        _ = model.transform_table(wrong, y)
    with assert_raises(contains="The feature names should match those that were passed during fit."):
        _ = model.score(wrong, y)
    with assert_raises(contains="input_features is not equal to feature_names_in_"):
        _ = model.get_feature_names_out(reordered.copy())
    var short_names: List[String] = ["mass"]
    with assert_raises(contains="input_features is not equal to feature_names_in_"):
        _ = model.get_feature_names_out(short_names^)
    with assert_raises(contains="inconsistent numbers of samples"):
        _ = model.fit(wrong, zeros[dtype](Shape(2)))
    assert_equal(model.feature_names_in_.value(), names)

    var archive = Schema.encode(model)
    assert_equal(archive.schema_version, 3)
    var restored = Schema.decode(loads(dumps(archive)))
    assert_true(restored.get_metadata_routing().score.value().isa[NoneType]())
    assert_equal(restored.feature_names_in_.value(), names)
    _ = model.set_score_request(sample_weight=String("validation_weight"))
    archive = Schema.encode(model)
    restored = Schema.decode(loads(dumps(archive)))
    same_archive(Schema.encode(restored), archive)
    assert_equal(restored.get_metadata_routing().score.requested_name().or_else(""), "validation_weight")
    same_array(restored.predict(table), model.predict(table))
    assert_true(not clone(restored).feature_names_in_)
    emit_named_case(
        model.name(), restored.feature_names_in_.value(),
        restored.transform_table(mixed, y),
    )

    var snapshot = restored.copy()
    _ = restored.fit_inplace(x, y)
    assert_true(not restored.feature_names_in_)
    assert_equal(snapshot.feature_names_in_.value(), names)
    _ = snapshot.fit(wrong, y)
    assert_equal(snapshot.feature_names_in_.value(), reordered)
    _ = snapshot.fit(x, y)
    assert_true(not snapshot.feature_names_in_)


@no_inline
def named_plssvd() raises:
    var x = inputs[DType.float32]()
    var y = targets[DType.float32](False)
    var names: List[String] = ["mass", "âge", "taux"]
    var table = ColumnTable(x, names)
    var wrong = ColumnTable(x, ["âge", "mass", "taux"])
    var model = PLSSVD[DType.float32](2)
    comptime Schema = PLSSVDSchema[DType.float32]
    with assert_raises(contains="NotFittedError"):
        _ = model.transform_table(table)
    var positional = clone(model)
    var scores = model.fit_transform(table, y)
    var components = model.get_feature_names_out(names.copy())
    var mixed = split_columns(x, names)
    same_table_scores(model.fit_transform_table(mixed, y), scores, components)
    same_array(model.transform(mixed), scores.x_scores)
    same_table_scores(positional.fit_transform_table(x, y), scores, components)
    same_table_scores(model.fit_transform_table(table, y), scores, components)
    same_table_scores(model.transform_table(x, y), scores, components)
    assert_equal(model.transform_table(table).schema().names, components)
    same_array(model.transform_table(x).values(), scores.x_scores)
    same_scores(scores, model.transform(table, y))
    same_scores(scores, model.transform(x, y))
    with assert_raises(contains="The feature names should match those that were passed during fit."):
        _ = model.transform_table(wrong)
    var text = mixed.copy()
    text.add_strings("text", ["a", "b", "c", "d", "e", "f"])
    with assert_raises(contains="column 'text' requires numeric storage"):
        _ = model.fit(text, y)
    assert_equal(model.feature_names_in_.value(), names)
    var archive = Schema.encode(model)
    assert_equal(archive.schema_version, 2)
    var restored = Schema.decode(loads(dumps(archive)))
    same_archive(Schema.encode(restored), archive)
    same_array(restored.transform(table), scores.x_scores)
    assert_true(not clone(restored).feature_names_in_)
    emit_named_case(
        model.name(), restored.feature_names_in_.value(),
        restored.transform_table(mixed, y),
    )
    var invalid = archive.copy()
    # All three names occupy 12 bytes each, including the length prefix.
    # Replace the last name with the first to simulate a malformed archive.
    var end = len(archive.manifest)
    for i in range(4):
        invalid.manifest[end - 4 + i] = archive.manifest[end - 28 + i]
    with assert_raises(contains="fitted feature names must be unique"):
        _ = Schema.decode(invalid)
    _ = invalid.manifest.pop()
    with assert_raises():
        _ = Schema.decode(invalid)
    model.feature_names_in_ = List[String]()
    with assert_raises(contains="feature names must match the fitted width"):
        _ = Schema.encode(model)
    _ = restored.fit(x, y)
    assert_true(not restored.feature_names_in_)
    same_array(restored.transform(table), scores.x_scores)
    assert_equal(Schema.encode(restored).schema_version, 1)


def main() raises:
    pls_round_trip(PLSRegression[DType.float32](2), True)
    pls_round_trip(PLSCanonical(2, algorithm="nipals"), False)
    pls_round_trip(PLSCanonical(2, algorithm="svd"), False)
    pls_round_trip(CCA(2), False)
    plssvd_round_trip[DType.float32]()
    plssvd_round_trip[DType.float64]()
    strided_state()
    constant_residual()
    score_requests()
    invalid_state()
    named_pls(PLSRegression[DType.float32](2), True)
    named_pls(PLSCanonical(2, algorithm="svd"), False)
    named_pls(CCA(2), False)
    named_plssvd()
    print("PASS: cross-decomposition model schemas")
