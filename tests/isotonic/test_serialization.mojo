from std.math import isnan
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from moxie.base import clone
from moxie.isotonic import Increasing, IsotonicRegression
from moxie.isotonic_serialization import IsotonicRegressionSchema
from moxie.serialization import (
    BinaryWriter,
    ModelArchive,
    dtype_code,
    dumps,
    load_model,
    save_model,
)
from numojo import array


def round_trips[dtype: DType]() raises:
    comptime Schema = IsotonicRegressionSchema[dtype]
    var path = "builds/isotonic-" + String(dtype) + ".moxie"
    var x: List[Scalar[dtype]] = [0, 1, 2]
    var y: List[Scalar[dtype]] = [3, 2, 1]
    var outside: List[Scalar[dtype]] = [-1, 3]
    var model = IsotonicRegression[dtype](
        y_min=0.0, increasing="auto", out_of_bounds="raise"
    )
    _ = model.set_fit_request(sample_weight="training_weights").set_score_request(sample_weight=True)
    save_model[Schema](model, path)
    var fresh = load_model[Schema](path)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.get_metadata_routing().fit.value()[String], "training_weights")
    assert_true(fresh.get_metadata_routing().score.value()[Bool])
    assert_equal(fresh.y_min.or_else(-1.0), 0.0)
    assert_false(Bool(fresh.y_max))
    assert_equal(fresh.increasing[String], "auto")
    assert_equal(fresh.out_of_bounds, "raise")
    with assert_raises(contains="NotFittedError"):
        _ = fresh.predict(x)

    _ = model.fit(x, y)
    var query: List[Scalar[dtype]] = [0.5, 1.5]
    var expected: List[Scalar[dtype]] = [2.5, 1.5]
    _ = model.set_params(y_max=0.5, increasing=Increasing(True), out_of_bounds="nan")
    with assert_raises(contains="outside"):
        _ = model.predict(outside)

    save_model[Schema](model, path)
    var fitted = load_model[Schema](path)
    assert_true(fitted.__sklearn_is_fitted__())
    assert_equal(fitted.get_metadata_routing().fit.value()[String], "training_weights")
    assert_true(fitted.get_metadata_routing().score.value()[Bool])
    assert_equal(fitted.y_min.or_else(-1.0), 0.0)
    assert_equal(fitted.y_max.or_else(-1.0), 0.5)
    assert_true(fitted.increasing[Bool])
    assert_false(fitted.increasing_)
    assert_equal(fitted.out_of_bounds, "nan")
    assert_equal(fitted.X_min_, Scalar[dtype](0))
    assert_equal(fitted.X_max_, Scalar[dtype](2))
    assert_equal(fitted.X_thresholds_, x)
    assert_equal(fitted.y_thresholds_, y)
    assert_equal(fitted.predict(query), expected)
    assert_equal(fitted.transform(array(query, [2, 1])).tolist(), expected)
    var predictions = fitted.predict(outside)
    assert_true(isnan(predictions[0]) and isnan(predictions[1]))
    assert_equal(dumps(Schema.encode(fitted)), dumps(Schema.encode(model)))
    assert_false(clone(fitted).__sklearn_is_fitted__())
    _ = fitted.set_fit_request(sample_weight=False).set_score_request(sample_weight=None)
    assert_equal(model.get_metadata_routing().fit.value()[String], "training_weights")
    assert_true(model.get_metadata_routing().score.value()[Bool])
    fitted.y_thresholds_[0] = 7
    assert_equal(model.y_thresholds_[0], Scalar[dtype](3))

    _ = model.set_params(out_of_bounds="clip")
    save_model[Schema](model, path)
    var clipped = load_model[Schema](path)
    var endpoints: List[Scalar[dtype]] = [3, 1]
    assert_equal(clipped.predict(outside), endpoints)
    var direct = clipped.f_(outside)
    assert_true(isnan(direct[0]) and isnan(direct[1]))

    var singleton: List[Scalar[dtype]] = [1]
    _ = model.set_params(out_of_bounds="raise").fit(singleton, singleton)
    save_model[Schema](model, path)
    var constant = load_model[Schema](path)
    var constant_values: List[Scalar[dtype]] = [0.5, 0.5]
    assert_equal(constant.predict(outside), constant_values)

    var tied_x: List[Scalar[dtype]] = [0, 0, 0, 0]
    var varied_y: List[Scalar[dtype]] = [1, 2, 3, 4]
    _ = model.set_params(increasing=Increasing(String("auto"))).fit(tied_x, varied_y)
    assert_equal(len(model.fit_warnings), 2)
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(len(restored.fit_warnings), 0)
    assert_equal(restored.predict(tied_x), model.predict(tied_x))


def version_one[dtype: DType](fitted: Bool) raises -> ModelArchive:
    comptime Schema = IsotonicRegressionSchema[dtype]
    var archive = ModelArchive(Schema.TypeId, 1)
    var writer = BinaryWriter()
    writer.uint(UInt64(dtype_code[dtype]()))
    writer.boolean(False)  # y_min absent
    writer.boolean(False)  # y_max absent
    writer.boolean(True)  # increasing is Bool
    writer.boolean(True)
    writer.string("clip")
    writer.boolean(fitted)
    if fitted:
        writer.scalar[dtype](0)
        writer.scalar[dtype](1)
        writer.boolean(True)
        var shape: List[Int] = [2]
        var thresholds: List[Scalar[dtype]] = [0, 1]
        archive.add_array[dtype]("X_thresholds_", shape, thresholds)
        archive.add_array[dtype]("y_thresholds_", shape, thresholds)
    archive.manifest = writer^.finish()
    return archive^


def legacy_round_trips[dtype: DType]() raises:
    comptime Schema = IsotonicRegressionSchema[dtype]
    for fitted in [False, True]:
        var model = Schema.decode(version_one[dtype](fitted))
        assert_equal(model.__sklearn_is_fitted__(), fitted)
        var routing = model.get_metadata_routing()
        assert_true(routing.fit.value().isa[NoneType]())
        assert_true(routing.score.value().isa[NoneType]())
        if fitted:
            var query: List[Scalar[dtype]] = [-1, 0.5, 2]
            var expected: List[Scalar[dtype]] = [0, 0.5, 1]
            assert_equal(model.predict(query), expected)
        assert_equal(Schema.encode(model).schema_version, 2)


def invalid_requests() raises:
    comptime Schema = IsotonicRegressionSchema[]
    var base = version_one[DType.float64](False)
    base.schema_version = Schema.StateVersion
    var unknown_tag = base.copy()
    var writer = BinaryWriter()
    writer.uint(4)
    unknown_tag.manifest.extend(writer^.finish())
    with assert_raises(contains="unknown isotonic sample-weight request"):
        _ = Schema.decode(unknown_tag)

    for invalid in ["invalid-alias", "$UNCHANGED$", "sample_weight", ""]:
        var bad = base.copy()
        var alias_writer = BinaryWriter()
        alias_writer.uint(3)
        alias_writer.string(invalid)
        alias_writer.uint(0)
        bad.manifest.extend(alias_writer^.finish())
        with assert_raises(contains="invalid isotonic sample-weight alias"):
            _ = Schema.decode(bad)


def invalid_archives() raises:
    comptime Schema = IsotonicRegressionSchema[]
    var model = IsotonicRegression()
    var unfitted = Schema.encode(model)
    var shape: List[Int] = [3]
    var x: List[Float64] = [0, 1, 2]
    unfitted.add_array[DType.float64]("X_thresholds_", shape, x)
    with assert_raises(contains="unexpected number"):
        _ = Schema.decode(unfitted)

    _ = model.fit(x, x)
    var archive = Schema.encode(model)
    with assert_raises(contains="dtype mismatch"):
        _ = IsotonicRegressionSchema[DType.float32].decode(archive)
    var bad = archive.copy()
    bad.type_id = "other"
    with assert_raises(contains="unsupported estimator"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.schema_version += 1
    with assert_raises(contains="unsupported estimator"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.manifest.append(0)
    with assert_raises(contains="trailing data"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    _ = bad.manifest.pop()
    with assert_raises(contains="truncated"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.payloads[0].name = "unknown"
    with assert_raises(contains="missing array"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.payloads[0].shape = [3, 1]
    with assert_raises(contains="one-dimensional"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    _ = bad.payloads.pop()
    var short_y: List[Float64] = [1, 2]
    var short_shape: List[Int] = [2]
    bad.add_array[DType.float64]("y_thresholds_", short_shape, short_y)
    with assert_raises(contains="threshold lengths"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    _ = bad.payloads.pop()
    var wrong_dtype: List[Float32] = [0, 1, 2]
    bad.add_array[DType.float32]("y_thresholds_", shape, wrong_dtype)
    with assert_raises(contains="dtype"):
        _ = Schema.decode(bad)
    for invalid_y in [[0.0, Float64("inf"), 2.0], [0.0, 3.0, 2.0]]:
        bad = archive.copy()
        _ = bad.payloads.pop()
        bad.add_array[DType.float64]("y_thresholds_", shape, invalid_y)
        with assert_raises(contains="thresholds"):
            _ = Schema.decode(bad)
    for invalid_x in [[0.0, 0.0, 2.0], [-1.0, 1.0, 2.0]]:
        bad = archive.copy()
        _ = bad.payloads.pop(0)
        bad.add_array[DType.float64]("X_thresholds_", shape, invalid_x)
        with assert_raises(contains="threshold"):
            _ = Schema.decode(bad)

    model.y_thresholds_[0] = Float64("nan")
    with assert_raises(contains="finite"):
        _ = Schema.encode(model)


def main() raises:
    round_trips[DType.float64]()
    round_trips[DType.float32]()
    legacy_round_trips[DType.float64]()
    legacy_round_trips[DType.float32]()
    invalid_requests()
    invalid_archives()
    print("Isotonic serialization contracts passed")
