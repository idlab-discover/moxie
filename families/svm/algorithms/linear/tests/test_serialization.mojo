import numojo as nm
from std.math import inf
from std.testing import assert_equal, assert_raises, assert_true
from moxie.base import clone
from moxie.serialization import dumps, loads, load_model, save_model
from moxie.svm import ClassWeight, LinearSVC, LinearSVR
from moxie.svm._contracts import Dense
from moxie.svm.linear_serialization import LinearSVCSchema, LinearSVRSchema


def _equal_array(actual: Dense, expected: Dense) raises:
    assert_equal(actual.shape, expected.shape)
    for i in range(expected.size):
        assert_equal(actual.item(i), expected.item(i))


def _check_label_type[Label: Copyable & Comparable & Deinitable](
    X: Dense, y: List[Label],
) raises:
    comptime Schema = LinearSVCSchema[Label]
    var model = LinearSVC[Label](random_state=10)
    _ = model.fit(X, y)
    var restored = Schema.decode(loads(dumps(Schema.encode(model))))
    assert_true(restored.classes_ == model.classes_)
    assert_true(restored.predict(X) == model.predict(X))
    _equal_array(restored.decision_function(X), model.decision_function(X))


def _check_classifier(X: Dense) raises:
    comptime Schema = LinearSVCSchema[String]
    var y: List[String] = ["z", "z", "a", "a", "β\x00", "β\x00"]
    var weight_labels: List[String] = ["β\x00", "z"]
    var weights: List[Float64] = [1.5, 2.0]
    var model = LinearSVC[String](
        class_weight=ClassWeight[String](weight_labels, weights), tol=1e-7, C=0.5,
    )
    var path = "builds/linear-svc.moxie"
    save_model[Schema](model, path)
    var unfitted = load_model[Schema](path)
    assert_true(not unfitted.__sklearn_is_fitted__())
    assert_true(unfitted.class_weight.labels == weight_labels)
    assert_true(unfitted.class_weight.values == weights)
    assert_equal(unfitted.C, 0.5)
    assert_equal(unfitted.random_state, -1)
    with assert_raises():
        _ = unfitted.predict(X)
    var empty = Schema.encode(unfitted)
    var shape: List[Int] = [1]
    var values: List[Float64] = [1.0]
    empty.add_array[DType.float64]("coef_", shape, values)
    with assert_raises():
        _ = Schema.decode(empty)

    _ = model.fit(X, y)
    _ = model.set_params(C=0.25, fit_intercept=False, max_iter=1)
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(restored.C, 0.25)
    assert_true(not restored.fit_intercept)
    assert_equal(restored.max_iter, 1)
    assert_equal(restored.n_features_in_, 2)
    assert_equal(restored.n_iter_, model.n_iter_)
    assert_equal(restored.fit_status_, model.fit_status_)
    var expected_classes: List[String] = ["a", "z", "β\x00"]
    assert_true(restored.classes_ == expected_classes)
    assert_true(restored.predict(X) == model.predict(X))
    assert_equal(restored.score(X, y), model.score(X, y))
    _equal_array(restored.decision_function(X), model.decision_function(X))
    _equal_array(restored.intercept_, model.intercept_)
    assert_true(not clone(restored).__sklearn_is_fitted__())
    assert_true(dumps(Schema.encode(restored)) == dumps(Schema.encode(model)))
    restored.coef_.unsafe_set(0, 999.0)
    assert_true(model.coef_.unsafe_get(0) != 999.0)

    # A later fit uses the saved configuration and resumes the owned RNG stream.
    _ = restored.fit(X, y)
    _ = model.fit(X, y)
    assert_true(dumps(Schema.encode(restored)) == dumps(Schema.encode(model)))
    var archive = Schema.encode(model)
    with assert_raises():
        _ = LinearSVCSchema[Float64].decode(archive)
    with assert_raises():
        _ = LinearSVRSchema.decode(archive)
    archive.manifest.append(0)
    with assert_raises():
        _ = Schema.decode(loads(dumps(archive)))
    model.classes_.append("a")
    with assert_raises():
        _ = Schema.encode(model)


def _check_regressor(X: Dense) raises:
    var y = nm.zeros[DType.float64](nm.Shape(6))
    for i in range(6):
        y.unsafe_set(i, Float64(i) / 3 - 1)
    var model = LinearSVR(epsilon=0.05, random_state=9, intercept_scaling=2.0)
    var path = "builds/linear-svr.moxie"
    save_model[LinearSVRSchema](model, path)
    var unfitted = load_model[LinearSVRSchema](path)
    assert_true(not unfitted.__sklearn_is_fitted__())
    assert_equal(unfitted.epsilon, 0.05)
    assert_equal(unfitted.intercept_scaling, 2.0)
    with assert_raises():
        _ = unfitted.predict(X)

    _ = model.fit(X.astype[DType.float32](), y)
    _ = model.set_params(epsilon=0.4, fit_intercept=False)
    save_model[LinearSVRSchema](model, path)
    var restored = load_model[LinearSVRSchema](path)
    assert_equal(restored.epsilon, 0.4)
    assert_true(not restored.fit_intercept)
    assert_equal(restored.n_features_in_, 2)
    assert_equal(restored.n_iter_, model.n_iter_)
    assert_equal(restored.fit_status_, model.fit_status_)
    _equal_array(restored.predict(X), model.predict(X))
    _equal_array(restored.predict(X.astype[DType.float32]()), model.predict(X.astype[DType.float32]()))
    assert_equal(restored.score(X, y), model.score(X, y))
    assert_true(not clone(restored).__sklearn_is_fitted__())

    var backing = nm.zeros[DType.float64](nm.Shape(4))
    backing.unsafe_set(3, model.coef_.unsafe_get(0))
    backing.unsafe_set(1, model.coef_.unsafe_get(1))
    restored.coef_ = backing.view_with_layout(
        nm.Shape(2), nm.NDArrayStrides(-2), 3,
    )
    var materialized = LinearSVRSchema.decode(LinearSVRSchema.encode(restored))
    _equal_array(materialized.coef_, model.coef_)

    var archive = LinearSVRSchema.encode(model)
    archive.schema_version += 1
    with assert_raises():
        _ = LinearSVRSchema.decode(loads(dumps(archive)))
    archive.schema_version -= 1
    archive.payloads[0].shape = [1, 2]
    with assert_raises():
        _ = LinearSVRSchema.decode(loads(dumps(archive)))
    archive.payloads[0].shape = [2]
    archive.payloads[0].name = "unknown"
    with assert_raises():
        _ = LinearSVRSchema.decode(loads(dumps(archive)))
    archive.payloads[0].name = "coef_"
    archive.payloads[0].dtype = 4
    with assert_raises():
        _ = LinearSVRSchema.decode(loads(dumps(archive)))
    model.coef_.unsafe_set(0, inf[DType.float64]())
    with assert_raises():
        _ = LinearSVRSchema.encode(model)


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(6, 2))
    for i in range(6):
        X.unsafe_set(i * 2, Float64(i) - 2.5)
        X.unsafe_set(i * 2 + 1, Float64(i % 2))
    _check_classifier(X)
    _check_regressor(X)
    _check_label_type[Int](X, [1, 1, 1, 2, 2, 2])
    _check_label_type[Int32](X, [1, 1, 1, 2, 2, 2])
    _check_label_type[Int64](X, [1, 1, 1, 2, 2, 2])
    _check_label_type[Float32](X, [1, 1, 1, 2, 2, 2])
    _check_label_type[Float64](X, [1, 1, 1, 2, 2, 2])
    print("PASS LinearSVC and LinearSVR serialization")
