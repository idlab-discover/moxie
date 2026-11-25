import numojo as nm
from std.math import inf
from std.testing import assert_equal, assert_raises, assert_true
from moxie.base import clone
from moxie.metadata_routing import FitScoreMetadataRequest
from moxie.serialization import ModelArchive, dumps, loads, load_model, save_model
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


def _check_requests_unset(requests: FitScoreMetadataRequest) raises:
    var metadata = Dict[String, Int]()
    metadata["sample_weight"] = 7
    with assert_raises(contains="UnsetMetadataPassedError"):
        _ = requests.fit.route(metadata)
    with assert_raises(contains="UnsetMetadataPassedError"):
        _ = requests.score.route(metadata)


def _without_v3_names(mut archive: ModelArchive):
    # Only the optional-name presence flag is stored when names are absent.
    for _ in range(8):
        _ = archive.manifest.pop()
    archive.schema_version = 2


def _as_version_one(mut archive: ModelArchive):
    _without_v3_names(archive)
    # The v2 extension is two unset requests, each stored as one 64-bit tag.
    for _ in range(16):
        _ = archive.manifest.pop()
    archive.schema_version = 1


def _check_classifier(X: Dense) raises:
    comptime Schema = LinearSVCSchema[String]
    var y: List[String] = ["z", "z", "a", "a", "β\x00", "β\x00"]
    var weight_labels: List[String] = ["β\x00", "z"]
    var weights: List[Float64] = [1.5, 2.0]
    var model = LinearSVC[String](
        class_weight=ClassWeight[String](weight_labels, weights), tol=1e-7, C=0.5,
    )
    var path = "builds/linear-svc.moxie"
    _check_requests_unset(model.get_metadata_routing())
    var legacy = Schema.encode(model)
    _as_version_one(legacy)
    _check_requests_unset(Schema.decode(legacy).get_metadata_routing())

    _ = model.set_fit_request(sample_weight="training_weight").set_score_request(
        sample_weight=False
    )
    _ = model.set_fit_request()
    with assert_raises(contains="metadata alias"):
        _ = model.set_fit_request(sample_weight="invalid-name")
    var metadata = Dict[String, Int]()
    metadata["training_weight"] = 7
    metadata["sample_weight"] = 3
    save_model[Schema](model, path)
    var unfitted = load_model[Schema](path)
    assert_true(not unfitted.__sklearn_is_fitted__())
    var requests = unfitted.get_metadata_routing()
    assert_equal(requests.fit.route(metadata).value(), 7)
    assert_true(not requests.score.route(metadata))
    requests.fit.set(False)
    assert_equal(unfitted.get_metadata_routing().fit.route(metadata).value(), 7)
    assert_equal(len(unfitted.fit_warnings), 0)
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

    var names: List[String] = ["height", "β\x00"]
    var wrong_names: List[String] = ["β\x00", "height"]
    var short_names: List[String] = ["height"]
    _ = model.fit(X, y, feature_names=names)
    var copied = model.copy()
    names[0] = "mutated"
    assert_equal(model.feature_names_in_.value()[0], "height")
    assert_equal(copied.feature_names_in_.value()[0], "height")
    names[0] = "height"
    with assert_raises(contains="must match X columns"):
        _ = model.fit(X, y, feature_names=short_names)
    with assert_raises(contains="target length"):
        _ = model.fit(X, List[String](), feature_names=wrong_names)
    assert_true(model.feature_names_in_.value() == names)
    _ = model.set_params(C=0.25, fit_intercept=False, max_iter=1)
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(restored.C, 0.25)
    assert_true(not restored.fit_intercept)
    assert_equal(restored.max_iter, 1)
    assert_equal(restored.n_features_in_, 2)
    assert_equal(restored.n_iter_, model.n_iter_)
    assert_equal(restored.fit_status_, model.fit_status_)
    assert_equal(len(restored.fit_warnings), 0)
    var expected_classes: List[String] = ["a", "z", "β\x00"]
    assert_true(restored.classes_ == expected_classes)
    assert_true(restored.feature_names_in_.value() == names)
    assert_true(restored.predict(X, feature_names=names) == model.predict(X))
    with assert_raises(contains="same order"):
        _ = restored.predict(X, feature_names=wrong_names)
    with assert_raises(contains="same order"):
        _ = restored.decision_function(X, feature_names=wrong_names)
    with assert_raises(contains="same order"):
        _ = restored.score(X, y, feature_names=wrong_names)
    assert_equal(restored.score(X, y), model.score(X, y))
    _equal_array(restored.decision_function(X), model.decision_function(X))
    _equal_array(restored.intercept_, model.intercept_)
    var cloned = clone(restored)
    assert_true(not cloned.__sklearn_is_fitted__())
    assert_true(not cloned.feature_names_in_)
    assert_equal(cloned.get_metadata_routing().fit.route(metadata).value(), 7)
    assert_true(not cloned.get_metadata_routing().score.route(metadata))
    _ = cloned.set_fit_request(sample_weight=None)
    with assert_raises(contains="UnsetMetadataPassedError"):
        _ = cloned.get_metadata_routing().fit.route(metadata)
    assert_equal(restored.get_metadata_routing().fit.route(metadata).value(), 7)
    assert_true(dumps(Schema.encode(restored)) == dumps(Schema.encode(model)))
    restored.coef_.unsafe_set(0, 999.0)
    assert_true(model.coef_.unsafe_get(0) != 999.0)

    # A later fit uses the saved configuration and resumes the owned RNG stream.
    _ = restored.fit(X, y)
    assert_true(not restored.feature_names_in_)
    _ = model.fit(X, y)
    assert_true(dumps(Schema.encode(restored)) == dumps(Schema.encode(model)))
    assert_equal(len(model.fit_warnings), 1)
    assert_equal(len(restored.fit_warnings), 1)
    assert_equal(restored.fit_warnings[0].category, "ConvergenceWarning")
    var fitted_copy = model.copy()
    fitted_copy.fit_warnings.clear()
    assert_equal(len(model.fit_warnings), 1)
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


def _check_regressor_warnings(X: Dense, y: Dense) raises:
    var model = LinearSVR(max_iter=1, random_state=9)
    assert_equal(len(model.fit_warnings), 0)
    _ = model.fit(X, y)
    assert_equal(len(model.fit_warnings), 1)
    assert_equal(model.fit_warnings[0].category, "ConvergenceWarning")
    assert_equal(len(clone(model).fit_warnings), 0)

    _ = model.set_params(max_iter=10000, C=-1.0)
    with assert_raises():
        _ = model.fit(X, y)
    assert_equal(len(model.fit_warnings), 1)
    _ = model.set_params(C=1.0)
    var restored = LinearSVRSchema.decode(loads(dumps(LinearSVRSchema.encode(model))))
    assert_equal(restored.max_iter, 10000)
    assert_equal(len(restored.fit_warnings), 1)
    assert_equal(restored.fit_warnings[0].message, model.fit_warnings[0].message)
    _ = model.fit(X, y)
    assert_equal(len(model.fit_warnings), 0)
    assert_equal(len(restored.fit_warnings), 1)


def _check_regressor(X: Dense) raises:
    var y = nm.zeros[DType.float64](nm.Shape(6))
    for i in range(6):
        y.unsafe_set(i, Float64(i) / 3 - 1)
    _check_regressor_warnings(X, y)
    var model = LinearSVR(epsilon=0.05, random_state=9, intercept_scaling=2.0)
    var path = "builds/linear-svr.moxie"
    var legacy = LinearSVRSchema.encode(model)
    _as_version_one(legacy)
    _check_requests_unset(LinearSVRSchema.decode(legacy).get_metadata_routing())
    var malformed = LinearSVRSchema.encode(model)
    _without_v3_names(malformed)
    # The final v2 tag is the score request. Four is not a recognized tag.
    malformed.manifest[len(malformed.manifest) - 8] = 4
    with assert_raises(contains="metadata request tag"):
        _ = LinearSVRSchema.decode(malformed)

    _ = model.set_fit_request(sample_weight=True).set_score_request(
        sample_weight="evaluation_weight"
    )
    var metadata = Dict[String, Int]()
    metadata["sample_weight"] = 3
    metadata["evaluation_weight"] = 7
    save_model[LinearSVRSchema](model, path)
    var unfitted = load_model[LinearSVRSchema](path)
    assert_true(not unfitted.__sklearn_is_fitted__())
    assert_equal(unfitted.get_metadata_routing().fit.route(metadata).value(), 3)
    assert_equal(unfitted.get_metadata_routing().score.route(metadata).value(), 7)
    assert_equal(unfitted.epsilon, 0.05)
    assert_equal(unfitted.intercept_scaling, 2.0)
    with assert_raises():
        _ = unfitted.predict(X)

    var legacy_v2 = LinearSVRSchema.encode(model)
    _without_v3_names(legacy_v2)
    var restored_v2 = LinearSVRSchema.decode(legacy_v2)
    assert_true(not restored_v2.feature_names_in_)
    assert_equal(restored_v2.get_metadata_routing().score.route(metadata).value(), 7)

    var names: List[String] = ["same name", "same name"]
    var wrong_names: List[String] = ["same name", "other"]
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(X, feature_names=names)
    _ = model.fit(
        X.astype[DType.float32](), y.astype[DType.float32](), feature_names=names
    )
    _ = model.set_params(epsilon=0.4, fit_intercept=False)
    save_model[LinearSVRSchema](model, path)
    var restored = load_model[LinearSVRSchema](path)
    assert_equal(restored.epsilon, 0.4)
    assert_true(not restored.fit_intercept)
    assert_equal(restored.n_features_in_, 2)
    assert_equal(restored.n_iter_, model.n_iter_)
    assert_equal(restored.fit_status_, model.fit_status_)
    assert_true(restored.feature_names_in_.value() == names)
    _equal_array(restored.predict(X, feature_names=names), model.predict(X))
    assert_equal(restored.score(X, y, feature_names=names), model.score(X, y))
    with assert_raises(contains="same order"):
        _ = restored.score(X, y, feature_names=wrong_names)
    _equal_array(restored.predict(X.astype[DType.float32]()), model.predict(X.astype[DType.float32]()))
    assert_equal(restored.score(X, y), model.score(X, y))
    var cloned = clone(restored)
    assert_true(not cloned.__sklearn_is_fitted__())
    assert_true(not cloned.feature_names_in_)
    assert_equal(cloned.get_metadata_routing().score.route(metadata).value(), 7)
    _ = cloned.set_score_request(sample_weight=False)
    assert_true(not cloned.get_metadata_routing().score.route(metadata))
    assert_equal(restored.get_metadata_routing().score.route(metadata).value(), 7)

    var backing = nm.zeros[DType.float64](nm.Shape(4))
    backing.unsafe_set(3, model.coef_.unsafe_get(0))
    backing.unsafe_set(1, model.coef_.unsafe_get(1))
    restored.coef_ = backing.view_with_layout(
        nm.Shape(2), nm.NDArrayStrides(-2), 3,
    )
    var materialized = LinearSVRSchema.decode(LinearSVRSchema.encode(restored))
    _equal_array(materialized.coef_, model.coef_)

    var archive = LinearSVRSchema.encode(model)
    var truncated = archive.copy()
    _ = truncated.manifest.pop()
    with assert_raises():
        _ = LinearSVRSchema.decode(truncated)
    var malformed_names = LinearSVRSchema.decode(archive)
    malformed_names.feature_names_in_ = List[String]()
    with assert_raises(contains="feature names count"):
        _ = LinearSVRSchema.encode(malformed_names)
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


def _check_named_numeric_target(X: Dense) raises:
    var y = nm.zeros[DType.int64](nm.Shape(6))
    for i in range(3, 6):
        y.unsafe_set(i, 1)
    var names: List[String] = ["height", "width"]
    var model = LinearSVC(random_state=0)
    var fitted = model.fit(X, y, feature_names=names).predict(
        X.astype[DType.float32](), feature_names=names
    )
    assert_true(fitted == model.predict(X))
    assert_true(model.feature_names_in_.value() == names)


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(6, 2))
    for i in range(6):
        X.unsafe_set(i * 2, Float64(i) - 2.5)
        X.unsafe_set(i * 2 + 1, Float64(i % 2))
    _check_named_numeric_target(X)
    _check_classifier(X)
    _check_regressor(X)
    _check_label_type[Int](X, [1, 1, 1, 2, 2, 2])
    _check_label_type[Int32](X, [1, 1, 1, 2, 2, 2])
    _check_label_type[Int64](X, [1, 1, 1, 2, 2, 2])
    _check_label_type[Float32](X, [1, 1, 1, 2, 2, 2])
    _check_label_type[Float64](X, [1, 1, 1, 2, 2, 2])
    print("PASS LinearSVC and LinearSVR serialization")
