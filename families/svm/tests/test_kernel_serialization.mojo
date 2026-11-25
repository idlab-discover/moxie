"""Lifecycle and malformed-state contracts for native dense kernel archives."""

import numojo as nm
from std.math import inf
from std.testing import assert_equal, assert_raises, assert_true
from moxie.base import clone
from moxie.serialization import ModelArchive, ModelSchema, dumps, load_model, loads, save_model
from moxie.svm import ClassWeight, Gamma, NuSVC, NuSVR, OneClassSVM, Probability, SVC, SVR
from moxie.svm._contracts import Dense, KernelProvider
from moxie.svm.kernel_serialization import (
    NuSVCSchema, NuSVRSchema, OneClassSVMSchema, SVCSchema, SVRSchema,
)


def _features() raises -> Dense:
    var values: List[Float64] = [-3, -1, -2, -1, -1, -2, 1, 1, 2, 2, 3, 1]
    var storage = nm.zeros[DType.float64](nm.Shape(6, 3))
    for i in range(6):
        storage.unsafe_set(3 * i + 1, values[2 * i])
        storage.unsafe_set(3 * i + 2, values[2 * i + 1])
    return storage.view_with_layout(nm.Shape(6, 2), storage.strides, 1)


def _equal_array[dtype: DType](
    actual: nm.NDArray[dtype], expected: nm.NDArray[dtype],
) raises:
    assert_equal(actual.shape, expected.shape)
    for i in range(actual.size):
        assert_equal(actual.item(i), expected.item(i))


def _round_trip[Schema: ModelSchema](model: Schema.Model) raises -> Schema.Model where conforms_to(Schema.Model, Deinitable):
    var path = "builds/kernel-svm.moxie"
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_true(dumps(Schema.encode(restored)) == dumps(Schema.encode(model)))
    return restored^


struct LinearProvider(KernelProvider):
    def __init__(out self):
        pass

    def evaluate(self, x: Dense, train: Dense) raises -> Dense:
        return x @ train.T()


def _check_classifier(X: Dense) raises:
    comptime Schema = SVCSchema[String]
    var y: List[String] = ["z", "z", "z", "β\x00", "β\x00", "β\x00"]
    var labels: List[String] = ["z"]
    var weights: List[Float64] = [1.5]
    var model = SVC[String](
        kernel="linear", C=2.0, probability=True, tol=1e-7,
        class_weight=ClassWeight[String](labels, weights),
    )
    var unfitted = _round_trip[Schema](model)
    assert_true(not unfitted.__sklearn_is_fitted__())
    assert_true(unfitted.class_weight.labels == labels)
    assert_true(unfitted.class_weight.values == weights)
    assert_equal(unfitted.C, 2.0)
    with assert_raises():
        _ = unfitted.predict(X)

    _ = model.fit(X, y)
    _ = model.set_params(kernel="rbf", C=0.75, gamma=Gamma(0.4), probability=Probability(False))
    var restored = _round_trip[Schema](model)
    assert_true(restored.__sklearn_is_fitted__())
    assert_equal(restored.kernel, String("rbf"))
    assert_equal(restored.gamma.value, 0.4)
    assert_equal(restored.C, 0.75)
    assert_equal(restored.shape_fit_, nm.Shape(6, 2))
    assert_true(not clone(restored).__sklearn_is_fitted__())
    assert_true(restored.classes_ == model.classes_)
    assert_true(restored.predict(X) == model.predict(X))
    _equal_array(restored.decision_function(X), model.decision_function(X))
    _equal_array(restored.support_vectors_, model.support_vectors_)
    _equal_array(restored.dual_coef_, model.dual_coef_)
    _equal_array(restored.probA_, model.probA_)
    with assert_raises():
        _ = restored.predict_proba(X)
    _ = restored.set_params(probability=Probability(True))
    _ = model.set_params(probability=Probability(True))
    _equal_array(restored.predict_proba(X), model.predict_proba(X))
    _equal_array(restored.predict_log_proba(X), model.predict_log_proba(X))
    restored.dual_coef_.unsafe_set(0, 999.0)
    assert_true(model.dual_coef_.unsafe_get(0) != 999.0)
    _equal_array(restored.decision_function(X), model.decision_function(X))
    _ = restored.fit(X, y)
    _ = model.fit(X, y)
    assert_true(dumps(Schema.encode(restored)) == dumps(Schema.encode(model)))


def _check_other_unfitted(X: Dense) raises:
    var svc = _round_trip[NuSVCSchema[]](NuSVC(nu=0.4, random_state=2026))
    var svr = _round_trip[SVRSchema](SVR(epsilon=0.2, max_iter=3))
    var nsvr = _round_trip[NuSVRSchema](NuSVR(C=3.0, nu=0.2))
    var one_class = _round_trip[OneClassSVMSchema](OneClassSVM(nu=0.7, kernel="sigmoid"))
    assert_equal(svc.random_state, 2026)
    assert_equal(svr.epsilon, 0.2)
    assert_equal(nsvr.C, 3.0)
    assert_equal(one_class.nu, 0.7)
    assert_true(not svc.__sklearn_is_fitted__())
    assert_true(not svr.__sklearn_is_fitted__())
    assert_true(not nsvr.__sklearn_is_fitted__())
    assert_true(not one_class.__sklearn_is_fitted__())
    with assert_raises():
        _ = svc.predict(X)
    with assert_raises():
        _ = svr.predict(X)
    with assert_raises():
        _ = nsvr.predict(X)
    with assert_raises():
        _ = one_class.predict(X)


def _check_fit_warnings(X: Dense) raises:
    comptime Schema = SVCSchema[]
    var y: List[Float64] = [1, 1, 1, 2, 2, 2]
    var model = SVC(max_iter=1, probability=False)
    assert_equal(len(model.fit_warnings), 0)
    _ = model.fit(X, y)
    assert_equal(len(model.fit_warnings), 2)
    assert_equal(model.fit_warnings[0].category, String("FutureWarning"))
    assert_equal(model.fit_warnings[1].category, String("ConvergenceWarning"))
    assert_equal(len(clone(model).fit_warnings), 0)
    var fitted_copy = model.copy()
    _ = fitted_copy.fit_warnings.pop()
    assert_equal(len(model.fit_warnings), 2)

    _ = model.set_params(max_iter=-1, probability=Probability("deprecated"))
    with assert_raises():
        _ = model.fit(X, List[Float64]())
    assert_equal(len(model.fit_warnings), 2)
    var restored = _round_trip[Schema](model)
    assert_equal(restored.max_iter, -1)
    assert_equal(restored.probability.mode, String("deprecated"))
    assert_equal(len(restored.fit_warnings), 2)
    for i in range(2):
        assert_equal(restored.fit_warnings[i].category, model.fit_warnings[i].category)
        assert_equal(restored.fit_warnings[i].message, model.fit_warnings[i].message)
    assert_true("max_iter=1" in restored.fit_warnings[1].message)

    var archive = Schema.encode(model)
    # Version 2 ended before the two unset request tags.
    archive.schema_version = 2
    for _ in range(16):
        _ = archive.manifest.pop()
    var v2 = Schema.decode(loads(dumps(archive)))
    assert_equal(len(v2.fit_warnings), 2)
    assert_true(v2.get_metadata_routing().fit.value().isa[NoneType]())
    assert_true(v2.get_metadata_routing().score.value().isa[NoneType]())
    var invalid = archive.copy()
    invalid.manifest[len(invalid.manifest) - 8] = 2
    with assert_raises():
        _ = Schema.decode(loads(dumps(invalid)))
    invalid = archive.copy()
    # The stored cap is one. A cap of two contradicts this uncalibrated fit.
    invalid.manifest[len(invalid.manifest) - 16] = 2
    with assert_raises():
        _ = Schema.decode(loads(dumps(invalid)))

    # Version 1 ended before the three 64-bit diagnostics fields.
    archive.schema_version = 1
    for _ in range(24):
        _ = archive.manifest.pop()
    var legacy = Schema.decode(loads(dumps(archive)))
    assert_equal(len(legacy.fit_warnings), 0)
    assert_true(legacy.get_metadata_routing().fit.value().isa[NoneType]())
    assert_true(legacy.get_metadata_routing().score.value().isa[NoneType]())
    assert_equal(legacy.fit_status_, restored.fit_status_)
    assert_true(legacy.predict(X) == restored.predict(X))
    var upgraded = _round_trip[Schema](legacy)
    assert_equal(len(upgraded.fit_warnings), 0)

    _ = restored.fit(X, y)
    assert_equal(len(restored.fit_warnings), 0)


def _check_requests() raises:
    comptime Schema = SVCSchema[]
    var model = SVC()
    var metadata = Dict[String, Int]()
    metadata["sample_weight"] = 3
    metadata["training_weight"] = 7
    with assert_raises(contains="UnsetMetadataPassedError"):
        _ = model.get_metadata_routing().fit.route(metadata)

    _ = model.set_fit_request(sample_weight="training_weight").set_score_request(
        sample_weight=False
    )
    _ = model.set_fit_request()
    with assert_raises(contains="ASCII identifier"):
        _ = model.set_fit_request(sample_weight="invalid-name")

    var restored = _round_trip[Schema](model)
    assert_true(not restored.__sklearn_is_fitted__())
    var routing = restored.get_metadata_routing()
    assert_equal(routing.fit.route(metadata).or_else(-1), 7)
    assert_true(not routing.score.route(metadata))
    routing.fit.set(False)
    var cloned = clone(restored)
    assert_equal(cloned.get_metadata_routing().fit.route(metadata).or_else(-1), 7)
    _ = cloned.set_fit_request(sample_weight=None).set_score_request(
        sample_weight="sample_weight"
    )
    with assert_raises(contains="UnsetMetadataPassedError"):
        _ = cloned.get_metadata_routing().fit.route(metadata)
    assert_equal(cloned.get_metadata_routing().score.route(metadata).or_else(-1), 3)
    assert_equal(restored.get_metadata_routing().fit.route(metadata).or_else(-1), 7)
    assert_true(not restored.get_metadata_routing().score.route(metadata))

    var invalid = Schema.encode(model)
    # Score is False, stored as the final 64-bit tag. Tag four is invalid.
    invalid.manifest[len(invalid.manifest) - 8] = 4
    with assert_raises(contains="metadata request tag"):
        _ = Schema.decode(loads(dumps(invalid)))


def _check_callable(X: Dense) raises:
    var provider = LinearProvider()
    var labels: List[Int32] = [1, 1, 1, 2, 2, 2]
    var target = nm.zeros[DType.float64](nm.Shape(6))
    for i in range(6):
        target.unsafe_set(i, Float64(labels[i]))
    var x32 = X.astype[DType.float32]()
    var svc = SVC[Int32](kernel="callable", probability=True, random_state=42)
    var nsvc = NuSVC[Int32](kernel="callable", probability=True, random_state=42)
    var svr = SVR(kernel="callable")
    var nsvr = NuSVR(kernel="callable")
    var one_class = OneClassSVM(kernel="callable")
    _ = svc.set_fit_request(sample_weight=True)
    _ = nsvc.set_fit_request(sample_weight=True)
    _ = svr.set_fit_request(sample_weight=True)
    _ = nsvr.set_fit_request(sample_weight=True)
    _ = one_class.set_fit_request(sample_weight=True)
    _ = svc.fit_with_kernel(X, labels, provider)
    _ = nsvc.fit_with_kernel(X, labels, provider)
    _ = svr.fit_with_kernel(x32, target, provider)
    _ = nsvr.fit_with_kernel(x32, target, provider)
    _ = one_class.fit_with_kernel(X, provider)
    var restored_svc = _round_trip[SVCSchema[Int32]](svc)
    var restored_nsvc = _round_trip[NuSVCSchema[Int32]](nsvc)
    var restored_svr = _round_trip[SVRSchema](svr)
    var restored_nsvr = _round_trip[NuSVRSchema](nsvr)
    var restored_one_class = _round_trip[OneClassSVMSchema](one_class)
    var metadata = Dict[String, Int]()
    metadata["sample_weight"] = 3
    assert_equal(restored_svc.get_metadata_routing().fit.route(metadata).or_else(-1), 3)
    assert_equal(restored_nsvc.get_metadata_routing().fit.route(metadata).or_else(-1), 3)
    assert_equal(restored_svr.get_metadata_routing().fit.route(metadata).or_else(-1), 3)
    assert_equal(restored_nsvr.get_metadata_routing().fit.route(metadata).or_else(-1), 3)
    assert_equal(restored_one_class.get_metadata_routing().route(metadata).or_else(-1), 3)
    assert_equal(len(restored_svc.fit_warnings), 1)
    assert_equal(len(restored_nsvc.fit_warnings), 1)
    assert_equal(restored_svc.fit_warnings[0].category, String("FutureWarning"))
    assert_equal(restored_nsvc.fit_warnings[0].category, String("FutureWarning"))
    assert_true(restored_svc.predict(X, provider) == svc.predict(X, provider))
    assert_true(restored_nsvc.predict(X, provider) == nsvc.predict(X, provider))
    _equal_array(restored_svc.predict_proba(X, provider), svc.predict_proba(X, provider))
    _equal_array(restored_nsvc.predict_proba(X, provider), nsvc.predict_proba(X, provider))
    _equal_array(restored_svr.predict(x32, provider), svr.predict(x32, provider))
    _equal_array(restored_nsvr.predict(x32, provider), nsvr.predict(x32, provider))
    _equal_array(restored_one_class.predict(X, provider), one_class.predict(X, provider))
    _equal_array(restored_one_class.score_samples(X, provider), one_class.score_samples(X, provider))
    with assert_raises():
        _ = restored_svc.predict(X)
    with assert_raises():
        _ = restored_one_class.predict(nm.zeros[DType.float64](nm.Shape(1, 1)), provider)


def _replace_array[dtype: DType](
    mut archive: ModelArchive, name: String, shape: List[Int], values: List[Scalar[dtype]],
) raises:
    var replacement = ModelArchive(archive.type_id.copy(), archive.schema_version)
    replacement.add_array[dtype](name, shape, values)
    for i in range(len(archive.payloads)):
        if archive.payloads[i].name == name:
            archive.payloads[i] = replacement.payloads[0].copy()
            return
    raise Error("fixture array missing")


def _check_invalid(X: Dense) raises:
    comptime Schema = SVCSchema[]
    var labels: List[Float64] = [1, 1, 1, 2, 2, 2]
    var model = SVC(probability=True, random_state=42)
    _ = model.fit(X, labels)
    var original = Schema.encode(model)
    with assert_raises():
        _ = SVCSchema[String].decode(original)
    with assert_raises():
        _ = NuSVCSchema[].decode(original)
    var altered = original.copy()
    altered.schema_version += 1
    with assert_raises():
        _ = Schema.decode(loads(dumps(altered)))
    altered = original.copy()
    altered.manifest.append(0)
    with assert_raises():
        _ = Schema.decode(loads(dumps(altered)))
    altered = original.copy()
    _replace_array[DType.int64](altered, "n_support", [2], [7, 0])
    with assert_raises():
        _ = Schema.decode(loads(dumps(altered)))
    altered = original.copy()
    var support = original.array("support").values[DType.int64]()
    support[0] = 6
    _replace_array[DType.int64](altered, "support", [len(support)], support)
    with assert_raises():
        _ = Schema.decode(loads(dumps(altered)))
    altered = original.copy()
    _replace_array[DType.float64](altered, "prob_b", [0], [])
    with assert_raises():
        _ = Schema.decode(loads(dumps(altered)))
    altered = original.copy()
    _replace_array[DType.float64](altered, "rho", [1], [inf[DType.float64]()])
    with assert_raises():
        _ = Schema.decode(loads(dumps(altered)))
    altered = original.copy()
    altered.payloads[0].name = "unexpected"
    with assert_raises():
        _ = Schema.decode(loads(dumps(altered)))
    var unfitted = Schema.encode(SVC())
    var shape: List[Int] = [1]
    var values: List[Float64] = [1.0]
    unfitted.add_array[DType.float64]("rho", shape, values)
    with assert_raises():
        _ = Schema.decode(loads(dumps(unfitted)))


def _check_empty_support(X: Dense) raises:
    var y = nm.ones[DType.float64](nm.Shape(X.shape[0]))
    var model = SVR(epsilon=1.0)
    _ = model.fit(X, y)
    assert_equal(model.support_.size, 0)
    var restored = _round_trip[SVRSchema](model)
    assert_equal(restored.support_vectors_.shape, nm.Shape(0, 2))
    _equal_array(restored.predict(X), model.predict(X))


def main() raises:
    var X = _features()
    _check_classifier(X)
    _check_other_unfitted(X)
    _check_fit_warnings(X)
    _check_requests()
    _check_callable(X)
    _check_invalid(X)
    _check_empty_support(X)
    print("PASS dense kernel SVM archives, callable inference, and malformed state")
