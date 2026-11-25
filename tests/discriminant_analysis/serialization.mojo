"""Archive contracts for discriminant analysis estimators."""
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from numojo import NDArray
from moxie.base import clone
from moxie.discriminant_analysis import ClassLabel, Shrinkage
from moxie.discriminant_analysis_serialization import (
    LinearDiscriminantAnalysisSchema,
    QuadraticDiscriminantAnalysisSchema,
)
from moxie.serialization import dumps, load_model, save_model
from support.fixtures import matrix


def training_data[dtype: DType]() raises -> NDArray[dtype]:
    return matrix[dtype](
        9, 2, [0.0, 0, -2, -2, -2, -1, -1, -1, -1, -2, 1, 3, 1, 2, 2, 1, 2, 2]
    )


def fitted_round_trips[dtype: DType, L: ClassLabel](y: List[L]) raises:
    comptime Schema = LinearDiscriminantAnalysisSchema[dtype, L]
    var path = "builds/discriminant-lda-" + String(dtype) + ".moxie"
    var X = training_data[dtype]()
    for solver in [String("svd"), String("eigen"), String("lsqr")]:
        var model = Schema.Model(solver=solver, n_components=1)
        _ = model.fit(X, y)
        # Changing configuration must leave the saved fitted projection intact.
        _ = model.set_params(
            solver="eigen" if solver != "eigen" else "svd",
            store_covariance=solver == "svd",
            shrinkage=Shrinkage(0.125),
            n_components=Optional[Int](2),
        )
        var archive = Schema.encode(model)
        assert_equal(archive.schema_version, 1)
        save_model[Schema](model, path)
        var restored = load_model[Schema](path)
        assert_true(restored.__sklearn_is_fitted__())
        assert_equal(len(restored.fit_warnings), 0)
        assert_equal(restored.get_n_features_in(), 2)
        assert_equal(restored._fitted_solver, solver)
        assert_equal(restored._max_components, 1)
        assert_equal(restored.solver, model.solver)
        assert_equal(restored.n_components.value(), 2)
        assert_equal(restored.shrinkage.value.value(), 0.125)
        assert_equal(restored.store_covariance, model.store_covariance)
        assert_equal(Bool(restored.covariance_), solver != "svd")
        assert_equal(dumps(Schema.encode(restored)), dumps(archive))
        assert_equal(restored.predict(X), model.predict(X))
        assert_equal(restored.predict_proba(X).tolist(), model.predict_proba(X).tolist())
        assert_equal(restored.predict_log_proba(X).tolist(), model.predict_log_proba(X).tolist())
        assert_equal(restored.decision_function(X).tolist(), model.decision_function(X).tolist())
        assert_equal(restored.score(X, y), model.score(X, y))
        if solver == "lsqr":
            with assert_raises(contains="model was fitted with"):
                _ = restored.transform(X)
        else:
            assert_equal(restored.transform(X).tolist(), model.transform(X).tolist())
        assert_false(clone(restored).__sklearn_is_fitted__())
        restored.coef_.unsafe_set(0, 12345)
        assert_equal(dumps(Schema.encode(model)), dumps(archive))


def unfitted_and_zero_width() raises:
    comptime Schema = LinearDiscriminantAnalysisSchema[]
    var path = "builds/discriminant-lda-unfitted.moxie"
    var X = training_data[DType.float64]()
    var priors: List[Float64] = [0.2, 0.8]
    var model = Schema.Model(
        solver="lsqr", shrinkage="auto", priors=priors.copy(), tol=1e-5, store_covariance=True
    )
    save_model[Schema](model, path)
    var fresh = load_model[Schema](path)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.solver, "lsqr")
    assert_equal(fresh.shrinkage.text.value(), "auto")
    assert_equal(fresh.priors.value(), priors)
    assert_equal(fresh.tol, 1e-5)
    assert_true(fresh.store_covariance)
    assert_false(Bool(fresh.n_components))
    with assert_raises(contains="NotFittedError"):
        _ = fresh.predict(X)
    var binary_labels: List[Int] = [1, 1, 1, 1, 1, 2, 2, 2, 2]
    _ = fresh.fit(X, binary_labels)
    assert_true(fresh.__sklearn_is_fitted__())

    var one_class = Schema.Model(store_covariance=True)
    var labels = List[Int](length=9, fill=-1)
    _ = one_class.fit(X, labels)
    var restored = Schema.decode(Schema.encode(one_class))
    assert_true(Bool(restored.covariance_))
    assert_equal(restored.scalings_.value().shape.tolist(), [2, 0])
    assert_equal(restored.transform(X).shape.tolist(), [9, 0])
    assert_equal(restored.predict(X), labels)


def invalid_state() raises:
    comptime Schema = LinearDiscriminantAnalysisSchema[]
    var model = Schema.Model()
    var unfitted = Schema.encode(model)
    var shape: List[Int] = [1]
    var values: List[Float64] = [0.0]
    unfitted.add_array[DType.float64]("extra", shape, values)
    with assert_raises(contains="unexpected LDA arrays"):
        _ = Schema.decode(unfitted)
    var labels: List[Int] = [1, 1, 1, 1, 1, 2, 2, 2, 2]
    _ = model.fit(training_data[DType.float64](), labels)
    var archive = Schema.encode(model)
    with assert_raises(contains="dtype mismatch"):
        _ = LinearDiscriminantAnalysisSchema[DType.float32].decode(archive)
    with assert_raises(contains="label type mismatch"):
        _ = LinearDiscriminantAnalysisSchema[DType.float64, String].decode(archive)
    var bad = archive.copy()
    bad.schema_version = Schema.StateVersion + 1
    with assert_raises(contains="unsupported estimator"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.manifest.append(0)
    with assert_raises(contains="trailing data"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.payloads[0].shape = [2, 1]
    with assert_raises(contains="invalid shape for coef_"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.payloads[0].name = "unknown"
    with assert_raises(contains="missing array coef_"):
        _ = Schema.decode(bad)
    model.classes_[1] = model.classes_[0]
    with assert_raises(contains="strictly ascending"):
        _ = Schema.encode(model)
    model.classes_[1] = 2
    model.xbar_ = None
    with assert_raises(contains="missing fitted svd center"):
        _ = Schema.encode(model)


def lda_fit_warnings_round_trip() raises:
    comptime Schema = LinearDiscriminantAnalysisSchema[]
    var X = training_data[DType.float64]()
    var labels: List[Int] = [1, 1, 1, 1, 1, 2, 2, 2, 2]
    var unnormalized: List[Float64] = [1.0, 3.0]
    var model = Schema.Model(solver="lsqr", priors=unnormalized^)
    _ = model.fit(X, labels)
    model.priors = None
    var path = "builds/discriminant-lda-fit-warning.moxie"
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(len(restored.fit_warnings), 1)
    assert_equal(restored.fit_warnings[0].category, "UserWarning")
    assert_equal(restored.fit_warnings[0].message, "The priors do not sum to 1. Renormalizing")
    assert_equal(restored.predict_proba(X).tolist(), model.predict_proba(X).tolist())

    var archive = Schema.encode(model)
    assert_equal(archive.schema_version, 2)
    assert_equal(dumps(Schema.encode(restored)), dumps(archive))
    # BinaryWriter encodes the trailing warning marker as a little-endian UInt64.
    archive.manifest[len(archive.manifest) - 8] = 0
    with assert_raises(contains="invalid LDA fit_warnings"):
        _ = Schema.decode(archive)
    restored.fit_warnings[0].message = "other diagnostic"
    with assert_raises(contains="invalid LDA fit_warnings"):
        _ = Schema.encode(restored)

    # Failed fits reset the fitted arrays while retaining the last successful warning.
    with assert_raises(contains="inconsistent numbers of samples"):
        _ = model.fit(X, List[Int]())
    restored = Schema.decode(Schema.encode(model))
    assert_false(restored.__sklearn_is_fitted__())
    assert_equal(len(restored.fit_warnings), 1)
    _ = restored.fit(X, labels)
    assert_equal(len(restored.fit_warnings), 0)
    assert_equal(Schema.encode(restored).schema_version, 1)


def qda_fitted_round_trips[dtype: DType, L: ClassLabel](y: List[L]) raises:
    comptime Schema = QuadraticDiscriminantAnalysisSchema[dtype, L]
    var path = "builds/discriminant-qda-" + String(dtype) + ".moxie"
    var X = training_data[dtype]()
    for solver in [String("svd"), String("eigen")]:
        for store_covariance in [False, True]:
            var model = Schema.Model(solver=solver, store_covariance=store_covariance)
            if solver == "svd":
                model.reg_param = 0.25
            else:
                model.shrinkage = Shrinkage(0.125)
            _ = model.fit(X, y)
            _ = model.set_params(
                solver="eigen" if solver == "svd" else "svd",
                store_covariance=not store_covariance,
                reg_param=0.75,
                shrinkage=Shrinkage("auto"),
                tol=0.5,
            )
            var archive = Schema.encode(model)
            save_model[Schema](model, path)
            var restored = load_model[Schema](path)
            assert_true(restored.__sklearn_is_fitted__())
            assert_equal(restored.get_n_features_in(), 2)
            assert_equal(restored.solver, model.solver)
            assert_equal(restored.reg_param, 0.75)
            assert_equal(restored.tol, 0.5)
            assert_equal(restored.shrinkage.text.value(), "auto")
            assert_equal(restored.store_covariance, not store_covariance)
            assert_equal(Bool(restored.covariance_), store_covariance)
            assert_equal(restored.classes_, model.classes_)
            assert_equal(dumps(Schema.encode(restored)), dumps(archive))
            assert_equal(restored.predict(X), model.predict(X))
            assert_equal(restored.predict_proba(X).tolist(), model.predict_proba(X).tolist())
            assert_equal(restored.predict_log_proba(X).tolist(), model.predict_log_proba(X).tolist())
            assert_equal(restored.decision_function(X).tolist(), model.decision_function(X).tolist())
            assert_equal(restored.score(X, y), model.score(X, y))
            assert_false(clone(restored).__sklearn_is_fitted__())
            restored.rotations_[0].unsafe_set(0, 10)
            assert_equal(dumps(Schema.encode(model)), dumps(archive))


def qda_unfitted_round_trip() raises:
    comptime Schema = QuadraticDiscriminantAnalysisSchema[]
    var path = "builds/discriminant-qda-unfitted.moxie"
    var priors: List[Float64] = [0.25, 0.75]
    var model = Schema.Model(
        solver="svd", priors=priors.copy(), reg_param=0.2, tol=1e-5, store_covariance=True
    )
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_false(restored.__sklearn_is_fitted__())
    assert_equal(restored.priors.value(), priors)
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))
    var X = matrix[DType.float64](4, 1, [-2.0, -1.0, 1.0, 2.0])
    var y: List[Int] = [1, 1, 2, 2]
    with assert_raises(contains="NotFittedError"):
        _ = restored.predict(X)
    _ = restored.fit(X, y)
    var fitted = Schema.decode(Schema.encode(restored))
    assert_equal(fitted.priors_.tolist(), priors)
    assert_equal(fitted.predict(X), y)


def qda_invalid_state() raises:
    comptime Schema = QuadraticDiscriminantAnalysisSchema[]
    var model = Schema.Model(store_covariance=True)
    var bad = Schema.encode(model)
    var extra_shape: List[Int] = [1]
    var extra_values: List[Float64] = [0.0]
    bad.add_array[DType.float64]("extra", extra_shape, extra_values)
    with assert_raises(contains="unexpected QDA arrays"):
        _ = Schema.decode(bad)

    var labels: List[Int] = [1, 1, 1, 1, 1, 2, 2, 2, 2]
    _ = model.fit(training_data[DType.float64](), labels)
    var archive = Schema.encode(model)
    with assert_raises(contains="dtype mismatch"):
        _ = QuadraticDiscriminantAnalysisSchema[DType.float32].decode(archive)
    with assert_raises(contains="label type mismatch"):
        _ = QuadraticDiscriminantAnalysisSchema[DType.float64, String].decode(archive)
    with assert_raises(contains="unsupported estimator"):
        _ = LinearDiscriminantAnalysisSchema[].decode(archive)
    bad = archive.copy()
    bad.schema_version += 1
    with assert_raises(contains="unsupported estimator"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.manifest.append(0)
    with assert_raises(contains="trailing data"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.payloads[2].shape = [2, 4]
    with assert_raises(contains="invalid shape for rotations_"):
        _ = Schema.decode(bad)
    bad = archive.copy()
    bad.payloads[4].name = "unknown"
    with assert_raises(contains="missing array covariance_"):
        _ = Schema.decode(bad)

    model.classes_[1] = model.classes_[0]
    with assert_raises(contains="strictly ascending"):
        _ = Schema.encode(model)
    model.classes_[1] = 2
    model.scalings_[0].unsafe_set(0, -1)
    with assert_raises(contains="invalid value for scalings_"):
        _ = Schema.encode(model)
    _ = model.scalings_.pop()
    with assert_raises(contains="factor count must match"):
        _ = Schema.encode(model)


def main() raises:
    var binary_labels: List[Int] = [-1, -1, -1, -1, -1, 2, 2, 2, 2]
    var text_labels: List[String] = ["β", "a", "c", "a", "c", "β", "a", "c", "β"]
    fitted_round_trips[DType.float32](binary_labels)
    fitted_round_trips[DType.float64](text_labels)
    unfitted_and_zero_width()
    invalid_state()
    lda_fit_warnings_round_trip()
    print("PASS: LDA serialization contracts")
    qda_fitted_round_trips[DType.float32](binary_labels)
    qda_fitted_round_trips[DType.float64](text_labels)
    qda_unfitted_round_trip()
    qda_invalid_state()
    print("PASS: QDA serialization contracts")
