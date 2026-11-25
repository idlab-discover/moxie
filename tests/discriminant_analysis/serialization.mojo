"""Archive contracts for discriminant analysis estimators."""
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from numojo import NDArray
from moxie.base import clone
from moxie.discriminant_analysis import ClassLabel, Shrinkage
from moxie.discriminant_analysis_serialization import LinearDiscriminantAnalysisSchema
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
        save_model[Schema](model, path)
        var restored = load_model[Schema](path)
        assert_true(restored.__sklearn_is_fitted__())
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
    bad.schema_version += 1
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


def main() raises:
    var binary_labels: List[Int] = [-1, -1, -1, -1, -1, 2, 2, 2, 2]
    var text_labels: List[String] = ["β", "a", "c", "a", "c", "β", "a", "c", "β"]
    fitted_round_trips[DType.float32](binary_labels)
    fitted_round_trips[DType.float64](text_labels)
    unfitted_and_zero_width()
    invalid_state()
    print("PASS: LDA serialization contracts")
