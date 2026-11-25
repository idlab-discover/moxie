"""Archive contracts for discriminant analysis estimators."""
from std.sys import argv
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from numojo import NDArray, Shape
from moxie.base import clone
from moxie.compose import ColumnTable
from moxie.discriminant_analysis import ClassLabel, Shrinkage
from moxie.metadata_routing import FitScoreMetadataRequestProvider
from moxie.discriminant_analysis_serialization import (
    LinearDiscriminantAnalysisSchema,
    QuadraticDiscriminantAnalysisSchema,
)
from moxie.serialization import ModelSchema, dumps, loads, load_model, save_model
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
    var projected = restored.transform_table(X)
    assert_equal(projected.values().shape.tolist(), [9, 0])
    assert_equal(projected.rows(), 9)
    assert_equal(len(projected.schema().names), 0)
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
    bad.schema_version = 0xFFFF
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
    bad.schema_version = 0xFFFF
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


def named_archive_round_trip[Schema: ModelSchema](
    model: Schema.Model, unfitted: Schema.Model, version: Int,
) raises -> Schema.Model where conforms_to(Schema.Model, Deinitable):
    var archive = Schema.encode(model)
    assert_equal(archive.schema_version, version)
    var restored = Schema.decode(loads(dumps(archive)))
    assert_equal(dumps(Schema.encode(restored)), dumps(archive))
    var truncated = archive.copy()
    _ = truncated.manifest.pop()
    with assert_raises():
        _ = Schema.decode(truncated)
    var bad = Schema.encode(unfitted)
    bad.schema_version = version
    with assert_raises():
        _ = Schema.decode(bad)
    return restored^


def check_unset_request[E: FitScoreMetadataRequestProvider](model: E) raises:
    var requests = model.get_metadata_routing()
    assert_true(requests.fit.value().isa[NoneType]())
    assert_true(requests.score.value().isa[NoneType]())
    with assert_raises(contains="UnsetMetadataPassedError"):
        _ = requests.score.route({"sample_weight": 1})


def check_score_request_archives[Schema: ModelSchema](
    declined: Schema.Model, accepted: Schema.Model, aliased: Schema.Model,
) raises where conforms_to(Schema.Model, FitScoreMetadataRequestProvider & Deinitable):
    var metadata = {"sample_weight": 1, "validation_weight": 2}
    var declined_archive = Schema.encode(declined)
    var restored_declined = Schema.decode(loads(dumps(declined_archive)))
    assert_false(restored_declined.__sklearn_is_fitted__())
    assert_false(Bool(restored_declined.get_metadata_routing().score.route(metadata)))

    var archive = Schema.encode(accepted)
    var restored_accepted = Schema.decode(loads(dumps(archive)))
    assert_false(restored_accepted.__sklearn_is_fitted__())
    assert_equal(restored_accepted.get_metadata_routing().score.route(metadata).value(), 1)
    archive.manifest[0] = UInt8(archive.schema_version)
    with assert_raises(contains="unsupported discriminant schema version"):
        _ = Schema.decode(archive)
    archive = Schema.encode(accepted)
    archive.manifest[len(archive.manifest) - 8] = 0
    with assert_raises(contains="unknown discriminant score request"):
        _ = Schema.decode(archive)
    archive = Schema.encode(aliased)
    archive.manifest[len(archive.manifest) - String("validation_weight").byte_length()] = 32
    with assert_raises(contains="invalid discriminant score request alias"):
        _ = Schema.decode(archive)


def check_cloned_request[E: FitScoreMetadataRequestProvider](
    model: E, cloned: E,
) raises:
    assert_true(model.__sklearn_is_fitted__())
    assert_false(cloned.__sklearn_is_fitted__())
    var metadata = {"validation_weight": 2}
    assert_equal(model.get_metadata_routing().score.route(metadata).value(), 2)
    assert_equal(cloned.get_metadata_routing().score.route(metadata).value(), 2)


def test_score_request_lifecycle_and_archives() raises:
    comptime LDA = LinearDiscriminantAnalysisSchema[]
    comptime QDA = QuadraticDiscriminantAnalysisSchema[]
    var X = training_data[DType.float64]()
    var y: List[Int] = [1, 1, 1, 1, 1, 2, 2, 2, 2]

    var lda = LDA.Model()
    var lda_declined = LDA.Model()
    var lda_accepted = LDA.Model()
    check_unset_request(lda)
    _ = lda_declined.set_score_request(sample_weight=False)
    _ = lda_accepted.set_score_request(sample_weight=String("sample_weight")).set_score_request()
    _ = lda.set_score_request(sample_weight=String("validation_weight")).fit(X, y)
    with assert_raises(contains="ASCII identifier"):
        _ = lda.set_score_request(sample_weight=String("invalid alias"))
    check_cloned_request(lda, clone(lda))
    check_score_request_archives[LDA](lda_declined, lda_accepted, lda)
    var snapshot = lda.get_metadata_routing()
    _ = lda.set_score_request(sample_weight=None)
    check_unset_request(lda)
    assert_equal(snapshot.score.route({"validation_weight": 2}).value(), 2)

    var qda = QDA.Model()
    var qda_declined = QDA.Model()
    var qda_accepted = QDA.Model()
    check_unset_request(qda)
    _ = qda_declined.set_score_request(sample_weight=False)
    _ = qda_accepted.set_score_request(sample_weight=True).set_score_request()
    _ = qda.set_score_request(sample_weight=String("validation_weight")).fit(X, y)
    check_cloned_request(qda, clone(qda))
    check_score_request_archives[QDA](qda_declined, qda_accepted, qda)
    _ = qda.set_score_request(sample_weight=None)
    check_unset_request(qda)


def routed_named_score[
    dtype: DType, E: FitScoreMetadataRequestProvider
](
    model: E, weights: NDArray[dtype]
) raises -> NDArray[dtype]:
    return model.get_metadata_routing().score.route({"validation_weight": weights.copy()}).value().copy()


def named_lda[dtype: DType](emit_parity: Bool) raises:
    comptime Schema = LinearDiscriminantAnalysisSchema[dtype]
    var X = training_data[dtype]()
    var y: List[Int] = [1, 1, 1, 1, 1, 2, 2, 2, 2]
    var names: List[String] = ["mass", "âge"]
    var reordered: List[String] = ["âge", "mass"]
    var table = ColumnTable(X, names)
    var reversed_table = ColumnTable(X, reordered)
    var model = Schema.Model()
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(table)
    with assert_raises(contains="NotFittedError"):
        _ = model.transform_table(table)
    with assert_raises(contains="requires y"):
        _ = model.fit_transform_table(table)
    var projection = model.fit_transform_table(table, y.copy())
    var components: List[String] = ["lineardiscriminantanalysis0"]
    assert_equal(projection.schema().names, components)
    assert_equal(projection.schema().dtypes, [String(dtype)])
    assert_equal(projection.values().shape.tolist(), [9, 1])
    assert_equal(projection.values().tolist(), model.transform(X).tolist())
    assert_equal(model.transform_table(table).values().tolist(), projection.values().tolist())
    assert_equal(model.transform_table(X).schema().names, components)
    with assert_raises(contains="feature names should match"):
        _ = model.transform_table(reversed_table)
    assert_equal(model.feature_names_in_.value(), names)
    assert_false(Bool(clone(model).feature_names_in_))
    assert_equal(model.get_feature_names_out(names.copy()), model.get_feature_names_out())
    with assert_raises(contains="input_features is not equal"):
        _ = model.get_feature_names_out(reordered.copy())
    assert_equal(model.predict(table), model.predict(X))
    assert_equal(model.decision_function(table).tolist(), model.decision_function(X).tolist())
    assert_equal(model.predict_proba(table).tolist(), model.predict_proba(X).tolist())
    assert_equal(model.predict_log_proba(table).tolist(), model.predict_log_proba(X).tolist())
    assert_equal(model.score(table, y), model.score(X, y))

    var restored = named_archive_round_trip[Schema](model, Schema.Model(), 3)
    assert_equal(restored.feature_names_in_.value(), names)
    check_unset_request(restored)
    assert_equal(restored.transform(table).tolist(), projection.values().tolist())
    with assert_raises(contains="feature names should match"):
        _ = restored.predict(reversed_table)
    var duplicates: List[String] = ["mass", "mass"]
    restored.feature_names_in_ = duplicates^
    with assert_raises(contains="feature_names_in_ must be unique"):
        _ = Schema.encode(restored)

    _ = model.set_score_request(sample_weight=String("validation_weight"))
    restored = named_archive_round_trip[Schema](model, Schema.Model(), Schema.StateVersion)
    var score_labels = y.copy()
    score_labels[0] = 2
    var score_weights = matrix[dtype](1, 9, [4.0, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(Shape(9))
    var routed_score = restored.score(
        table, score_labels, sample_weight=routed_named_score(restored, score_weights),
    )
    assert_equal(routed_score, model.score(table, score_labels, sample_weight=score_weights^))
    assert_false(routed_score == model.score(table, score_labels))
    if emit_parity:
        print("lda|", String(dtype), "|", names, "|", model.predict(table),
              "|", model.predict_proba(table).tolist(), "|", routed_score, sep="")
        print("projection|", String(dtype), "|", projection.schema().names,
              "|", projection.values().tolist(), sep="")

    var unnormalized: List[Float64] = [1.0, 3.0]
    model.priors = unnormalized^
    _ = model.fit(table, y)
    var combined = Schema.decode(Schema.encode(model))
    assert_equal(len(combined.fit_warnings), 1)
    assert_equal(combined.feature_names_in_.value(), names)
    model.priors = None
    _ = model.fit(X, y)
    assert_false(Bool(model.feature_names_in_))
    assert_equal(model.predict(table), model.predict(X))
    _ = model.fit(table, y)
    with assert_raises(contains="inconsistent numbers of samples"):
        _ = model.fit(table, List[Int]())
    assert_false(Bool(model.feature_names_in_))
    assert_false(model.__sklearn_is_fitted__())


def rank_reduced_projection[dtype: DType](emit_parity: Bool) raises:
    comptime Model = LinearDiscriminantAnalysisSchema[dtype].Model
    var X = matrix[dtype](12, 2, [
        -4.0, 0, -2, 0, -3, -1, -3, 1,
        -1, 0, 1, 0, 0, -1, 0, 1,
        2, 0, 4, 0, 3, -1, 3, 1,
    ])
    var y: List[Int] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    var model = Model(n_components=2)
    var projected = model.fit_transform_table(X, y.copy())
    assert_equal(len(model.get_feature_names_out()), 2)
    assert_equal(projected.schema().names, [String("lineardiscriminantanalysis0")])
    assert_equal(projected.values().shape.tolist(), [12, 1])
    assert_equal(projected.values().tolist(), model.transform(X).tolist())
    assert_false(Bool(model.feature_names_in_))
    if emit_parity:
        print("rank_reduced_projection|", String(dtype), "|", projected.schema().names,
              "|", projected.values().tolist(), sep="")
    var lsqr = Model(solver="lsqr")
    with assert_raises(contains="NotImplementedError"):
        _ = lsqr.fit_transform_table(X, y.copy())
    assert_true(lsqr.__sklearn_is_fitted__())


def named_qda[dtype: DType](emit_parity: Bool) raises:
    comptime Schema = QuadraticDiscriminantAnalysisSchema[dtype]
    var X = training_data[dtype]()
    var y: List[Int] = [1, 1, 1, 1, 1, 2, 2, 2, 2]
    var names: List[String] = ["mass", "âge"]
    var reordered: List[String] = ["âge", "mass"]
    var renamed: List[String] = ["height", "âge"]
    var table = ColumnTable(X, names)
    var reversed_table = ColumnTable(X, reordered)
    var renamed_table = ColumnTable(X, renamed)
    var model = Schema.Model()
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(table)
    assert_equal(model.fit(table, y).get_n_features_in(), 2)
    assert_equal(model.feature_names_in_.value(), names)
    assert_false(Bool(clone(model).feature_names_in_))
    assert_equal(model.predict(table), model.predict(X))
    assert_equal(model.decision_function(table).tolist(), model.decision_function(X).tolist())
    assert_equal(model.predict_proba(table).tolist(), model.predict_proba(X).tolist())
    assert_equal(model.predict_log_proba(table).tolist(), model.predict_log_proba(X).tolist())
    var weights = matrix[dtype](1, 9, [1.0, 1, 1, 2, 1, 2, 1, 2, 1]).reshape(Shape(9))
    assert_equal(
        model.score(table, y, sample_weight=weights.copy()),
        model.score(X, y, sample_weight=weights.copy()),
    )
    with assert_raises(contains="feature names should match"):
        _ = model.predict(reversed_table)
    with assert_raises(contains="feature names should match"):
        _ = model.predict_log_proba(renamed_table)

    var restored = named_archive_round_trip[Schema](model, Schema.Model(), 2)
    assert_equal(restored.feature_names_in_.value(), names)
    check_unset_request(restored)
    assert_equal(restored.predict_proba(table).tolist(), model.predict_proba(table).tolist())
    with assert_raises(contains="feature names should match"):
        _ = restored.predict(reversed_table)
    restored.feature_names_in_ = List[String]()
    with assert_raises(contains="invalid feature_names_in_ count"):
        _ = Schema.encode(restored)

    _ = model.set_score_request(sample_weight=String("validation_weight"))
    restored = named_archive_round_trip[Schema](model, Schema.Model(), Schema.StateVersion)
    var score_labels = y.copy()
    score_labels[0] = 2
    var score_weights = matrix[dtype](1, 9, [4.0, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(Shape(9))
    var routed_score = restored.score(
        table, score_labels, sample_weight=routed_named_score(restored, score_weights),
    )
    assert_equal(routed_score, model.score(table, score_labels, sample_weight=score_weights^))
    assert_false(routed_score == model.score(table, score_labels))
    if emit_parity:
        print("qda|", String(dtype), "|", names, "|", model.predict(table),
              "|", model.predict_proba(table).tolist(), "|", routed_score, sep="")

    _ = model.fit(reversed_table, y)
    assert_equal(model.feature_names_in_.value(), reordered)
    _ = model.fit(X, y)
    assert_false(Bool(model.feature_names_in_))
    assert_equal(model.predict(table), model.predict(X))
    _ = model.fit(table, y)
    var mixed = table.copy()
    mixed.add_strings("comment", List[String](length=9, fill="text"))
    with assert_raises(contains="mixed ColumnTable"):
        _ = model.fit(mixed, y)
    assert_false(Bool(model.feature_names_in_))
    assert_false(model.__sklearn_is_fitted__())


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
    test_score_request_lifecycle_and_archives()
    print("PASS: LDA and QDA score request lifecycle and archives")
    var emit_parity = len(argv()) > 1 and argv()[1] == "--named-parity"
    named_lda[DType.float32](emit_parity)
    named_lda[DType.float64](emit_parity)
    rank_reduced_projection[DType.float32](emit_parity)
    rank_reduced_projection[DType.float64](emit_parity)
    named_qda[DType.float32](emit_parity)
    named_qda[DType.float64](emit_parity)
    print("PASS: discriminant named input and archive contracts")
