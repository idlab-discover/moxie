"""File round trips, malformed archives, and restored inference for pinned parity."""
from moxie.covariance import (
    EmpiricalCovariance, EmpiricalCovarianceSchema,
    ShrunkCovariance, ShrunkCovarianceSchema,
    LedoitWolf, LedoitWolfSchema, OAS, OASSchema,
    MinCovDet, MinCovDetSchema, EllipticEnvelope, EllipticEnvelopeSchema,
    GraphicalLasso, GraphicalLassoSchema, GraphicalLassoCV, GraphicalLassoCVSchema,
    empirical_covariance,
)
from moxie.serialization import ModelSchema, dumps, load_model, save_model
from numojo import NDArray, NDArrayStrides, Shape, zeros
from parity_io import Tokens, emit, emit_scalar
from std.math import inf, nan
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


def check_robust[dtype: DType](X: NDArray[dtype], probe: NDArray[dtype], path: String) raises:
    comptime MCDSchema = MinCovDetSchema[dtype]
    comptime EnvelopeSchema = EllipticEnvelopeSchema[dtype]
    var mcd = MinCovDet[dtype](store_precision=False, support_fraction=0.75, random_state=0)
    var empty_mcd = roundtrip[MCDSchema](mcd, path)
    assert_false(empty_mcd.__sklearn_is_fitted__())
    assert_equal(empty_mcd.support_fraction.value(), 0.75)
    assert_equal(empty_mcd.random_state.value(), 0)
    with assert_raises():
        _ = empty_mcd.mahalanobis(probe)

    _ = mcd.fit(X)
    _ = mcd.set_params(store_precision=True, assume_centered=True, support_fraction=0.9)
    var restored_mcd = roundtrip[MCDSchema](mcd, path)
    assert_true(restored_mcd.store_precision)
    assert_true(restored_mcd.assume_centered)
    assert_equal(restored_mcd.support_fraction.value(), 0.9)
    assert_false(Bool(restored_mcd.precision_()))
    assert_equal(restored_mcd.n_features_in_(), X.shape[1])
    inference(
        "mcd", restored_mcd.score(probe), mcd.score(probe),
        restored_mcd.mahalanobis(probe), mcd.mahalanobis(probe), probe.shape[0],
    )
    emit("mcd_raw_location", restored_mcd.raw_location_())
    emit("mcd_raw_covariance", restored_mcd.raw_covariance_())
    emit("mcd_raw_support", restored_mcd.raw_support_())
    emit("mcd_support", restored_mcd.support_())
    emit("mcd_dist", restored_mcd.dist_())
    _ = restored_mcd.correct_covariance(X)
    _ = mcd.correct_covariance(X)
    _ = restored_mcd.reweight_covariance(X)
    _ = mcd.reweight_covariance(X)
    assert_true(dumps(MCDSchema.encode(restored_mcd)) == dumps(MCDSchema.encode(mcd)))
    assert_true(Bool(restored_mcd.precision_()))

    var envelope = EllipticEnvelope[dtype](contamination=0.2)
    var empty_envelope = roundtrip[EnvelopeSchema](envelope, path)
    assert_false(empty_envelope.__sklearn_is_fitted__())
    assert_false(Bool(empty_envelope.random_state))
    assert_false(Bool(empty_envelope.support_fraction))
    with assert_raises():
        _ = empty_envelope.predict(probe)
    with assert_raises():
        _ = empty_envelope.offset_()

    _ = envelope.set_params(random_state=0)
    _ = envelope.fit(X)
    _ = envelope.set_params(contamination=0.4, store_precision=False)
    var restored_envelope = roundtrip[EnvelopeSchema](envelope, path)
    assert_equal(restored_envelope.contamination, 0.4)
    assert_false(restored_envelope.store_precision)
    assert_true(Bool(restored_envelope.precision_()))
    assert_equal(restored_envelope.offset_(), envelope.offset_())
    var labels = restored_envelope.predict(probe)
    var decisions = restored_envelope.decision_function(probe)
    var expected_labels = envelope.predict(probe)
    var expected_decisions = envelope.decision_function(probe)
    for i in range(probe.shape[0]):
        assert_equal(labels.item(i), expected_labels.item(i))
        assert_equal(decisions.item(i), expected_decisions.item(i))
    emit_scalar("envelope_offset", restored_envelope.offset_())
    emit("envelope_predict", labels)
    emit("envelope_decision", decisions)
    emit("envelope_score_samples", restored_envelope.score_samples(probe))
    emit("envelope_mahalanobis", restored_envelope.mahalanobis(probe))


def invalid_robust_archive(X: NDArray[DType.float32]) raises:
    comptime Schema = MinCovDetSchema[DType.float32]
    var model = MinCovDet[DType.float32](random_state=0)
    _ = model.fit(X)
    var archive = Schema.encode(model)
    with assert_raises():
        _ = EllipticEnvelopeSchema[DType.float32].decode(archive)
    for ref payload in archive.payloads:
        if payload.name == "raw_support_":
            payload.data[0] = 2
    with assert_raises():
        _ = Schema.decode(archive)

    archive = Schema.encode(model)
    for ref payload in archive.payloads:
        if payload.name == "dist_":
            payload.shape = [1, X.shape[0]]
    with assert_raises():
        _ = Schema.decode(archive)

    model._fit.robust = None
    with assert_raises():
        _ = Schema.encode(model)

    var envelope = EllipticEnvelope[DType.float32](random_state=0)
    _ = envelope.fit(X)
    envelope._offset = nan[DType.float64]()
    with assert_raises():
        _ = EllipticEnvelopeSchema[DType.float32].encode(envelope)


def emit_graphical_diagnostics(
    name: String, iterations: Int, costs: List[Tuple[Float64, Float64]],
) raises:
    var values = zeros[DType.float64](Shape(len(costs), 2))
    for i in range(len(costs)):
        values.unsafe_set(2 * i, costs[i][0])
        values.unsafe_set(2 * i + 1, costs[i][1])
    emit(name + "_costs", values)
    emit_scalar(name + "_n_iter", Float64(iterations))


def check_graphical[dtype: DType](X: NDArray[dtype], probe: NDArray[dtype], path: String) raises:
    comptime FixedSchema = GraphicalLassoSchema[dtype]
    comptime CVSchema = GraphicalLassoCVSchema[dtype]
    var fixed = GraphicalLasso[dtype](0.1, mode="lars", max_iter=70)
    var empty = roundtrip[FixedSchema](fixed, path)
    assert_false(empty.__sklearn_is_fitted__())
    with assert_raises():
        _ = empty.score(probe)
    _ = fixed.fit(X)
    _ = fixed.set_params(alpha=0.8, max_iter=0, assume_centered=True, covariance=String("precomputed"))
    var restored = roundtrip[FixedSchema](fixed, path)
    assert_equal(restored.alpha, 0.8)
    assert_equal(restored.max_iter, 0)
    assert_true(restored.assume_centered)
    assert_equal(restored.covariance.value(), "precomputed")
    assert_equal(restored.n_features_in_(), X.shape[1])
    assert_false(restored.__sklearn_clone__().__sklearn_is_fitted__())
    inference("graphical", restored.score(probe), fixed.score(probe),
              restored.mahalanobis(probe), fixed.mahalanobis(probe), probe.shape[0])
    emit("graphical_covariance", restored.covariance_())
    emit("graphical_precision", restored.get_precision())
    emit("graphical_location", restored.location_())
    emit_graphical_diagnostics("graphical", restored.n_iter_(), restored.costs_())

    var edge = GraphicalLasso[dtype](0, covariance=String("precomputed"), max_iter=0)
    _ = edge.fit(empirical_covariance(X))
    var zero_alpha = roundtrip[FixedSchema](edge, path)
    assert_equal(zero_alpha.n_iter_(), 0)
    assert_equal(len(zero_alpha.costs_()), 1)
    _ = edge.set_params(alpha=0.1)
    _ = edge.fit(empirical_covariance(X))
    var zero_budget = roundtrip[FixedSchema](edge, path)
    assert_equal(zero_budget.n_iter_(), 1)
    assert_equal(len(zero_budget.costs_()), 0)

    var alphas: List[Float64] = [0.2, 0.4, 0.1]
    var cv = GraphicalLassoCV[dtype](alphas=alphas, cv=3, n_jobs=1, mode="lars")
    var empty_cv = roundtrip[CVSchema](cv, path)
    assert_false(empty_cv.__sklearn_is_fitted__())
    assert_true(Bool(empty_cv.alphas.values))
    assert_equal(empty_cv.alphas.values.value()[0], 0.2)
    assert_equal(empty_cv.cv.value(), 3)
    assert_equal(empty_cv.n_jobs.value(), 1)
    with assert_raises():
        _ = empty_cv.cv_results_()
    _ = cv.fit(X)
    _ = cv.set_params(alphas=5, cv=2, n_refinements=2, mode="cd", n_jobs=3, max_iter=0)
    var restored_cv = roundtrip[CVSchema](cv, path)
    assert_equal(restored_cv.alphas.count, 5)
    assert_false(Bool(restored_cv.alphas.values))
    assert_equal(restored_cv.cv.value(), 2)
    assert_equal(restored_cv.n_jobs.value(), 3)
    assert_equal(restored_cv.n_refinements, 2)
    assert_equal(restored_cv.mode, "cd")
    assert_equal(restored_cv.max_iter, 0)
    assert_false(restored_cv.__sklearn_clone__().__sklearn_is_fitted__())
    inference("graphical_cv", restored_cv.score(probe), cv.score(probe),
              restored_cv.mahalanobis(probe), cv.mahalanobis(probe), probe.shape[0])
    emit("graphical_cv_covariance", restored_cv.covariance_())
    emit("graphical_cv_precision", restored_cv.get_precision())
    emit("graphical_cv_location", restored_cv.location_())
    emit_scalar("graphical_cv_alpha", restored_cv.alpha_())
    emit_graphical_diagnostics("graphical_cv", restored_cv.n_iter_(), restored_cv.costs_())
    var results = restored_cv.cv_results_()
    assert_equal(results.split_test_score.shape[0], 4)
    assert_equal(results.split_test_score.shape[1], 3)
    emit("graphical_cv_alphas", results.alphas)
    emit("graphical_cv_splits", results.split_test_score)
    emit("graphical_cv_mean", results.mean_test_score)
    emit("graphical_cv_std", results.std_test_score)

    # The solver and path use IEEE exceptional values as diagnostics.
    # Serialization also accepts the same logical alpha order in a strided view.
    var count = results.alphas.size
    var reversed_alphas = zeros[DType.float64](Shape(count))
    for i in range(count):
        reversed_alphas.unsafe_set(count - 1 - i, results.alphas.unsafe_get(i))
    results.alphas = reversed_alphas.view_with_layout(Shape(count), NDArrayStrides(-1), count - 1)
    results.split_test_score.unsafe_set(0, -inf[DType.float64]())
    results.mean_test_score.unsafe_set(0, -inf[DType.float64]())
    results.std_test_score.unsafe_set(0, nan[DType.float64]())
    cv._results = results^
    _ = roundtrip[CVSchema](cv, path)

    var archive = CVSchema.encode(cv)
    with assert_raises():
        _ = FixedSchema.decode(archive)
    for i in range(len(archive.payloads)):
        if archive.payloads[i].name == "split_test_score":
            archive.payloads[i].shape = [3, 4]
    with assert_raises():
        _ = CVSchema.decode(archive)
    archive = CVSchema.encode(cv)
    for i in range(len(archive.manifest) - 8, len(archive.manifest)):
        archive.manifest[i] = 0
    with assert_raises():
        _ = CVSchema.decode(archive)
    var fixed_archive = FixedSchema.encode(fixed)
    for i in range(len(fixed_archive.payloads)):
        if fixed_archive.payloads[i].name == "costs_":
            fixed_archive.payloads[i].shape = [2 * len(fixed.costs_())]
    with assert_raises():
        _ = FixedSchema.decode(fixed_archive)
    fixed._n_iter = -1
    with assert_raises():
        _ = FixedSchema.encode(fixed)
    fixed._n_iter = restored.n_iter_()
    fixed._model._precision64 = None
    with assert_raises():
        _ = FixedSchema.encode(fixed)


def run_case[dtype: DType](mut tokens: Tokens, path: String) raises:
    var X = tokens.matrix[dtype]()
    var probe = tokens.matrix[dtype]()
    var robust_X = tokens.matrix[dtype]()
    check_models(X, probe, path)
    check_robust(robust_X, probe, path)
    check_graphical(robust_X, probe, path)
    comptime if dtype == DType.float32:
        strided_state(rebind[NDArray[DType.float32]](X))
        invalid_archives(rebind[NDArray[DType.float32]](X^))
        invalid_robust_archive(rebind[NDArray[DType.float32]](robust_X^))


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
