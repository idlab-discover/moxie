"""Native contracts for MinCovDet, EllipticEnvelope and fast_mcd."""
import numojo as nm
from numojo import NDArray, Shape
from std.math import abs, inf, nan
from std.testing import assert_equal, assert_true, assert_false
from moxie.base import clone
from moxie._random import RandomState
from moxie.covariance import MinCovDet, EllipticEnvelope, fast_mcd
from moxie.covariance._robust_covariance import consistency_factor, chi2_isf


def close(a: Float64, b: Float64, tol: Float64 = 1e-6) -> Bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def has(message: String, error: Error) -> Bool:
    return String(error).find(message) >= 0


def data[dtype: DType](n: Int, p: Int, seed: Int) raises -> NDArray[dtype]:
    var rng = RandomState(seed)
    var X = nm.zeros[dtype](Shape(n, p))
    for k in range(n * p):
        X.unsafe_set(k, rng.standard_normal().cast[dtype]())
    for j in range(p):
        X.unsafe_set(j, (20.0 + Float64(j)).cast[dtype]())
    return X^


def count(mask: NDArray[DType.bool]) -> Int:
    var total = 0
    for k in range(mask.size):
        if mask.unsafe_get(k):
            total += 1
    return total


def same[dtype: DType](a: NDArray[dtype], b: NDArray[dtype]) -> Bool:
    if a.size != b.size:
        return False
    for k in range(a.size):
        if a.unsafe_get(k) != b.unsafe_get(k):
            return False
    return True


def check[dtype: DType]() raises:
    assert_true(close(consistency_factor(3, 0.975), 1.078478718355326, 1e-12))
    assert_true(close(consistency_factor(1, 0.6), 4.659969541210519, 1e-12))
    assert_true(close(chi2_isf(0.025, 4), 11.143286781877796, 1e-12))
    var X = data[dtype](60, 3, 3)
    var model = MinCovDet[dtype](random_state=7)
    assert_false(model.__sklearn_is_fitted__())
    try:
        _ = model.raw_location_()
        assert_true(False)
    except e:
        assert_true(has("NotFittedError: This MinCovDet instance", e))
    model.fit(X)
    # Shapes, support sizes and the outlier row excluded.
    assert_equal(model.raw_location_().shape[0], 3)
    assert_equal(model.raw_covariance_().shape[0], 3)
    assert_equal(model.dist_().shape[0], 60)
    assert_equal(count(model.raw_support_()), 32)  # ceil((60 + 3 + 1) / 2)
    assert_false(model.support_().unsafe_get(0))
    # Integer seeds are reproducible; clone is unfitted with equal parameters.
    var again = clone(model)
    assert_false(again.__sklearn_is_fitted__())
    again.fit(X)
    assert_true(same(model.raw_support_(), again.raw_support_()))
    assert_true(same(model.covariance_(), again.covariance_()))
    # A shared mutable stream advances between fits (sklearn RandomState instance).
    var stream = RandomState(7)
    var first = MinCovDet[dtype]()
    first.fit(X, stream)
    assert_true(same(first.raw_support_(), model.raw_support_()))
    var second = MinCovDet[dtype]()
    second.fit(X, stream)
    assert_true(count(second.raw_support_()) == 32)
    # fast_mcd agrees with the estimator's raw fit for the same seed.
    var raw = fast_mcd(X, None, 7)
    assert_true(same(raw.support, model.raw_support_()))
    # Invalid inputs and parameters.
    var bad = MinCovDet[dtype](support_fraction=1.5)
    try:
        bad.fit(X)
        assert_true(False)
    except e:
        assert_true(has("support_fraction", e))
    try:
        var tiny = MinCovDet[dtype](random_state=0)
        tiny.fit(nm.ones[dtype](Shape(1, 2)))
        assert_true(False)
    except e:
        assert_true(has("minimum of 2 is required", e))
    var constant = nm.zeros[dtype](Shape(40, 2))
    var spikes: List[Float64] = [5.0, 1.0, -4.0, 2.0, 3.0, -3.0]
    for k in range(6):
        constant.unsafe_set(k, spikes[k].cast[dtype]())
    try:
        var degenerate = MinCovDet[dtype](random_state=0)
        degenerate.fit(constant)
        assert_true(False)
    except e:
        assert_true(has("covariance matrix of the support data is equal to 0", e), String(e))
    # EllipticEnvelope: labels, decision offset and accuracy.
    var detector = EllipticEnvelope[dtype](contamination=0.1, random_state=7)
    detector.fit(X)
    var labels = detector.predict(X)
    var decision = detector.decision_function(X)
    var samples = detector.score_samples(X)
    var outliers = 0
    for k in range(60):
        assert_true(labels.unsafe_get(k) == 1 or labels.unsafe_get(k) == -1)
        assert_equal(labels.unsafe_get(k) == 1, decision.unsafe_get(k) >= 0)
        assert_true(close(Float64(samples.unsafe_get(k)) - detector.offset_(), Float64(decision.unsafe_get(k)), 1e-4))
        if labels.unsafe_get(k) == -1:
            outliers += 1
    assert_equal(labels.unsafe_get(0), -1)
    assert_true(outliers >= 6 and outliers <= 7)
    assert_true(close(detector.score(X, labels), 1.0))
    # Weighted accuracy, signed weights, and noncontiguous views.
    var truth = labels.copy()
    truth.unsafe_set(0, -truth.unsafe_get(0))
    var weights = nm.ones[DType.float64](Shape(60))
    weights.unsafe_set(0, 2.0)
    assert_true(close(detector.score(X, truth, weights.copy()), 59.0 / 61.0, 1e-12))
    weights.unsafe_set(0, -1.0)
    assert_true(close(detector.score(X, truth, weights.copy()), 59.0 / 58.0, 1e-12))
    var backing = nm.ones[DType.float64](Shape(121))
    backing.unsafe_set(119, 2.0)
    var reversed_weights = backing.view_with_layout(Shape(60), nm.NDArrayStrides(-2), 119)
    assert_true(close(detector.score(X, truth, reversed_weights^), 59.0 / 61.0, 1e-12))
    var forward_weights = backing.view_with_layout(Shape(60), nm.NDArrayStrides(2), 1)
    assert_true(close(detector.score(X, truth, forward_weights^), 60.0 / 61.0, 1e-12))
    for kind in range(9):
        var invalid_weights = nm.zeros[DType.float64](Shape(60))
        var message = String("sample_weight")
        if kind == 0:
            invalid_weights = nm.ones[DType.float64](Shape(59))
        elif kind == 1:
            invalid_weights = nm.ones[DType.float64](Shape(61))
        elif kind == 2:
            invalid_weights = nm.ones[DType.float64](Shape(60, 1))
        elif kind == 3:
            invalid_weights = nm.ones[DType.float64](Shape(0))
        elif kind == 4:
            invalid_weights.unsafe_set(0, nan[DType.float64]())
        elif kind == 5:
            invalid_weights.unsafe_set(0, inf[DType.float64]())
        elif kind == 6:
            invalid_weights.unsafe_set(0, -inf[DType.float64]())
        elif kind == 7:
            message = "ValueError: Sample weights must contain at least one non-zero"
        else:
            invalid_weights.unsafe_set(0, 1.0)
            invalid_weights.unsafe_set(1, -1.0)
            message = "ZeroDivisionError: Weights sum to zero"
        try:
            _ = detector.score(X, truth, invalid_weights^)
            assert_true(False)
        except e:
            assert_true(has(message, e), String(e))
    assert_true(same(detector.predict(X), labels))
    # fit_predict fits an unfitted clone, ignores y and replaces prior state.
    var combined = clone(detector)
    var ignored_y = nm.zeros[DType.int64](Shape(1))
    assert_true(same(combined.fit_predict(X, ignored_y^), labels))
    assert_true(same(combined.covariance_(), detector.covariance_()))
    var next_X = data[dtype](31, 2, 11)
    var expected = clone(detector)
    expected.fit(next_X)
    var next_labels = combined.fit_predict(next_X)
    assert_equal(next_labels.ndim, 1)
    assert_equal(next_labels.size, 31)
    assert_equal(combined.n_features_in_(), 2)
    assert_true(same(next_labels, expected.predict(next_X)))
    # The explicit stream overload consumes exactly the same draws as fit.
    var combined_stream = RandomState(13)
    var expected_stream = RandomState(13)
    expected.fit(X, expected_stream)
    assert_true(same(combined.fit_predict(X, combined_stream), expected.predict(X)))
    assert_equal(combined_stream.standard_normal(), expected_stream.standard_normal())
    try:
        _ = combined.fit_predict(nm.ones[dtype](Shape(1, 2)))
        assert_true(False)
    except e:
        assert_true(has("minimum of 2 is required", e))
    var invalid = EllipticEnvelope[dtype](contamination=0.7)
    try:
        _ = invalid.fit_predict(X)
        assert_true(False)
    except e:
        assert_true(has("contamination", e))
    try:
        _ = invalid.predict(X)
        assert_true(False)
    except e:
        assert_true(has("NotFittedError", e))
    # Univariate branch and full support.
    var univariate = MinCovDet[dtype](random_state=0)
    univariate.fit(data[dtype](25, 1, 1))
    assert_equal(univariate.covariance_().shape[0], 1)
    var full = MinCovDet[dtype](support_fraction=1.0, random_state=0)
    full.fit(X)
    assert_equal(count(full.raw_support_()), 60)
    print("PASS robust contracts", dtype)


def check_postprocessing[dtype: DType]() raises:
    var X = data[dtype](60, 3, 3)
    var model = MinCovDet[dtype](random_state=7)
    var detector = EllipticEnvelope[dtype](random_state=7)
    for method in range(4):
        try:
            if method == 0:
                _ = model.correct_covariance(X)
            elif method == 1:
                _ = model.reweight_covariance(X)
            elif method == 2:
                _ = detector.correct_covariance(X)
            else:
                _ = detector.reweight_covariance(X)
            assert_true(False)
        except e:
            assert_true(has("NotFittedError", e))
    model.fit(X)
    detector.fit(X)
    var before = model.covariance_()
    var raw = model.raw_covariance_()
    var original_dist = model.dist_()
    var factor = consistency_factor(3, Float64(count(model.support_())) / 60.0)
    # The reference ignores correction's data argument, including its shape.
    var corrected = model.correct_covariance(nm.zeros[dtype](Shape(0)))
    assert_equal(corrected.ndim, 2)
    assert_equal(corrected.shape[0], 3)
    assert_true(same(before, model.covariance_()))
    for k in range(9):
        assert_true(close(Float64(corrected.unsafe_get(k)), Float64(raw.unsafe_get(k)) * factor))
    for k in range(60):
        assert_true(close(Float64(model.dist_().unsafe_get(k)), Float64(original_dist.unsafe_get(k)) / factor))
    # Represent identical data with negative row/column strides and an offset.
    var backing = nm.zeros[dtype](Shape(362))
    for i in range(60):
        for j in range(3):
            backing.unsafe_set(359 - 6 * i - 2 * j, X.unsafe_get(3 * i + j))
    var strided = backing.view_with_layout(Shape(60, 3), nm.NDArrayStrides(-6, -2), 359)
    var contiguous = model.copy()
    var result = model.reweight_covariance(strided)
    var expected = contiguous.reweight_covariance(X)
    assert_true(same(result[0], expected[0]))
    assert_true(same(result[1], expected[1]))
    assert_true(same(result[2], expected[2]))
    assert_equal(result[0].ndim, 1)
    assert_equal(result[1].ndim, 2)
    assert_equal(result[2].size, 60)
    assert_true(same(raw, model.raw_covariance_()))
    factor = consistency_factor(3, 0.975)
    for k in range(9):
        assert_true(close(Float64(model.covariance_().unsafe_get(k)), Float64(result[1].unsafe_get(k)) * factor))
    var installed_location = model.location_()
    var installed_covariance = model.covariance_()
    var installed_support = model.support_()
    result[0].unsafe_set(0, 999)
    result[1].unsafe_set(0, 999)
    result[2].unsafe_set(0, not result[2].unsafe_get(0))
    assert_true(same(installed_location, model.location_()))
    assert_true(same(installed_covariance, model.covariance_()))
    assert_true(same(installed_support, model.support_()))
    # Invalid input rejects before changing learned state.
    var saved_dist = model.dist_()
    for kind in range(5):
        var bad = nm.ones[dtype](Shape(59, 3))
        if kind == 1:
            bad = nm.ones[dtype](Shape(60, 2))
        elif kind == 2:
            bad = nm.ones[dtype](Shape(180))
        elif kind == 3:
            bad = X.copy()
            bad.unsafe_set(0, nan[dtype]())
        elif kind == 4:
            bad = X.copy()
            bad.unsafe_set(0, inf[dtype]())
        try:
            _ = model.reweight_covariance(bad)
            assert_true(False)
        except e:
            assert_true(has("ValueError", e))
        assert_true(same(installed_covariance, model.covariance_()))
        assert_true(same(saved_dist, model.dist_()))
    # Current constructor settings govern reweighting; offset stays fitted.
    var offset = detector.offset_()
    _ = detector.correct_covariance(X)
    detector.assume_centered = True
    detector.store_precision = False
    _ = detector.reweight_covariance(strided)
    assert_equal(detector.offset_(), offset)
    assert_false(Bool(detector.precision_()))
    for k in range(3):
        assert_equal(detector.location_().unsafe_get(k), 0)
    model.store_precision = False
    _ = model.reweight_covariance(X)
    assert_false(Bool(model.precision_()))
    model.store_precision = True
    _ = model.reweight_covariance(X)
    assert_true(Bool(model.precision_()))
    # Empty support and zero raw covariance use explicit, transactional errors.
    var snapshot = model._fit.state("MinCovDet")
    for k in range(60):
        snapshot.dist_.unsafe_set(k, 1e20)
    model._fit.robust = snapshot^
    try:
        _ = model.reweight_covariance(X)
        assert_true(False)
    except e:
        assert_true(has("no sample passes", e))
    snapshot = model._fit.state("MinCovDet")
    snapshot.raw_covariance_ = nm.zeros[dtype](Shape(3, 3))
    snapshot.support_.unsafe_set(0, False)
    model._fit.robust = snapshot^
    try:
        _ = model.correct_covariance(X)
        assert_true(False)
    except e:
        assert_true(has("covariance matrix of the support data is equal to 0", e))
    print("PASS robust postprocessing contracts", dtype)


def check_error_norm[dtype: DType]() raises:
    var detector = EllipticEnvelope[dtype](random_state=7)
    var comparison = nm.ones[dtype](Shape(3, 3))
    try:
        _ = detector.error_norm(comparison)
        assert_true(False)
    except e:
        assert_true(has("NotFittedError: This EllipticEnvelope instance", e), String(e))
    detector.fit(data[dtype](60, 3, 3))
    var covariance = detector.covariance_()
    var offset = detector.offset_()
    var backing = nm.zeros[dtype](Shape(20))
    for k in range(9):
        backing.unsafe_set(1 + 2 * k, 1)
    var strided = backing.view_with_layout(Shape(3, 3), nm.NDArrayStrides(6, 2), 1)
    for norm in [String("frobenius"), String("spectral")]:
        assert_equal(detector.error_norm(covariance, norm), 0)
        for scaling in range(2):
            for squared in range(2):
                assert_equal(detector.error_norm(comparison, norm, scaling == 1, squared == 1),
                             detector.error_norm(strided, norm, scaling == 1, squared == 1))
    for kind in range(5):
        var bad = nm.zeros[dtype](Shape(3))
        if kind == 1:
            bad = nm.zeros[dtype](Shape(2, 3))
        elif kind == 2:
            bad = nm.zeros[dtype](Shape(0, 0))
        elif kind == 3:
            bad = nm.zeros[dtype](Shape(3, 3, 1))
        elif kind == 4:
            bad = nm.zeros[dtype](Shape(1, 3))
        try:
            _ = detector.error_norm(bad)
            assert_true(False)
        except e:
            assert_true(has("ValueError: error_norm: comp_cov shape", e), String(e))
    assert_true(same(covariance, detector.covariance_()))
    assert_equal(offset, detector.offset_())
    print("PASS EllipticEnvelope error_norm contracts", dtype)


def main() raises:
    check[DType.float64]()
    check[DType.float32]()
    check_postprocessing[DType.float64]()
    check_postprocessing[DType.float32]()
    check_error_norm[DType.float64]()
    check_error_norm[DType.float32]()
