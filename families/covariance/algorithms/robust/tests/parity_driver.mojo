"""Native side of the robust parity harness (tests/parity.py writes the cases)."""
import numojo as nm
from numojo import NDArray, Shape
from std.sys import argv
from std.math import inf, nan
from parity_io import Tokens, emit, emit_scalar, emit_error
from moxie.covariance import MinCovDet, EllipticEnvelope, fast_mcd


def mask(values: NDArray[DType.bool]) raises -> NDArray[DType.float64]:
    var result = nm.zeros[DType.float64](Shape(values.size))
    for k in range(values.size):
        result.unsafe_set(k, 1.0 if values.unsafe_get(k) else 0.0)
    return result^


def labels(values: NDArray[DType.int64]) raises -> NDArray[DType.float64]:
    var result = nm.zeros[DType.float64](Shape(values.size))
    for k in range(values.size):
        result.unsafe_set(k, Float64(values.unsafe_get(k)))
    return result^


def run[dtype: DType](mut tokens: Tokens) raises:
    var X = tokens.matrix[dtype]()
    var X_test = tokens.matrix[dtype]()
    var y_values = tokens.matrix[DType.float64]()
    var fraction_value = tokens.real()
    var seed = tokens.integer()
    var assume = tokens.integer() == 1
    var contamination = tokens.real()
    var fraction: Optional[Float64] = None
    if fraction_value > 0:
        fraction = fraction_value
    var y = nm.zeros[DType.int64](Shape(y_values.size))
    for k in range(y_values.size):
        y.unsafe_set(k, Int64(Int(y_values.unsafe_get(k))))
    try:
        var result = fast_mcd(X, fraction, seed)
        emit("fast_mcd_location", result.location)
        emit("fast_mcd_covariance", result.covariance)
        emit("fast_mcd_support", mask(result.support))
        emit("fast_mcd_dist", result.dist)
    except e:
        emit_error("fast_mcd", e)
    for store in range(2):
        var key = String("mcd_store", store)
        try:
            var model = MinCovDet[dtype](store_precision=store == 1, assume_centered=assume, support_fraction=fraction, random_state=seed)
            model.fit(X)
            emit(key + "_raw_location", model.raw_location_())
            emit(key + "_raw_covariance", model.raw_covariance_())
            emit(key + "_raw_support", mask(model.raw_support_()))
            emit(key + "_location", model.location_())
            emit(key + "_covariance", model.covariance_())
            emit(key + "_precision", model.get_precision())
            emit(key + "_support", mask(model.support_()))
            emit(key + "_dist", model.dist_())
            emit_scalar(key + "_score", model.score(X_test))
            emit(key + "_mahalanobis", model.mahalanobis(X_test))
            for step in range(3):
                var post = key + String("_post", step)
                if step != 1:
                    emit(post + "_corrected", model.correct_covariance(X))
                    emit(post + "_corrected_dist", model.dist_())
                var result = model.reweight_covariance(X)
                emit(post + "_returned_location", result[0])
                emit(post + "_returned_covariance", result[1])
                emit(post + "_returned_support", mask(result[2]))
                emit(post + "_location", model.location_())
                emit(post + "_covariance", model.covariance_())
                emit(post + "_dist", model.dist_())
                emit(post + "_raw_covariance", model.raw_covariance_())
                emit(post + "_raw_location", model.raw_location_())
                emit(post + "_support", mask(model.support_()))
                emit(post + "_precision", model.get_precision())
        except e:
            emit_error(key, e)
    try:
        var detector = EllipticEnvelope[dtype](assume_centered=assume, support_fraction=fraction, contamination=contamination, random_state=seed)
        emit("ee_fit_predict", labels(detector.fit_predict(X, y.copy())))
        emit_scalar("ee_offset", detector.offset_())
        emit("ee_predict", labels(detector.predict(X_test)))
        emit("ee_decision", detector.decision_function(X_test))
        emit("ee_score_samples", detector.score_samples(X_test))
        emit_scalar("ee_score", detector.score(X_test, y))
        emit("ee_covariance", detector.covariance_())
        var p = X.shape[1]
        var comparison = nm.zeros[dtype](Shape(p, p))
        var backing = nm.zeros[dtype](Shape(2 * p * p + 2))
        for i in range(p):
            for j in range(p):
                var value = Float64(2 * i - j + 1).cast[dtype]()
                comparison.unsafe_set(i * p + j, value)
                backing.unsafe_set(2 * p * p - 1 - 2 * (i * p + j), value)
        var strided = backing.view_with_layout(Shape(p, p), nm.NDArrayStrides(-2 * p, -2), 2 * p * p - 1)
        emit_scalar("ee_error_default", detector.error_norm(comparison))
        for norm in [String("frobenius"), String("spectral")]:
            for scaling in range(2):
                for squared in range(2):
                    var key = String("ee_error_", norm, "_", scaling, "_", squared)
                    emit_scalar(key, detector.error_norm(comparison, norm=norm, scaling=scaling == 1, squared=squared == 1))
                    emit_scalar(key + "_strided", detector.error_norm(strided, norm, scaling == 1, squared == 1))
        try:
            emit_scalar("ee_error_invalid_norm", detector.error_norm(comparison, norm="invalid"))
        except e:
            emit_error("ee_error_invalid_norm", e)
        try:
            emit_scalar("ee_error_invalid_shape", detector.error_norm(nm.zeros[dtype](Shape(0, 0)), norm="spectral"))
        except e:
            emit_error("ee_error_invalid_shape", e)
        for kind in range(3):
            var nonfinite = comparison.copy()
            nonfinite.unsafe_set(0, nan[dtype]() if kind == 0 else (inf[dtype]() if kind == 1 else -inf[dtype]()))
            emit_scalar(String("ee_error_nonfinite_", kind), detector.error_norm(nonfinite))
        for kind in range(11):
            var key = String("ee_weighted_", kind)
            var weights = nm.ones[DType.float64](Shape(y.size))
            if kind == 0 or kind == 1:
                for k in range(y.size):
                    weights.unsafe_set(k, Float64(k + 1))
                if kind == 1:
                    weights.unsafe_set(0, -2.0)
            elif kind == 2:
                weights = nm.ones[DType.float64](Shape(y.size - 1))
            elif kind == 3:
                weights = nm.ones[DType.float64](Shape(y.size + 1))
            elif kind == 4:
                weights = nm.ones[DType.float64](Shape(y.size, 1))
            elif kind == 5:
                weights = nm.ones[DType.float64](Shape(0))
            elif kind == 6:
                weights.unsafe_set(0, nan[DType.float64]())
            elif kind == 7:
                weights.unsafe_set(0, inf[DType.float64]())
            elif kind == 8:
                weights.unsafe_set(0, -inf[DType.float64]())
            else:
                weights = nm.zeros[DType.float64](Shape(y.size))
                if kind == 10:
                    weights.unsafe_set(0, 1.0)
                    weights.unsafe_set(1, -1.0)
            try:
                emit_scalar(key, detector.score(X_test, y, weights^))
            except e:
                emit_error(key, e)

        for step in range(3):
            var post = String("ee") + String("_post", step)
            if step != 1:
                emit(post + "_corrected", detector.correct_covariance(X))
                emit(post + "_corrected_dist", detector.dist_())
            var result = detector.reweight_covariance(X)
            emit(post + "_returned_location", result[0])
            emit(post + "_returned_covariance", result[1])
            emit(post + "_returned_support", mask(result[2]))
            emit(post + "_location", detector.location_())
            emit(post + "_covariance", detector.covariance_())
            emit(post + "_dist", detector.dist_())
            emit(post + "_raw_covariance", detector.raw_covariance_())
            emit(post + "_raw_location", detector.raw_location_())
            emit(post + "_support", mask(detector.support_()))
            emit(post + "_precision", detector.get_precision())
            emit_scalar(post + "_offset", detector.offset_())
            emit(post + "_predict", labels(detector.predict(X_test)))

    except e:
        emit_error("ee", e)


def main() raises:
    var tokens = Tokens(String(argv()[1]))
    if tokens.integer() == 32:
        run[DType.float32](tokens)
    else:
        run[DType.float64](tokens)
