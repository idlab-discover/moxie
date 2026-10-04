"""Native side of the empirical parity harness (tests/parity.py writes the cases)."""
import numojo as nm
from numojo import NDArray, Shape
from std.sys import argv
from parity_io import Tokens, emit, emit_scalar, emit_error
from moxie.covariance import EmpiricalCovariance, ShrunkCovariance, empirical_covariance, log_likelihood, shrunk_covariance
from moxie.covariance._empirical_covariance import fast_logdet


def estimator_outputs[dtype: DType](prefix: String, X: NDArray[dtype], X_test: NDArray[dtype], assume_centered: Bool,
                                    shrinkage: Optional[Float64]) raises:
    for store in range(2):
        var key = String(prefix, "_store", store)
        try:
            if shrinkage:
                var model = ShrunkCovariance[dtype](store_precision=store == 1, assume_centered=assume_centered, shrinkage=shrinkage.value())
                model.fit(X)
                emit(key + "_location", model.location_())
                emit(key + "_covariance", model.covariance_())
                emit(key + "_precision", model.get_precision())
                emit_scalar(key + "_score", model.score(X_test))
                emit(key + "_mahalanobis", model.mahalanobis(X_test))
            else:
                var model = EmpiricalCovariance[dtype](store_precision=store == 1, assume_centered=assume_centered)
                model.fit(X)
                emit(key + "_location", model.location_())
                emit(key + "_covariance", model.covariance_())
                emit(key + "_precision", model.get_precision())
                emit_scalar(key + "_score", model.score(X_test))
                emit(key + "_mahalanobis", model.mahalanobis(X_test))
        except e:
            emit_error(key, e)


def run[dtype: DType](mut tokens: Tokens) raises:
    var X = tokens.matrix[dtype]()
    var X_test = tokens.matrix[dtype]()
    var comparison = tokens.matrix[dtype]()
    var shrinkage = tokens.real()
    var batches = tokens.integer()
    var p = comparison.shape[0]
    var batch = nm.zeros[dtype](Shape(batches, p, p))
    for b in range(batches):
        var m = tokens.matrix[dtype]()
        for k in range(p * p):
            batch.unsafe_set(b * p * p + k, m.unsafe_get(k))

    for centered in range(2):
        try:
            emit(String("empirical_assume", centered), empirical_covariance(X, assume_centered=centered == 1))
        except e:
            emit_error(String("empirical_assume", centered), e)
        estimator_outputs(String("ec_assume", centered), X, X_test, centered == 1, None)
        estimator_outputs(String("sc_assume", centered), X, X_test, centered == 1, shrinkage)

    try:
        var model = EmpiricalCovariance[dtype]()
        model.fit(X)
        emit_scalar("fast_logdet", fast_logdet(model.covariance_()))
        emit_scalar("log_likelihood", log_likelihood(empirical_covariance(X_test), model.get_precision()))
        for norm_index in range(2):
            var norm = "frobenius" if norm_index == 0 else "spectral"
            for scaling in range(2):
                for squared in range(2):
                    emit_scalar(String("error_", norm, "_", scaling, squared),
                                model.error_norm(comparison, norm=norm, scaling=scaling == 1, squared=squared == 1))
    except e:
        emit_error("model", e)
    try:
        emit("shrunk_batch", shrunk_covariance(batch, shrinkage))
        emit("shrunk_single", shrunk_covariance(comparison, shrinkage))
    except e:
        emit_error("shrunk", e)


def main() raises:
    var tokens = Tokens(String(argv()[1]))
    if tokens.integer() == 32:
        run[DType.float32](tokens)
    else:
        run[DType.float64](tokens)
