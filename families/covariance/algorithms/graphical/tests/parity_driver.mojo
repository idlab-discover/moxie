"""Native side of the graphical lasso parity harness."""
import numojo as nm
from numojo import NDArray, Shape
from std.sys import argv
from parity_io import Tokens, emit, emit_scalar, emit_error
from moxie.covariance import GraphicalLasso, graphical_lasso


def costs(values: List[Tuple[Float64, Float64]]) raises -> NDArray[DType.float64]:
    var result = nm.zeros[DType.float64](Shape(2 * len(values)))
    for k in range(len(values)):
        result.unsafe_set(2 * k, values[k][0])
        result.unsafe_set(2 * k + 1, values[k][1])
    return result^


def run[dtype: DType](mut tokens: Tokens) raises:
    var X = tokens.matrix[dtype]()
    var X_test = tokens.matrix[dtype]()
    var emp = tokens.matrix[dtype]()
    var alpha = tokens.real()
    var max_iter = tokens.integer()
    var mode = tokens.next()
    for centered in range(2):
        var key = String("gl_assume", centered)
        try:
            var model = GraphicalLasso[dtype](alpha, assume_centered=centered == 1, max_iter=max_iter, mode=mode.copy())
            model.fit(X)
            emit(key + "_location", model.location_())
            emit(key + "_covariance", model.covariance_())
            emit(key + "_precision", model.get_precision())
            emit(key + "_costs", costs(model.costs_()))
            emit_scalar(key + "_n_iter", Float64(model.n_iter_()))
            emit_scalar(key + "_score", model.score(X_test))
            emit(key + "_mahalanobis", model.mahalanobis(X_test))
        except e:
            emit_error(key, e)
    for zero in range(2):
        var key = String("fn_alpha", zero)
        try:
            var result = graphical_lasso(emp, 0.0 if zero == 1 else alpha, max_iter=max_iter, mode=mode.copy())
            emit(key + "_covariance", result.covariance)
            emit(key + "_precision", result.precision)
            emit(key + "_costs", costs(result.costs))
            emit_scalar(key + "_n_iter", Float64(result.n_iter))
        except e:
            emit_error(key, e)


def main() raises:
    var tokens = Tokens(String(argv()[1]))
    if tokens.integer() == 32:
        run[DType.float32](tokens)
    else:
        run[DType.float64](tokens)
