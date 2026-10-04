"""Factorization probes on identical inputs, independent of empirical centering."""
from std.sys import argv
from parity_io import Tokens, emit, emit_scalar, emit_error
from moxie.covariance._empirical_covariance import fast_logdet, pinvh, to_float64
from moxie.covariance._graph_lasso_solver import _inverse
from moxie.covariance import EmpiricalCovariance


def run[dtype: DType](mut tokens: Tokens) raises:
    var A = tokens.matrix[dtype]()
    if tokens.position < len(tokens.items):
        # Expose the actual scoring workspace, before public Float32 snapshots.
        var test = tokens.matrix[dtype]()
        var model = EmpiricalCovariance[dtype]()
        _ = model.fit(A)
        emit("covariance", model._model._covariance64.value())
        emit("precision", model._model.precision64())
        emit_scalar("score", model.score(test))
        return
    try:
        emit_scalar("logdet", fast_logdet(A))
    except e:
        emit_error("logdet", e)
    try:
        var P = pinvh(A)
        emit("pinvh", P)
        emit_scalar("pinvh_logdet", fast_logdet(P))
    except e:
        emit_error("pinvh", e)
    try:
        var inverse = _inverse(to_float64(A))
        emit("inverse", inverse)
    except e:
        emit_error("inverse", e)


def main() raises:
    var tokens = Tokens(String(argv()[1]))
    if tokens.integer() == 32:
        run[DType.float32](tokens)
    else:
        run[DType.float64](tokens)
