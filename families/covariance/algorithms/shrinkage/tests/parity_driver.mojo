"""Native side of the shrinkage parity harness (tests/parity.py writes the cases)."""
from numojo import NDArray
from std.sys import argv
from parity_io import Tokens, emit, emit_scalar, emit_error
from moxie.covariance import LedoitWolf, OAS, ledoit_wolf, ledoit_wolf_shrinkage, oas


def run[dtype: DType](mut tokens: Tokens) raises:
    var X = tokens.matrix[dtype]()
    var X_test = tokens.matrix[dtype]()
    for centered in range(2):
        var assume = centered == 1
        var tag = String("_assume", centered)
        for block_size in [1000, 1, 3]:
            try:
                emit_scalar(String("lw_shrinkage", tag, "_bs", block_size), ledoit_wolf_shrinkage(X, assume_centered=assume, block_size=block_size))
            except e:
                emit_error(String("lw_shrinkage", tag, "_bs", block_size), e)
        try:
            var result = ledoit_wolf(X, assume_centered=assume, block_size=2)
            emit("ledoit_wolf" + tag + "_covariance", result[0])
            emit_scalar("ledoit_wolf" + tag + "_shrinkage", result[1])
            var other = oas(X, assume_centered=assume)
            emit("oas" + tag + "_covariance", other[0])
            emit_scalar("oas" + tag + "_shrinkage", other[1])
        except e:
            emit_error("functions" + tag, e)
        for store in range(2):
            var key = String(tag, "_store", store)
            try:
                var lw = LedoitWolf[dtype](store_precision=store == 1, assume_centered=assume)
                lw.fit(X)
                emit("LW" + key + "_location", lw.location_())
                emit("LW" + key + "_covariance", lw.covariance_())
                emit_scalar("LW" + key + "_shrinkage", lw.shrinkage_())
                emit("LW" + key + "_precision", lw.get_precision())
                emit_scalar("LW" + key + "_score", lw.score(X_test))
                emit("LW" + key + "_mahalanobis", lw.mahalanobis(X_test))
                var model = OAS[dtype](store_precision=store == 1, assume_centered=assume)
                model.fit(X)
                emit("OAS" + key + "_location", model.location_())
                emit("OAS" + key + "_covariance", model.covariance_())
                emit_scalar("OAS" + key + "_shrinkage", model.shrinkage_())
                emit("OAS" + key + "_precision", model.get_precision())
                emit_scalar("OAS" + key + "_score", model.score(X_test))
                emit("OAS" + key + "_mahalanobis", model.mahalanobis(X_test))
            except e:
                emit_error("estimators" + key, e)


def main() raises:
    var tokens = Tokens(String(argv()[1]))
    if tokens.integer() == 32:
        run[DType.float32](tokens)
    else:
        run[DType.float64](tokens)
