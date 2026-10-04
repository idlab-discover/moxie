"""Float32 arithmetic boundaries and observable failed-fit state."""
from std.sys import argv
from numojo import Shape, NDArrayStrides
from parity_io import Tokens, emit, emit_scalar, emit_error
from moxie.covariance import LedoitWolf, OAS, ledoit_wolf_shrinkage

def run_LedoitWolf[dtype: DType](mut tokens: Tokens) raises:
    var X = tokens.matrix[dtype]()
    var Y = tokens.matrix[dtype]()
    var centered = tokens.integer() != 0
    var store = tokens.integer() != 0
    var block = tokens.integer()
    var refit = tokens.integer() != 0
    var layout = tokens.integer()
    if layout == 1:
        Y = Y.view_with_layout(Shape(Y.shape[1], Y.shape[0]), NDArrayStrides(1, Y.shape[1]), 0)
    elif layout == 2:
        Y = Y.view_with_layout(Shape(Y.shape[0], Y.shape[1]), NDArrayStrides(-Y.shape[1], 1), (Y.shape[0] - 1) * Y.shape[1])
    var model = LedoitWolf[dtype](assume_centered=centered, store_precision=store, block_size=block)
    if refit:
        model.fit(X)
        emit("before_covariance", model.covariance_())
        emit("before_precision", model.get_precision())
    try:
        model.fit(Y)
        emit_scalar("success", 1)
    except e:
        emit_error("fit", e)
    try:
        emit("location", model.location_())
    except e:
        emit_error("location", e)
    try:
        emit("covariance", model.covariance_())
    except e:
        emit_error("covariance", e)
    try:
        emit("precision", model.get_precision())
    except e:
        emit_error("precision", e)
    try:
        emit_scalar("shrinkage", model.shrinkage_())
    except e:
        emit_error("shrinkage", e)
    try:
        emit_scalar("n_features_in", Float64(model.n_features_in_()))
    except e:
        emit_error("n_features_in", e)
    emit_scalar("coefficient", ledoit_wolf_shrinkage(Y, assume_centered=centered, block_size=block))

def run_OAS[dtype: DType](mut tokens: Tokens) raises:
    var X = tokens.matrix[dtype]()
    var Y = tokens.matrix[dtype]()
    var centered = tokens.integer() != 0
    var store = tokens.integer() != 0
    var block = tokens.integer()
    var refit = tokens.integer() != 0
    var layout = tokens.integer()
    if layout == 1:
        Y = Y.view_with_layout(Shape(Y.shape[1], Y.shape[0]), NDArrayStrides(1, Y.shape[1]), 0)
    elif layout == 2:
        Y = Y.view_with_layout(Shape(Y.shape[0], Y.shape[1]), NDArrayStrides(-Y.shape[1], 1), (Y.shape[0] - 1) * Y.shape[1])
    var model = OAS[dtype](assume_centered=centered, store_precision=store)
    if refit:
        model.fit(X)
        emit("before_covariance", model.covariance_())
        emit("before_precision", model.get_precision())
    try:
        model.fit(Y)
        emit_scalar("success", 1)
    except e:
        emit_error("fit", e)
    try:
        emit("location", model.location_())
    except e:
        emit_error("location", e)
    try:
        emit("covariance", model.covariance_())
    except e:
        emit_error("covariance", e)
    try:
        emit("precision", model.get_precision())
    except e:
        emit_error("precision", e)
    try:
        emit_scalar("shrinkage", model.shrinkage_())
    except e:
        emit_error("shrinkage", e)
    try:
        emit_scalar("n_features_in", Float64(model.n_features_in_()))
    except e:
        emit_error("n_features_in", e)

def run[dtype: DType](mut tokens: Tokens, kind: Int) raises:
    if kind == 0:
        run_LedoitWolf[dtype](tokens)
    else:
        run_OAS[dtype](tokens)


def main() raises:
    var tokens = Tokens(String(argv()[1]))
    var bits = tokens.integer()
    var kind = tokens.integer()
    if bits == 32:
        run[DType.float32](tokens, kind)
    else:
        run[DType.float64](tokens, kind)
