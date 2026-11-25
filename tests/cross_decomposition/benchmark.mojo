"""Time dense fits with fixture loading and model construction excluded."""

from std.sys import argv
from std.time import perf_counter_ns

from moxie.cross_decomposition import CCA, PLSCanonical, PLSRegression, PLSSVD
from moxie.cross_decomposition._base import CrossDecompositionTransformer
from numojo import NDArray, Shape, zeros


def read_array(
    path: String, rows: Int, columns: Int
) raises -> NDArray[DType.float64]:
    var width = max(1, columns)
    var matrix = zeros[DType.float64](Shape(rows, width))
    var contents = open(path, "r").read()
    var lines = contents.split("\n")
    if len(lines) != rows + 1 or lines[rows].byte_length() != 0:
        raise Error("Unexpected row count in " + path)
    for row in range(rows):
        var fields = lines[row].split(",")
        if len(fields) != width:
            raise Error("Unexpected column count in " + path)
        for column in range(width):
            matrix.unsafe_set(
                row * width + column, Float64(String(fields[column]))
            )
    if columns == 0:
        return matrix.reshape(Shape(rows))
    return matrix^


def timed_fit[
    E: CrossDecompositionTransformer & Deinitable
](var estimator: E, X: E.Input, y: E.Target, loops: Int) raises -> E:
    if loops == 0:
        _ = estimator.fit(X, y)
    else:
        var start = perf_counter_ns()
        for _ in range(loops):
            _ = estimator.fit(X, y)
        print(Float64(perf_counter_ns() - start) / 1e9)
    return estimator^


def emit_weights(weights: NDArray[DType.float64]) raises:
    for row in range(weights.shape[0]):
        for column in range(weights.shape[1]):
            print(
                weights.unsafe_get(
                    row * weights.strides[0] + column * weights.strides[1]
                )
            )


def main() raises:
    var args = argv()
    var name = String(args[1])
    var components = Int(String(args[2]))
    var loops = Int(String(args[3]))
    var rows = Int(String(args[4]))
    var X = read_array(String(args[5]), rows, Int(String(args[6])))
    var y = read_array(String(args[7]), rows, Int(String(args[8])))
    if name == "PLSSVD":
        var estimator = timed_fit(PLSSVD(components), X, y, loops)
        if loops == 0:
            emit_weights(estimator.x_weights_)
            emit_weights(estimator.y_weights_)
    elif name == "PLSRegression":
        var estimator = timed_fit(PLSRegression(components), X, y, loops)
        if loops == 0:
            emit_weights(estimator.x_weights_)
            emit_weights(estimator.y_weights_)
    elif name == "CCA":
        var estimator = timed_fit(CCA(components), X, y, loops)
        if loops == 0:
            emit_weights(estimator.x_weights_)
            emit_weights(estimator.y_weights_)
    elif name == "PLSCanonical-nipals" or name == "PLSCanonical-svd":
        var algorithm = "svd" if name == "PLSCanonical-svd" else "nipals"
        var estimator = timed_fit(
            PLSCanonical(components, algorithm=algorithm), X, y, loops
        )
        if loops == 0:
            emit_weights(estimator.x_weights_)
            emit_weights(estimator.y_weights_)
    else:
        raise Error("Unknown estimator: " + name)
