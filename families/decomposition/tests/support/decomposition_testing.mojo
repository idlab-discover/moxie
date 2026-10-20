"""Shared helpers for the native differential drivers (validation only).

Drivers read a whitespace-separated input file written by the Python oracle and
print `name value` lines that the oracle parses. Nothing here is library code.
"""
from numojo import NDArray, Shape, zeros


def emit[dtype: DType](name: String, x: NDArray[dtype]) raises:
    """Print every logical element of x in C order as `name value` lines."""
    if x.size == 0: return
    var flat = x.flatten()
    for i in range(flat.size): print(name, Float64(flat.unsafe_get(i)))


def emit_scalar(name: String, value: Float64):
    print(name, value)


def read_matrix[dtype: DType](path: String) raises -> NDArray[dtype]:
    """Read `rows cols v0 v1 ...` (C order) into an owned (rows, cols) array."""
    var text: String
    with open(path, "r") as handle:
        text = handle.read()
    var tokens = text.split()
    if len(tokens) < 2:
        raise Error("ValueError: matrix file " + path + " has no shape header")
    var rows = Int(String(tokens[0]))
    var cols = Int(String(tokens[1]))
    if len(tokens) != 2 + rows * cols:
        raise Error("ValueError: matrix file " + path + " has the wrong element count")
    var x = zeros[dtype](Shape(rows, cols))
    for i in range(rows * cols):
        x.unsafe_set(i, Scalar[dtype](Float64(String(tokens[2 + i]))))
    return x^
