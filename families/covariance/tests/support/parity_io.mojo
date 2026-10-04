"""Exact bit-pattern text protocol shared by covariance parity drivers.

Input: whitespace-separated tokens. Arrays are written by the Python harness
as `rows cols` followed by rows*cols IEEE-754 binary64 bit patterns (decimal
Int64), so no decimal parsing is involved. Output lines are
`key count bits...`, again bit patterns, decoded by the harness.
"""
import numojo as nm
from numojo import NDArray, Shape
from std.memory import bitcast


struct Tokens(Movable):
    """Sequential reader over the case file tokens."""

    var items: List[String]
    var position: Int

    def __init__(out self, path: String) raises:
        var text = open(path, "r").read()
        self.items = List[String]()
        for token in text.split():
            self.items.append(String(token))
        self.position = 0

    def next(mut self) raises -> String:
        if self.position >= len(self.items):
            raise Error("parity input exhausted")
        var token = self.items[self.position].copy()
        self.position += 1
        return token^

    def integer(mut self) raises -> Int:
        return Int(self.next())

    def real(mut self) raises -> Float64:
        return bitcast[DType.float64](Int64(Int(self.next())))

    def matrix[dtype: DType](mut self) raises -> NDArray[dtype]:
        var rows = self.integer()
        var cols = self.integer()
        var result = nm.zeros[dtype](Shape(rows, cols))
        for k in range(rows * cols):
            result.unsafe_set(k, self.real().cast[dtype]())
        return result^


def emit[dtype: DType](key: String, values: NDArray[dtype]):
    var line = String(key, " ", values.size)
    for k in range(values.size):
        line += String(" ", bitcast[DType.int64](Float64(values.unsafe_get(k))))
    print(line)


def emit_scalar(key: String, value: Float64):
    print(String(key, " 1 ", bitcast[DType.int64](value)))


def emit_error(key: String, error: Error):
    print(String(key, " error ", String(error).replace(" ", "_")))
