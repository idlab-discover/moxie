"""Private Gram path differential driver (validation tooling only)."""
from std.sys import argv
from numojo import Shape, zeros
from parity_io import Tokens, emit, emit_error
from moxie.covariance._lars_gram import lars_gram


def main() raises:
    var tokens = Tokens(String(argv()[1]))
    var gram = tokens.matrix[DType.float64]()
    var column = tokens.matrix[DType.float64]()
    var xy = zeros[DType.float64](Shape(column.size))
    for j in range(xy.size):
        xy.unsafe_set(j, column.unsafe_get(j))
    var penalty = tokens.real()
    var eps = tokens.real()
    try:
        emit("coef", lars_gram(gram, xy, penalty, eps))
        emit("gram_after", gram)
        emit("xy_after", xy)
    except error:
        emit_error("coef", error)
