"""Validation-only probe of the excluded NuMojo solve, not a runtime solver."""
from numojo import ones, Shape, Item
from numojo.routines.linalg.solving import solve
from moxie.linear_model._contracts import validate_xy, validate_weights


def probe[dtype: DType]() raises:
    var X = ones[dtype](Shape(2, 2))
    var y = ones[dtype](Shape(2))
    var weights = ones[dtype](Shape(2))
    validate_xy(X, y)
    validate_weights(weights, 2)
    var h = ones[dtype](Shape(2, 2))
    var rhs = ones[dtype](Shape(2, 1))
    # Duplicate columns of X: X.T @ X / 2 = [[1, 1], [1, 1]].
    var singular = solve(h, rhs)
    print("singular", singular[Item(0, 0)], singular[Item(1, 0)])
    h[Item(0, 0)] = 0
    h[Item(1, 1)] = 0
    # Nonsingular symmetric matrix requiring pivoting.
    var pivot = solve(h, rhs)
    print("pivot", pivot[Item(0, 0)], pivot[Item(1, 0)])


def main() raises:
    probe[DType.float32]()
    probe[DType.float64]()
