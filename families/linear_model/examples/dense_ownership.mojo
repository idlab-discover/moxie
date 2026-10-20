"""Borrow training data and retain an independent native snapshot."""
from numojo import ones, zeros, Shape
from moxie.linear_model._contracts import validate_xy, snapshot

def main() raises:
    var X = ones[DType.float64](Shape(4, 2))
    var y = zeros[DType.float64](Shape(4))
    validate_xy(X, y)
    var saved = snapshot(X)
    X.unsafe_set(0, 9)
    print("Independent snapshot value:", saved.unsafe_get(0))
