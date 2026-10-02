"""Shared moments storage example, not a fitted scaler."""
from numojo import Shape, full, zeros
from moxie.preprocessing._contracts import FeatureMoments

def main() raises:
    var count = full[DType.float64](Shape(1), 2)
    var mean = full[DType.float64](Shape(1), 3)
    var variance = zeros[DType.float64](Shape(1))
    var snapshot = FeatureMoments(count, mean, variance)
    print("Stored feature mean:", snapshot.mean.item(0))
