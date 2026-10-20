"""Producer fixture, not an estimator implementation."""
from numojo import NDArray, Shape, zeros
from moxie.semi_supervised._contracts import snapshot_features

def produce() raises -> NDArray[DType.float64]:
    var caller = zeros[DType.float64](Shape(2, 2))
    caller.unsafe_set(0, 3)
    var owned = snapshot_features(caller, 2)
    caller.unsafe_set(0, 99)
    return owned^
