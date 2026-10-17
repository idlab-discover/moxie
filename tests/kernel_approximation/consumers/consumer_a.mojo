"""Isolated ownership probe, not an estimator implementation."""
from numojo import NDArray, Item, Shape, zeros
from moxie._kernel_features import owned_snapshot, FeatureMapState
from moxie._dense import row_squared_norms
from moxie._random import RandomState
from std.testing import assert_equal

struct ProbeA:
    var state: FeatureMapState
    def __init__(out self):
        self.state = FeatureMapState()
    def fit[dtype: DType](mut self, x: NDArray[dtype]) raises -> NDArray[dtype]:
        self.state.record_fit(x, 1)
        return owned_snapshot(x)

def run() raises:
    var x = zeros[DType.float64](Shape(1, 2))
    x[Item(0, 0)] = 3
    x[Item(0, 1)] = 4
    var model = ProbeA()
    var saved = model.fit(x)
    x[Item(0, 0)] = 99
    var norms = row_squared_norms(saved)
    assert_equal(norms.item(0), 25)
    var rng = RandomState(42)
    assert_equal(rng.random_sample(), Float64(0.3745401188473625))
    print("PASS: isolated dense/RNG consumer")


def main() raises:
    run()
