"""Consumer diagnostics; these are not public estimator acceptance tests."""
from std.testing import assert_equal, assert_true
from std.math import isnan
from numojo import Shape, zeros
from moxie.preprocessing._moments import MomentsStateV1, update_moments_weighted
from moxie.preprocessing._dense_transforms import transform

def original_blocker() raises:
    var x = zeros[DType.float64](Shape(2, 1))
    var w = zeros[DType.float64](Shape(2))
    w.unsafe_set(0, 1)
    w.unsafe_set(1, -1)
    for multiplier in range(1, 3):
        x.unsafe_set(0, Float64(multiplier))
        x.unsafe_set(1, Float64(2 * multiplier))
        var state = MomentsStateV1(1)
        state.count.unsafe_set(0, 1)
        update_moments_weighted(x, w, state, tile_rows=1)
        assert_equal(state.count.unsafe_get(0), 1)
        assert_equal(state.mean.unsafe_get(0), -Float64(multiplier))
        assert_true(isnan(state.variance.unsafe_get(0)))
    var shift = zeros[DType.float64](Shape(1))
    var divisor = zeros[DType.float64](Shape(1))
    divisor.unsafe_set(0, 2)
    var transformed = transform(x, shift, divisor)
    assert_equal(transformed.unsafe_get(0), 1)
    assert_equal(transformed.unsafe_get(1), 2)
    print("PASS: original information-loss blocker resolved by pinned raw-batch update")

def cancellation[d: DType](tile: Int) raises:
    var x = zeros[d](Shape(8, 1))
    var w = zeros[d](Shape(8))
    for i in range(8):
        x.unsafe_set(i, Float64(i + 1).cast[d]())
    w.unsafe_set(0, Float64(1e16).cast[d]())
    w.unsafe_set(1, 1)
    w.unsafe_set(2, Float64(-1e16).cast[d]())
    w.unsafe_set(3, 1)
    var state = MomentsStateV1(1)
    state.count.unsafe_set(0, 1)
    update_moments_weighted(x, w, state, tile_rows=tile)
    print("cancellation", "f32" if d == DType.float32 else "f64", tile,
          state.count.unsafe_get(0), state.mean.unsafe_get(0), state.variance.unsafe_get(0))

def main() raises:
    original_blocker()
    for tile in [1, 3, 256]:
        cancellation[DType.float32](tile)
        cancellation[DType.float64](tile)
