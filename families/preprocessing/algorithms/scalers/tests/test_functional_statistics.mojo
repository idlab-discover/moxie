"""Positive consumer regression for the distinct input-precision owner API."""
from numojo import Shape, zeros
from moxie.preprocessing._moments import MomentsStateV1, update_moments, nanmean_std_at
from moxie.preprocessing._dense_transforms import transform


def probe[d: DType](permutation: Int, tile: Int) raises:
    var x = zeros[d](Shape(3, 1))
    x.unsafe_set(0, Float64(100000000).cast[d]())
    x.unsafe_set(1, Float64(100000008 if permutation == 0 else 100000016).cast[d]())
    x.unsafe_set(2, Float64(100000016 if permutation == 0 else 100000008).cast[d]())
    var state = MomentsStateV1(1)
    update_moments(x, state, tile_rows=tile)
    var shift = zeros[d](Shape(1))
    var divisor = zeros[d](Shape(1))
    divisor.unsafe_set(0, nanmean_std_at(x, 0, 0).stddev)
    var result = transform(x, shift, divisor)
    print("functional_statistics", "f32" if d == DType.float32 else "f64",
          permutation, tile, state.count.unsafe_get(0), state.mean.unsafe_get(0),
          state.variance.unsafe_get(0), state.new_sum.unsafe_get(0),
          Float64(divisor.unsafe_get(0)), result.unsafe_get(0), result.unsafe_get(1),
          result.unsafe_get(2))


def main() raises:
    for tile in [1, 3, 256]:
        for permutation in range(2):
            probe[DType.float32](permutation, tile)
            probe[DType.float64](permutation, tile)
