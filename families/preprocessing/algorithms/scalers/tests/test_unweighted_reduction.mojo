"""Unweighted fast-axis cancellation/overflow diagnostic on borrowed storage."""
from numojo import Shape, zeros
from moxie.preprocessing._moments import MomentsStateV1, update_moments
from moxie.preprocessing._dense_transforms import transform


def probe(magnitude: Float64, tile: Int, track_variance: Bool) raises:
    # Pairwise sum is zero; serial sum overflows for magnitude=1e308.
    var x = zeros[DType.float64](Shape(8, 1))
    x.unsafe_set(0, magnitude)
    x.unsafe_set(2, magnitude)
    x.unsafe_set(3, -magnitude)
    x.unsafe_set(4, -magnitude)
    var state = MomentsStateV1(1, track_variance=track_variance)
    update_moments(x, state, tile_rows=tile)
    var divisor = zeros[DType.float64](Shape(1))
    divisor.unsafe_set(0, 1)
    # StandardScaler(with_std=False)'s centering composition.
    var centered = transform(x, state.mean, divisor)
    print("unweighted", magnitude, tile, Int(track_variance),
          state.count.unsafe_get(0), state.mean.unsafe_get(0),
          state.variance.unsafe_get(0), state.new_sum.unsafe_get(0),
          centered.unsafe_get(0), centered.unsafe_get(1))


def main() raises:
    for tile in [1, 3, 256]:
        for track_variance in [False, True]:
            probe(1e100, tile, track_variance)
            probe(1e308, tile, track_variance)
