from std.testing import assert_equal, assert_true, assert_raises
from std.math import isnan
from numojo import zeros, Shape
from moxie.preprocessing._moments import MomentsStateV1, update_moments, update_moments_weighted, update_moments_scalar, ExtremaStateV1, update_extrema, robust_quantiles

def main() raises:
    var x = zeros[DType.float64](Shape(2, 1))
    x.unsafe_set(0, 1)
    x.unsafe_set(1, 2)
    var w = zeros[DType.float64](Shape(2))
    w.unsafe_set(0, 1)
    w.unsafe_set(1, -1)
    var s = MomentsStateV1(1)
    s.count.unsafe_set(0, 1)
    update_moments_weighted(x, w, s, tile_rows=1)
    assert_equal(s.mean.unsafe_get(0), -1)
    assert_equal(s.new_sum.unsafe_get(0), -1)
    assert_true(isnan(s.variance.unsafe_get(0)))
    var extrema = ExtremaStateV1(1)
    update_extrema(x, extrema)
    assert_equal(extrema.minimum.unsafe_get(0), 1)
    var q = robust_quantiles(x)
    assert_equal(q.median.unsafe_get(0), 1.5)
    assert_equal(q.lower.unsafe_get(0), 1.25)
    var plain = MomentsStateV1(1)
    update_moments(x, plain)
    assert_equal(plain.variance.unsafe_get(0), 0.25)
    assert_true(plain.scalar_count())
    assert_true(not plain.constant_at(0))
    var scalar = MomentsStateV1(1)
    update_moments_scalar(x, -1, scalar)
    assert_equal(scalar.mean.unsafe_get(0), 1.5)
    with assert_raises():
        update_moments_scalar(x, 0, scalar)
    print('PASS native moments')
