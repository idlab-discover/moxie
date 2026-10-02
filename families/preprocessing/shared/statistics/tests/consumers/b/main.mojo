from numojo import zeros, Shape
from std.testing import assert_equal, assert_true
from std.math import isnan
from moxie.preprocessing._moments import nanmean_at, nanmean_std_at, INPUT_PRECISION_API_VERSION, MomentsStateV1, update_moments_weighted, MOMENTS_API_VERSION
from moxie.preprocessing._contracts import FeatureMoments
from moxie.preprocessing._metadata import FeatureNames
from moxie.preprocessing._validation import validate_dense
from moxie.utils.validation import FittedState
from moxie._dense import sum_axis2d

def main() raises:
    var x = zeros[DType.float64](Shape(2, 1))
    var w = zeros[DType.float64](Shape(2))
    x.unsafe_set(0, 2)
    x.unsafe_set(1, 4)
    w.unsafe_set(0, 1)
    w.unsafe_set(1, -1)
    validate_dense(x)
    var names = FeatureNames()
    names.record_fit(2, 1)
    var outputs = names.get_feature_names_out()
    assert_equal(outputs[0], 'x0')
    var s = MomentsStateV1(1)
    s.count.unsafe_set(0, 1)
    update_moments_weighted(x, w, s, tile_rows=2)
    assert_equal(s.mean.unsafe_get(0), -2)
    assert_equal(s.new_sum.unsafe_get(0), -2)
    assert_true(isnan(s.variance.unsafe_get(0)))
    var snapshot = FeatureMoments(s.count, s.mean, s.variance)
    s.mean.unsafe_set(0, 999)
    assert_equal(snapshot.mean.unsafe_get(0), -2)
    var sums = sum_axis2d(x, 0)
    assert_equal(sums.unsafe_get(0), 6)
    assert_equal(MOMENTS_API_VERSION, 1)
    print('PASS isolated consumer b: signed batch mean -2; scaffold composes')

    var functional = zeros[DType.float32](Shape(3, 1))
    functional.unsafe_set(0, 100000000)
    functional.unsafe_set(1, 100000008)
    functional.unsafe_set(2, 100000016)
    var first = nanmean_std_at(functional, 0, 0)
    functional.unsafe_set(1, 100000016)
    functional.unsafe_set(2, 100000008)
    var second = nanmean_std_at(functional, 0, 0)
    assert_equal(INPUT_PRECISION_API_VERSION, 1)
    assert_equal(first.stddev, Float32(10.32795524597168))
    assert_equal(second.stddev, Float32(6.531972885131836))
    assert_equal(nanmean_at(functional, 0, 0), second.mean)
    print('PASS isolated input-precision consumer: distinct scale divisors')
