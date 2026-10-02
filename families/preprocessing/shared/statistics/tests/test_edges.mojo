from std.testing import assert_equal, assert_true, assert_raises
from std.math import isnan
from numojo import zeros, full, Shape, NDArrayStrides
from moxie.preprocessing._moments import MomentsStateV1, update_moments, update_moments_weighted, update_moments_scalar, ExtremaStateV1, update_extrema, robust_quantiles
from moxie.preprocessing._validation import require_dense_output

def checks[dtype: DType]() raises:
    var x = full[dtype](Shape(3, 2), 2)
    var state = MomentsStateV1(2)
    update_moments(x, state)
    assert_true(state.scalar_count())
    assert_true(state.constant_at(0))
    assert_equal(state.count.unsafe_get(0), 3)
    assert_equal(state.scalar_count_value(), 3)
    assert_equal(state.integer_count_at(0), 3)
    var snapshot = state.copy()
    update_moments(x, state)
    assert_equal(snapshot.count.unsafe_get(0), 3)
    assert_equal(state.count.unsafe_get(0), 6)
    var w = zeros[DType.float64](Shape(3))
    with assert_raises():
        update_moments_weighted(x, w, state)
    assert_equal(state.count.unsafe_get(0), 6)
    w.unsafe_set(0, Float64('nan'))
    with assert_raises():
        update_moments_weighted(x, w, state)
    w.unsafe_set(0, Float64('inf'))
    with assert_raises():
        update_moments_weighted(x, w, state)
    var bad_weights = zeros[DType.float64](Shape(3, 1))
    with assert_raises():
        update_moments_weighted(x, bad_weights, state)
    var short_weights = zeros[DType.float64](Shape(2))
    with assert_raises():
        update_moments_weighted(x, short_weights, state)
    with assert_raises():
        update_moments(x, state, tile_rows=0)
    assert_equal(state.count.unsafe_get(0), 6)
    var wrong = MomentsStateV1(1)
    with assert_raises():
        update_moments(x, wrong)
    var rank1 = zeros[dtype](Shape(2))
    with assert_raises():
        update_moments(rank1, state)
    var empty = zeros[dtype](Shape(0, 2))
    with assert_raises():
        update_moments(empty, state)
    x.unsafe_set(5, Float64('inf').cast[dtype]())
    with assert_raises():
        update_moments(x, state)
    assert_equal(state.count.unsafe_get(0), 6)
    x.unsafe_set(5, Float64('nan').cast[dtype]())
    var missing = MomentsStateV1(2)
    update_moments(x, missing)
    assert_true(not missing.scalar_count())
    with assert_raises():
        _ = missing.scalar_count_value()
    assert_equal(missing.integer_count_at(1), 2)
    assert_equal(missing.count.unsafe_get(0), 3)
    assert_equal(missing.count.unsafe_get(1), 2)
    with assert_raises():
        _ = missing.constant_at(2)
    var mean_only = MomentsStateV1(2, track_variance=False)
    update_moments(x, mean_only)
    with assert_raises():
        _ = mean_only.constant_at(0)
    var e = ExtremaStateV1(2)
    update_extrema(x, e)
    assert_equal(e.minimum.unsafe_get(0), 2)
    for i in range(3):
        x.unsafe_set(i * 2, Float64('nan').cast[dtype]())
    update_extrema(x, e)
    assert_true(isnan(e.minimum.unsafe_get(0)))
    assert_true(isnan(e.maximum.unsafe_get(0)))
    assert_true(isnan(e.max_abs.unsafe_get(0)))
    x.unsafe_set(0, 3)
    update_extrema(x, e)
    assert_true(isnan(e.minimum.unsafe_get(0)))
    var quantiles = robust_quantiles(x, column_budget_bytes=24)
    assert_equal(quantiles.median.unsafe_get(0), 3)
    with assert_raises():
        _ = robust_quantiles(x, column_budget_bytes=23)
    with assert_raises():
        _ = robust_quantiles(x, q_min=-1)
    with assert_raises():
        _ = robust_quantiles(x, q_min=80, q_max=20)
    with assert_raises():
        _ = robust_quantiles(x, q_max=Float64('nan'))
    # Missing old mean at zero count remains NaN; variance's zero branch differs.
    var poisoned = MomentsStateV1(2)
    poisoned.mean.unsafe_set(0, Float64('nan'))
    update_moments(x, poisoned)
    assert_true(isnan(poisoned.mean.unsafe_get(0)))
    assert_equal(poisoned.variance.unsafe_get(0), 0)
    # Mixed weights/input dtype is a supported engine mode.
    w.unsafe_set(0, 1)
    update_moments_weighted(x, w, missing)
    assert_true(missing.weighted)
    with assert_raises():
        _ = missing.integer_count_at(0)

def main() raises:
    checks[DType.float32]()
    checks[DType.float64]()
    var nan_count = MomentsStateV1(1)
    nan_count.count.unsafe_set(0, Float64('nan'))
    assert_true(not nan_count.scalar_count())
    with assert_raises():
        _ = nan_count.scalar_count_value()
    require_dense_output(False)
    with assert_raises():
        require_dense_output(True)
    print('PASS lifecycle, validation, missingness, extrema, quantile budget, dense scope')
