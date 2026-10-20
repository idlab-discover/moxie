import numojo as nm
from std.testing import assert_equal, assert_raises
from moxie.feature_selection._contracts import ScoreResult
from moxie.feature_selection._univariate_policy import UnivariatePolicy

def percentile_count(n: Int, percentile: Float64) raises -> Int:
    var scores = nm.zeros[DType.float64](nm.Shape(n))
    for i in range(n):
        scores.unsafe_set(i, Float64(i + 1))
    var mask = UnivariatePolicy[DType.float64]("percentile", percentile).select(ScoreResult[DType.float64](scores^))
    var count = 0
    for selected in mask:
        count += Int(selected)
    return count

def main() raises:
    # np.percentile computes (n - 1) * ((100 - p) / 100); for these sizes the
    # interpolated threshold falls just below a score and sklearn keeps one
    # feature more than int(n * p / 100).
    assert_equal(percentile_count(51, 42), 22)
    assert_equal(percentile_count(91, 30), 28)
    assert_equal(percentile_count(46, 30), 14)
    assert_equal(percentile_count(10, 50), 5)
    var scores = nm.ones[DType.float64](nm.Shape(4))
    var p = nm.zeros[DType.float64](nm.Shape(4))
    p.unsafe_set(0, 0.01)
    p.unsafe_set(1, 0.04)
    p.unsafe_set(2, 0.03)
    p.unsafe_set(3, 0.9)
    var result = ScoreResult[DType.float64](scores^, p^)
    var policy = UnivariatePolicy[DType.float64]("k_best", 2)
    var mask = policy.select(result)
    assert_equal(mask[0], False)
    assert_equal(mask[2], True)
    policy.mode = "percentile"
    policy.param = 50
    mask = policy.select(result)
    assert_equal(mask[0], True)
    assert_equal(mask[2], False)
    policy.mode = "fdr"
    policy.param = 0.06
    mask = policy.select(result)
    assert_equal(mask[0], True)
    assert_equal(mask[1], True)
    assert_equal(mask[2], True)
    assert_equal(mask[3], False)
    policy.mode = "fpr"
    policy.param = 0.04
    mask = policy.select(result)
    assert_equal(mask[1], False)
    policy.mode = "fwe"
    mask = policy.select(result)
    assert_equal(mask[0], False)
    policy.param = -1
    with assert_raises():
        _ = policy.select(result)
    var only_scores = ScoreResult[DType.float64](nm.ones[DType.float64](nm.Shape(2)))
    policy.param = 0.05
    with assert_raises():
        _ = policy.select(only_scores)
    print("policy contracts passed")
