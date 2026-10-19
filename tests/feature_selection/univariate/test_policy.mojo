import numojo as nm
from std.testing import assert_equal, assert_raises
from moxie.feature_selection._contracts import ScoreResult
from moxie.feature_selection._univariate_policy import UnivariatePolicy

def main() raises:
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
