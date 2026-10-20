"""ComplementNB norm=True tie regression (sklearn predicts the first class).

With alpha floored to 1e-10, sklearn's feature_log_prob_ row for class -3 is
0x1.974ebc53ddf2fp-2, 0x1.3458a1d611069p-1 and row 4 is exactly 0.5. Mojo 1.0's
std.math.log is ~1e-10 relative off and broke this tie; the family log is now
within one ulp of glibc, so the weights are bit-identical and both scores are 3.
"""
from std.memory import bitcast
from std.testing import assert_equal
from moxie._nb_counts import ComplementNB

def main() raises:
    var X: List[List[Float64]] = [[6.5, 6.5], [6.0, 4.5]]
    var y: List[Int] = [-3, 4]
    var query: List[List[Float64]] = [[3.0, 3.0]]
    var model = ComplementNB(alpha=0, force_alpha=False, fit_prior=False, norm=True)
    _ = model.fit(X, y)
    assert_equal(bitcast[DType.uint64](model.feature_log_prob_.item(0, 0)), UInt64(0x3FD974EBC53DDF2F))
    assert_equal(bitcast[DType.uint64](model.feature_log_prob_.item(0, 1)), UInt64(0x3FE3458A1D611069))
    assert_equal(model.feature_log_prob_.item(1, 0), 0.5)
    assert_equal(model.predict(query)[0], -3)
    print("PASS ComplementNB normalization tie regression")
