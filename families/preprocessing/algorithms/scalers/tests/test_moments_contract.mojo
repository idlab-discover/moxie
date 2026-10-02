"""Contract probe only; this is not an estimator implementation."""
from std.testing import assert_equal, assert_true, assert_raises
from std.math import isnan
from numojo import Shape, full, zeros
from moxie.preprocessing._contracts import FeatureMoments
from moxie.preprocessing._validation import validate_dense, require_dense_output

def main() raises:
    var count = zeros[DType.float64](Shape(1))
    var mean = full[DType.float64](Shape(1), Float64("-inf"))
    var variance = full[DType.float64](Shape(1), Float64("nan"))
    var first = FeatureMoments(count, mean, variance)
    var second = FeatureMoments(count, mean, variance)
    assert_equal(first.count.item(0), second.count.item(0))
    assert_equal(first.mean.item(0), second.mean.item(0))
    assert_true(isnan(first.variance.item(0)))
    assert_true(isnan(second.variance.item(0)))
    count.unsafe_set(0, 8)
    assert_equal(first.count.item(0), 0)
    var copied = first.copy()
    copied.count.unsafe_set(0, 9)
    assert_equal(first.count.item(0), 0)
    var x = zeros[DType.float64](Shape(2, 1))
    validate_dense(x, allow_nan=True)
    with assert_raises():
        require_dense_output(True)
    print("PASS: pinned moments storage and sparse-output rejection")
