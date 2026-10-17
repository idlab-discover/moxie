from std.testing import assert_equal, assert_false, assert_true, assert_raises
from moxie.isotonic import IsotonicRegression, isotonic_regression, check_increasing
from moxie.base import clone

def main() raises:
    var x: List[Float32] = [0, 0.0000001, 1, 2]
    var y: List[Float32] = [0, 2, 3, 4]
    var model = IsotonicRegression[DType.float32](out_of_bounds="clip")
    _ = model.fit(x, y)
    assert_equal(len(model.X_thresholds_), 3)
    assert_equal(model.y_thresholds_[0], Float32(1))
    var predicted: List[Float32] = model.predict(x)
    assert_equal(predicted[3], Float32(4))
    var fitted: List[Float32] = isotonic_regression(y)
    assert_equal(fitted[3], Float32(4))
    assert_true(check_increasing(x, y))
    var copied = model.copy()
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    _ = model.fit(List[Float32]([Float32(2)]), List[Float32]([Float32(9)]))
    assert_equal(copied.predict(x)[3], Float32(4))
    assert_equal(model.get_feature_names_out(List[String]([String("ignored")]))[0], "isotonicregression0")
    assert_equal(copied.score(x, predicted), 1.0)
    with assert_raises():
        _ = copied.fit(x, List[Float32]([Float32(1)]))
    assert_equal(copied.X_max_, Float32(2))
    print("Float32 lifecycle and tie contracts passed")
