"""Calibration's array boundary preserves shared isotonic behavior."""
import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie._calibration_algorithms import _IsotonicCalibration, _array
from moxie.isotonic import IsotonicRegression


def main() raises:
    var xs: List[Float64] = [3, 0, 1e-16, 2, 2, 1, 4]
    var ys: List[Float64] = [1, 1, 0, 0, 1, 1, 0]
    var ws: List[Float64] = [1, 1, 3, 2, 1, 0, 0]
    var query: List[Float64] = [-1, 0, 1e-16, 0.5, 1, 2, 2.5, 3, 4]
    var native = _IsotonicCalibration()
    var reference = IsotonicRegression[DType.float64](out_of_bounds="clip")
    var X = _array(xs)
    var y = _array(ys)
    var w = _array(ws)
    native.fit(X, y, w.copy())
    _ = reference.fit(xs, ys, sample_weight=ws)
    var actual = native.predict(_array(query))
    var expected = reference.predict(query)
    for i in range(len(query)):
        assert_true(abs(actual.item(i) - expected[i]) <= 2e-12 * (1 + abs(expected[i])))
    assert_equal(native.X_thresholds_.item(0), 0)
    assert_equal(actual.item(0), actual.item(1))
    assert_equal(actual.item(7), actual.item(8))
    X[nm.Item(0)] = 100
    y[nm.Item(0)] = 100
    w[nm.Item(0)] = -1
    with assert_raises():
        native.fit(X, y, w.copy())
    var retained = native.predict(_array(query))
    for i in range(len(query)):
        assert_equal(retained.item(i), actual.item(i))
    print("PASS: shared isotonic duplicates, resolution, weights, clipping and failed fit")
