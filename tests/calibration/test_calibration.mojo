from std.testing import assert_equal, assert_true, assert_raises
import numojo as nm
from moxie.calibration import CalibratedClassifierCV, calibration_curve
from moxie.base import clone
from moxie.calibration import CalibrationCV
from moxie._calibration_cv import _clip_rounding
from std.utils import Variant
from moxie._calibration_algorithms import (
    Array,
    _SigmoidCalibration,
    _IsotonicCalibration,
    _TemperatureScaling,
)
from calibration_centroid import Centroid


def array(values: List[Float64]) raises -> Array:
    var result = nm.zeros[DType.float64](nm.Shape(len(values)))
    for i in range(len(values)):
        result[nm.Item(i)] = values[i]
    return result^


def main() raises:
    var x = array([5.0, -4.0, 1.0])
    var y = array([1.0, -1.0, -1.0])
    var sigmoid = _SigmoidCalibration()
    with assert_raises():
        _ = sigmoid.predict(x)
    sigmoid.fit(x, y)
    assert_true(abs(sigmoid.a_ + 0.20261450632534656) < 1e-6)
    assert_true(abs(sigmoid.b_ - 0.6523666844521088) < 1e-6)
    var bad = array([0.0, 0.0, 0.0])
    with assert_raises():
        sigmoid.fit(x, y, bad.copy())
    var isotonic = _IsotonicCalibration()
    with assert_raises():
        _ = isotonic.predict(x)
    var ix = array([0.0, 1.0, 1.0, 2.0, 3.0])
    var iy = array([0.0, 1.0, 0.0, 0.0, 1.0])
    isotonic.fit(ix, iy)
    var query = array([-1.0, 1.0, 1.5, 4.0])
    var ip = isotonic.predict(query)
    assert_equal(ip.item(0), 0)
    assert_true(abs(ip.item(1) - 1.0 / 3) < 1e-14)
    assert_true(abs(ip.item(2) - 1.0 / 3) < 1e-14)
    assert_equal(ip.item(3), 1)
    var truth = array([0.0, 0.0, 1.0, 1.0])
    var probabilities = array([0.0, 0.5, 0.5, 1.0])
    var curve = calibration_curve(truth, probabilities, n_bins=2)
    assert_true(abs(curve[0].item(0) - 1.0 / 3) < 1e-14)
    assert_equal(curve[0].item(1), 1)
    with assert_raises():
        _ = calibration_curve(truth, probabilities, n_bins=0)
    var temperature = _TemperatureScaling()
    with assert_raises():
        _ = temperature.predict(probabilities)
    temperature.fit(probabilities, truth)
    var tp = temperature.predict(probabilities)
    for i in range(4):
        assert_true(abs(tp.item(i, 0) + tp.item(i, 1) - 1) < 1e-14)
    var X = nm.zeros[DType.float64](nm.Shape(18, 2))
    var Y = nm.zeros[DType.float64](nm.Shape(18))
    for i in range(18):
        X[nm.Item(i, 0)] = Float64(i % 3) + Float64(i % 5) / 4
        X[nm.Item(i, 1)] = Float64(i % 7)
        Y[nm.Item(i)] = Float64(i % 3) * 3 - 2
    var model = CalibratedClassifierCV(Centroid(), cv=3)
    with assert_raises():
        _ = model.predict(X)
    _ = model.fit(X, Y)
    assert_equal(len(model.calibrated_classifiers_), 3)
    assert_equal(model.classes_.item(0), -2)
    var p = model.predict_proba(X)
    for i in range(18):
        assert_true(abs(p.item(i, 0) + p.item(i, 1) + p.item(i, 2) - 1) < 1e-14)
    var copied = clone(model)
    assert_true(not copied.__sklearn_is_fitted__())
    assert_true(not copied.estimator.__sklearn_is_fitted__())
    var params = model.get_params()
    params.method = "isotonic"
    params.ensemble = False
    _ = model.set_params(params)
    _ = model.fit(X, Y)
    assert_equal(len(model.calibrated_classifiers_), 1)
    # Keyword partial updates leave omitted parameters untouched.
    _ = model.set_params(method="temperature", cv=CalibrationCV(4))
    assert_equal(model.method, "temperature")
    assert_equal(model.cv[Int], 4)
    assert_true(model.ensemble.isa[Bool]() and not model.ensemble[Bool])
    _ = model.set_params(ensemble=Variant[Bool, String]("auto"), cv=CalibrationCV(3))
    assert_equal(model.ensemble[String], "auto")
    _ = model.set_params(ensemble=Variant[Bool, String](False))
    _ = model.fit(X, Y)
    assert_equal(len(model.calibrated_classifiers_), 1)
    assert_true(model.score(X, Y) >= 0)
    # The reference sets probabilities in (1, 1 + 1e-5] to exactly one.
    var rounded = nm.zeros[DType.float64](nm.Shape(1, 3))
    rounded[nm.Item(0, 0)] = 1.0 + 1e-6
    rounded[nm.Item(0, 1)] = 1.0 + 1e-4
    rounded[nm.Item(0, 2)] = 0.25
    var clipped = _clip_rounding(rounded^)
    assert_equal(clipped.item(0, 0), 1.0)
    assert_equal(clipped.item(0, 1), 1.0 + 1e-4)
    assert_equal(clipped.item(0, 2), 0.25)
    var frozen = Centroid()
    frozen.fit(X, Y)
    frozen.frozen = True
    var frozen_model = CalibratedClassifierCV(frozen^, cv=3)
    _ = frozen_model.fit(X, Y)
    assert_equal(len(frozen_model.calibrated_classifiers_), 1)
    var changed = nm.zeros[DType.float64](nm.Shape(18, 1))
    with assert_raises():
        _ = model.predict(changed)
    model.method = "unsupported"
    with assert_raises():
        _ = model.fit(X, Y)
    assert_true(model.__sklearn_is_fitted__())
    model.method = "sigmoid"
    _ = model.fit(X, Y)
    var before = model.predict_proba(X)
    _ = model.fit(X, Y)
    var after = model.predict_proba(X)
    for i in range(18):
        for j in range(3):
            assert_equal(before.item(i, j), after.item(i, j))
    var weighted_score = nm.ones[DType.float64](nm.Shape(18))
    var predictions = model.predict(X)
    weighted_score[nm.Item(0)] = 4
    var correct = Float64(0)
    for i in range(18):
        if predictions.item(i) == Y.item(i):
            correct += weighted_score.item(i)
    assert_equal(model.score(X, Y, weighted_score.copy()), correct / 21)
    var invalid = array([0.0, Float64("nan"), 1.0])
    with assert_raises():
        sigmoid.fit(invalid, y)
    invalid[nm.Item(1)] = Float64("inf")
    with assert_raises():
        _ = isotonic.predict(invalid)
    with assert_raises():
        _ = calibration_curve(truth, invalid)
    var wrong_weights = array([1.0])
    with assert_raises():
        _ = model.fit(X, Y, wrong_weights.copy())
    weighted_score[nm.Item(0)] = -1
    with assert_raises():
        _ = model.fit(X, Y, weighted_score.copy())
    var saved_y = Y.copy()
    Y[nm.Item(0)] = 0.25
    with assert_raises():
        _ = model.fit(X, Y)
    Y = saved_y^
    model.n_jobs = 2
    with assert_raises():
        _ = model.fit(X, Y)
    model.n_jobs = None
    model.cv = 1
    with assert_raises():
        _ = model.fit(X, Y)
    model.cv = 7
    with assert_raises():
        _ = model.fit(X, Y)
    model.cv = 3
    model.ensemble = "invalid"
    with assert_raises():
        _ = model.fit(X, Y)
    model.ensemble = "auto"
    var backing = nm.zeros[DType.float64](nm.Shape(18, 4))
    for i in range(18):
        for j in range(2):
            backing[nm.Item(i, 2 * j)] = X.item(i, j)
    var strided = backing.view_with_layout(
        nm.Shape(18, 2), nm.NDArrayStrides(4, 2), 0
    )
    _ = model.fit(strided, Y)
    var strided_result = model.predict_proba(strided)
    _ = model.fit(X, Y)
    var contiguous_result = model.predict_proba(X)
    for i in range(18):
        for j in range(3):
            assert_equal(
                strided_result.item(i, j), contiguous_result.item(i, j)
            )
    var cube = calibration_curve(truth, probabilities, n_bins="cube_root")
    assert_equal(cube[0].size, curve[0].size)
    for i in range(curve[0].size):
        assert_equal(cube[0].item(i), curve[0].item(i))
    var arbitrary = array([2.0, 2.0, 3.0, 3.0])
    with assert_raises():
        _ = calibration_curve(arbitrary, probabilities)
    var explicit_curve = calibration_curve(
        arbitrary, probabilities, pos_label=Float64(3), n_bins=2
    )
    assert_equal(explicit_curve[0].item(0), curve[0].item(0))
    print("PASS: calibration contracts")
