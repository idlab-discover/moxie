from std.testing import assert_equal, assert_true, assert_raises
import numojo as nm
from moxie.calibration import CalibratedClassifierCV, CalibrationSplit
from moxie.base import clone
from moxie._calibration_algorithms import Array
from calibration_centroid import Centroid


def data() raises -> Tuple[Array, Array]:
    var X = nm.zeros[DType.float64](nm.Shape(12, 2))
    var y = nm.zeros[DType.float64](nm.Shape(12))
    for i in range(12):
        X[nm.Item(i, 0)] = Float64(i)
        X[nm.Item(i, 1)] = Float64(i % 5)
        y[nm.Item(i)] = Float64(i // 3)
    return (X^, y^)


def folds() -> List[CalibrationSplit]:
    var out = List[CalibrationSplit]()
    for fold in range(3):
        var train = List[Int]()
        var test = List[Int]()
        for i in range(12):
            if i // 4 == fold:
                test.append(i)
            else:
                train.append(i)
        out.append(CalibrationSplit(train^, test^))
    return out^


def missing_classes() raises:
    var inputs = data()
    var splits = folds()
    var model = CalibratedClassifierCV(Centroid(True), cv=splits.copy())
    _ = model.fit(inputs[0], inputs[1])
    assert_equal(len(model.calibrated_classifiers_), 3)
    var first = model.calibrated_classifiers_[0].predict_proba(inputs[0])
    var last = model.calibrated_classifiers_[2].predict_proba(inputs[0])
    for i in range(12):
        assert_equal(first.item(i, 0), 0)
        assert_equal(last.item(i, 3), 0)
    var params = model.get_params()
    params.cv[List[CalibrationSplit]][0].test[0] = 11
    assert_equal(model.cv[List[CalibrationSplit]][0].test[0], 0)
    var copied = clone(model)
    assert_true(not copied.__sklearn_is_fitted__())
    assert_equal(len(copied.cv[List[CalibrationSplit]]), 3)
    model.ensemble = False
    _ = model.fit(inputs[0], inputs[1])
    assert_equal(len(model.calibrated_classifiers_), 1)
    var p = model.predict_proba(inputs[0])
    for i in range(12):
        var total = Float64(0)
        for j in range(4):
            total += p.item(i, j)
        assert_true(abs(total - 1) < 1e-14)


def invalid_splits() raises:
    var inputs = data()
    var splits = folds()
    var model = CalibratedClassifierCV(
        Centroid(True), cv=splits.copy(), ensemble=False
    )
    _ = model.fit(inputs[0], inputs[1])
    var before = model.predict_proba(inputs[0])
    splits[0].test[0] = 4
    model.cv = splits.copy()
    with assert_raises():
        _ = model.fit(inputs[0], inputs[1])
    var after = model.predict_proba(inputs[0])
    assert_equal(before.item(0, 0), after.item(0, 0))
    model.ensemble = True
    _ = model.fit(inputs[0], inputs[1])
    splits[0].test[0] = 12
    model.cv = splits.copy()
    with assert_raises():
        _ = model.fit(inputs[0], inputs[1])
    splits[0].test[0] = -1
    model.cv = splits.copy()
    with assert_raises():
        _ = model.fit(inputs[0], inputs[1])
    model.cv = List[CalibrationSplit]()
    with assert_raises():
        _ = model.fit(inputs[0], inputs[1])
    splits = folds()
    splits[0].train = List[Int]()
    model.cv = splits.copy()
    with assert_raises():
        _ = model.fit(inputs[0], inputs[1])


def main() raises:
    missing_classes()
    invalid_splits()
    print("PASS: custom split and missing-class contracts")
