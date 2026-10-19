"""Isolated consumer using Moxie's production LDA through an explicit adapter."""
import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.sys import argv
from std.memory import ArcPointer
from moxie.base import clone, SupervisedEstimator
from moxie.calibration import (
    CalibrationEstimator, CalibratedClassifierCV, CalibrationSplit,
    FrozenCalibrationEstimator, calibration_curve, CalibrationDisplay,
)
from moxie.discriminant_analysis import LinearDiscriminantAnalysis
from moxie.frozen import FrozenEstimator

comptime Array = nm.NDArray[DType.float64]


struct LDAAdapter(CalibrationEstimator):
    """Test adapter: LDA supplies decision responses but has no weighted fit."""
    comptime Params = LinearDiscriminantAnalysis[DType.float64, Int].Params
    var model: ArcPointer[LinearDiscriminantAnalysis[DType.float64, Int]]

    def __init__(out self) raises:
        self.model = ArcPointer(LinearDiscriminantAnalysis[DType.float64, Int]())

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.model[].get_params(deep)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        var model = self.model[].__sklearn_clone__()
        _ = model.set_params(params)
        self.model = ArcPointer(model^)
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.model = ArcPointer(self.model[].__sklearn_clone__())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self.model[].__sklearn_is_fitted__()

    def fit(mut self, X: Array, y: Array, sample_weight: Optional[Array] = None) raises:
        if sample_weight:
            raise Error("ValueError: LDA does not support weighted fitting")
        var labels = List[Int]()
        for i in range(y.size):
            labels.append(Int(y.item(i)))
        var model = self.model[].__sklearn_clone__()
        _ = model.fit(X, labels)
        self.model = ArcPointer(model^)

    def class_labels(self) raises -> Array:
        var result = nm.zeros[DType.float64](nm.Shape(len(self.model[].classes_)))
        for i in range(len(self.model[].classes_)):
            result[nm.Item(i)] = Float64(self.model[].classes_[i])
        return result^

    def uses_probabilities(self) -> Bool:
        return False

    def response(self, X: Array) raises -> Array:
        var response = self.model[].decision_function(X)
        if response.ndim == 1:
            return response.reshape(nm.Shape(response.size, 1))
        return response^


def generic_fit[E: SupervisedEstimator](mut model: E, X: E.Input, y: E.Target) raises:
    _ = model.fit(X, y)


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(30, 2))
    var y = nm.zeros[DType.float64](nm.Shape(30))
    for i in range(30):
        X[nm.Item(i, 0)] = Float64(i % 2) + Float64(i % 7) / 4
        X[nm.Item(i, 1)] = Float64(i % 11) / 3
        y[nm.Item(i)] = -3.0 if i % 2 == 0 else 7.0
    var unfitted = FrozenCalibrationEstimator(FrozenEstimator(LDAAdapter()))
    with assert_raises():
        unfitted.fit(X, y)
    var splits = List[CalibrationSplit]()
    for fold in range(3):
        var train = List[Int]()
        var test = List[Int]()
        for i in range(30):
            if i % 3 == fold:
                test.append(i)
            else:
                train.append(i)
        splits.append(CalibrationSplit(train^, test^))
    var methods: List[String] = ["sigmoid", "isotonic", "temperature"]
    for method in methods:
        var model = CalibratedClassifierCV(LDAAdapter(), method=method.copy(), cv=splits.copy(), ensemble=False)
        generic_fit(model, X, y)
        assert_equal(len(model.calibrated_classifiers_), 1)
        var params = model.get_params()
        var probabilities = model.set_params(params).fit(X, y).predict_proba(X)
        assert_equal(probabilities.shape[1], 2)
        assert_equal(model.classes_.item(0), -3)
        assert_equal(model.classes_.item(1), 7)
        for i in range(30):
            assert_true(abs(probabilities.item(i, 0) + probabilities.item(i, 1) - 1) < 1e-12)
        var before = model.predict_proba(X)
        with assert_raises():
            _ = model.fit(X, y, nm.ones[DType.float64](nm.Shape(30)))
        var after = model.predict_proba(X)
        assert_equal(before.item(0, 1), after.item(0, 1))
        assert_false(clone(model).__sklearn_is_fitted__())
        var base = LDAAdapter()
        base.fit(X, y)
        var frozen = FrozenCalibrationEstimator(FrozenEstimator(base^))
        var copied = clone(frozen)
        assert_true(copied.__sklearn_is_fitted__())
        var frozen_model = CalibratedClassifierCV(frozen^, method=method.copy(), cv=splits.copy())
        _ = frozen_model.fit(X, y, nm.ones[DType.float64](nm.Shape(30)))
        assert_equal(len(frozen_model.calibrated_classifiers_), 1)
        var original = copied.response(X)
        var changed_y = y.copy()
        for i in range(30):
            changed_y[nm.Item(i)] = -3
        _ = copied.fit(X, changed_y)
        assert_equal(copied.response(X).item(0, 0), original.item(0, 0))
        var display = CalibrationDisplay.from_estimator(model, X, y)
        var args = argv()
        var path = args[1] if len(args) > 1 else "builds/calibration-consumer.svg"
        display.figure_.value().savefig(path)
        var p = nm.zeros[DType.float64](nm.Shape(30))
        for i in range(30):
            p[nm.Item(i)] = probabilities.item(i, 1)
        var curve = calibration_curve(y, p, pos_label=7.0, n_bins="cube_root")
        assert_true(curve[0].size > 0)
    print("PASS: production LDA, fluent contracts, frozen ownership, splits, curves and SVG")
