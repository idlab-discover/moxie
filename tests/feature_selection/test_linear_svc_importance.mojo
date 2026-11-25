"""Native adapter ownership, capability and label-slicing contracts."""
import numojo as nm
from moxie.base import DecisionFunction, ProbabilityPredictor, clone
from moxie.feature_selection import (
    DenseSupervisedCV,
    LinearSVCImportance,
    RFE,
    RFECV,
)
from moxie.svm import LinearSVC
from std.testing import assert_equal, assert_false, assert_raises, assert_true


def check_cv_labels[
    Label: Copyable & Comparable & Deinitable
](y: List[Label]) raises:
    comptime E = LinearSVCImportance[Label]
    var cv = DenseSupervisedCV[E]()
    var X = nm.zeros[DType.float64](nm.Shape(4, 2))
    assert_equal(cv.n_samples(X, y), 4)
    assert_equal(cv.cv_labels(y).value().kind, "binary")
    var rows = nm.zeros[DType.int64](nm.Shape(3))
    rows[nm.Item(0)] = 3
    var sliced = cv.target_rows(y, rows)
    assert_equal(len(sliced), 3)
    assert_true(sliced[0] == y[3])
    assert_true(sliced[1] == y[0])
    sliced[1] = y[3].copy()
    assert_true(sliced[1] != y[0])
    rows[nm.Item(0)] = -1
    with assert_raises(contains="out of bounds"):
        _ = cv.target_rows(y, rows)
    with assert_raises(contains="inconsistent sample counts"):
        _ = cv.n_samples(X, List[Label]())


def main() raises:
    var inferred = LinearSVCImportance(LinearSVC[String]())
    comptime assert type_of(inferred) == LinearSVCImportance[String]
    check_cv_labels[String](["a", "b", "a", "b"])
    check_cv_labels[Int]([1, 2, 1, 2])
    check_cv_labels[Int64](
        [9007199254740992, 9007199254740993, 9007199254740992, 9007199254740993]
    )
    check_cv_labels[Float32]([1, 2, 1, 2])
    check_cv_labels[Float64]([1, 2, 1, 2])
    var float_cv = DenseSupervisedCV[LinearSVCImportance[Float64]]()
    with assert_raises(contains="finite"):
        _ = float_cv.cv_labels([1, 2, Float64("nan"), 2])

    comptime E = LinearSVCImportance[String, DType.float32]
    comptime CV = DenseSupervisedCV[E, DType.float32]
    comptime assert conforms_to(RFE[E], DecisionFunction)
    comptime assert conforms_to(RFECV[CV], DecisionFunction)
    comptime assert not conforms_to(E, ProbabilityPredictor)
    comptime assert not conforms_to(RFE[E], ProbabilityPredictor)
    comptime assert not conforms_to(RFECV[CV], ProbabilityPredictor)

    var model = E(LinearSVC[String](dual=False, random_state=11))
    var X = nm.zeros[DType.float32](nm.Shape(4, 2))
    X[nm.Item(0, 0)] = -2
    X[nm.Item(1, 0)] = 2
    X[nm.Item(2, 0)] = -3
    X[nm.Item(3, 0)] = 3
    var y: List[String] = ["é", "z", "é", "z"]
    with assert_raises(contains="NotFittedError"):
        _ = model.classes_()
    with assert_raises(contains="NotFittedError"):
        _ = model.importances("auto")
    with assert_raises(contains="coef_"):
        _ = model.importances("feature_importances_")
    with assert_raises(contains="requires y"):
        model.fit_selection(X, None, None)
    var decision = model.fit(X, y).decision_function(X)
    assert_equal(decision.ndim, 1)
    assert_equal(decision.shape[0], 4)
    assert_equal(model.score(X, y), 1.0)
    assert_equal(model.classes_(), ["z", "é"])
    var classes = model.classes_()
    classes[0] = "changed"
    assert_equal(model.classes_(), ["z", "é"])
    var coefficients = model.importances("coef_")
    assert_equal(coefficients.ndim, 2)
    assert_equal(coefficients.shape[0], 1)
    assert_equal(coefficients.shape[1], 2)
    coefficients[nm.Item(0, 0)] = 99
    assert_true(model.importances("auto")[nm.Item(0, 0)] != 99)
    var snapshot = model.copy()
    snapshot.model.coef_[nm.Item(0, 0)] = 99
    assert_true(model.model.coef_[nm.Item(0, 0)] != 99)

    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.get_params().random_state, 11)
    var params = fresh.get_params()
    params.C = 2
    params.penalty = "l1"
    _ = fresh.set_params(params)
    assert_true(fresh.uses_l1_threshold())
    assert_false(model.uses_l1_threshold())
    var weight_backing = nm.ones[DType.float32](nm.Shape(8))
    var weights = weight_backing.view_with_layout(
        nm.Shape(4), nm.NDArrayStrides(-2), 7
    )
    var replacement: List[String] = ["b", "c", "b", "c"]
    assert_equal(
        fresh.fit(X, replacement, sample_weight=weights).score(
            X, replacement, weights.copy()
        ),
        1.0,
    )
    assert_equal(fresh.classes_(), ["b", "c"])
    with assert_raises(contains="inconsistent target length"):
        _ = model.fit(X, List[String]())
    assert_equal(model.classes_(), ["z", "é"])
    _ = model.fit(X, replacement)
    assert_equal(model.classes_(), ["b", "c"])
    print("PASS LinearSVCImportance contracts")
