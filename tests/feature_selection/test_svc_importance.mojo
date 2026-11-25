"""Lifecycle and capability contracts for native linear-kernel SVC selectors."""
import numojo as nm
from moxie.base import DecisionFunction, LogProbabilityPredictor, clone
from moxie.feature_selection import (
    DenseSupervisedCV,
    RFE,
    RFECV,
    SVCImportance,
    SelectFromModel,
)
from moxie.svm import SVC
from std.testing import assert_equal, assert_false, assert_raises, assert_true


def main() raises:
    var inferred = SVCImportance(SVC[String](kernel="linear"))
    comptime assert type_of(inferred) == SVCImportance[String]
    assert_equal(inferred.get_params().kernel, "linear")
    comptime E = SVCImportance[String, DType.float32]
    comptime CV = DenseSupervisedCV[E, DType.float32]
    comptime assert conforms_to(RFE[E], DecisionFunction)
    comptime assert conforms_to(RFECV[CV], DecisionFunction)
    comptime assert conforms_to(RFE[E], LogProbabilityPredictor)
    comptime assert conforms_to(RFECV[CV], LogProbabilityPredictor)

    var X = nm.array[DType.float32]([-2, 0, 2, 0, -3, 1, 3, 1], [4, 2])
    var y: List[String] = ["é", "z", "é", "z"]
    var model = E(SVC[String](kernel="linear", random_state=11))
    with assert_raises(contains="NotFittedError"):
        _ = model.classes_()
    with assert_raises(contains="NotFittedError"):
        _ = model.importances("auto")
    with assert_raises(contains="exposes coef_"):
        _ = model.importances("feature_importances_")
    with assert_raises(contains="requires y"):
        model.fit_selection(X, None, None)

    var invalid = SelectFromModel(E(SVC[String]()))
    with assert_raises(contains="requires kernel"):
        _ = invalid.fit(X, y.copy())
    with assert_raises(contains="NotFittedError"):
        _ = invalid.transform(X)

    var decision = model.fit(X, y).decision_function(X)
    assert_equal(decision.ndim, 1)
    assert_equal(decision.shape[0], 4)
    assert_equal(model.score(X, y), 1.0)
    with assert_raises(contains="probability=True"):
        _ = model.predict_proba(X)
    var classes = model.classes_()
    assert_equal(classes, ["z", "é"])
    classes[0] = "changed"
    assert_equal(model.classes_(), ["z", "é"])

    var coefficients = model.importances("coef_")
    assert_equal(coefficients.ndim, 2)
    assert_equal(coefficients.shape[0], 1)
    assert_equal(coefficients.shape[1], 2)
    coefficients[nm.Item(0, 0)] = 99
    assert_true(model.importances("auto")[nm.Item(0, 0)] != 99)
    var snapshot = model.copy()
    snapshot.model.support_vectors_[nm.Item(0, 0)] = 99
    assert_true(model.model.support_vectors_[nm.Item(0, 0)] != 99)
    var prefit = SelectFromModel(model.copy(), prefit=True)
    assert_equal(prefit.transform(X).shape[1], 1)
    _ = prefit.fit(X, y.copy())
    var fitted = prefit.estimator_()
    fitted.model.classes_[0] = "changed"
    assert_equal(prefit.estimator_().classes_(), ["z", "é"])

    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.get_params().random_state, 11)
    var params = fresh.get_params()
    params.C = 2
    _ = fresh.set_params(params)
    assert_equal(fresh.get_params().C, 2.0)
    assert_equal(model.get_params().C, 1.0)
    var weights = nm.array[DType.float32]([0.5, 0.75, 1.25, 1.5], [4])
    var replacement: List[String] = ["b", "c", "b", "c"]
    assert_equal(
        fresh.fit(X, replacement, sample_weight=weights).score(
            X, replacement, weights.copy()
        ),
        1.0,
    )
    with assert_raises(contains="inconsistent target length"):
        _ = model.fit(X, List[String]())
    assert_equal(model.classes_(), ["z", "é"])
    _ = model.fit(X, replacement)
    assert_equal(model.classes_(), ["b", "c"])
    assert_equal(snapshot.classes_(), ["z", "é"])
    print("PASS SVCImportance contracts")
