"""Native contract tests for OneVsRestClassifier; values cross-checked by ovr_parity.py."""
from numojo import NDArray, Shape, Item
import numojo as nm
from std.math import nan
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)
from moxie.base import clone
from moxie.multiclass import OneVsRestClassifier, OneVsRestParams
from ovr_fixtures import (
    MarginClassifier,
    MarginProbaClassifier,
    ProbaClassifier,
    ShiftParams,
    ScoreRegressor,
    DecisionRegressor,
    Float32Regressor,
)


def matrix(rows: List[List[Float64]]) raises -> NDArray[DType.float64]:
    var result = nm.zeros[DType.float64](Shape(len(rows), len(rows[0])))
    for i in range(len(rows)):
        for j in range(len(rows[0])):
            result[Item(i, j)] = rows[i][j]
    return result^


def labels(values: List[Int]) raises -> NDArray[DType.int64]:
    var result = nm.zeros[DType.int64](Shape(len(values)))
    for i in range(len(values)):
        result[Item(i)] = Int64(values[i])
    return result^


def X4() raises -> NDArray[DType.float64]:
    return matrix([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0], [0.5, 2.5], [2.5, 0.0]])


def assert_labels(got: NDArray[DType.int64], expected: List[Int]) raises:
    assert_equal(got.ndim, 1)
    assert_equal(got.size, len(expected))
    for i in range(len(expected)):
        assert_equal(Int(got[Item(i)]), expected[i])


def test_multiclass_shapes_and_order() raises:
    var model = OneVsRestClassifier(MarginProbaClassifier())
    assert_false(model.__sklearn_is_fitted__())
    model.fit(X4(), labels([7, 3, 5, 5, 7, 3]))
    assert_true(model.__sklearn_is_fitted__())
    assert_labels(model.classes_, [3, 5, 7])
    assert_equal(model.n_classes_(), 3)
    assert_equal(len(model.estimators_), 3)
    assert_false(model.multilabel_())
    assert_equal(model.n_features_in_.value(), 2)
    var decision = model.decision_function(X4())
    assert_equal(decision.shape[0], 6)
    assert_equal(decision.shape[1], 3)
    var proba = model.predict_proba(X4())
    assert_equal(proba.shape[1], 3)
    for i in range(6):
        var total: Float64 = 0.0
        for k in range(3):
            total += proba[Item(i, k)]
        assert_almost_equal(total, 1.0, atol=1e-15)
    # Predictions are the argmax of the decision columns, mapped to classes_.
    var predicted = model.predict(X4())
    for i in range(6):
        var best = 0
        for k in range(1, 3):
            if decision[Item(i, k)] > decision[Item(i, best)]:
                best = k
        assert_equal(predicted[Item(i)], model.classes_[Item(best)])


def test_components_receive_binary_columns() raises:
    var model = OneVsRestClassifier(MarginClassifier(shift=0.25))
    model.fit(X4(), labels([2, 0, 1, 1, 2, 0]))
    for k in range(3):
        assert_false(model.estimators_[k].is_constant())
        ref component = model.estimators_[k].estimator()
        assert_equal(component.fits, 1)
        assert_equal(component.shift, 0.25)
    with assert_raises(contains="not a constant"):
        _ = model.estimators_[0].constant()
    # The wrapped prototype itself is never fitted.
    assert_equal(model.estimator.fits, 0)
    assert_false(model.estimator.__sklearn_is_fitted__())


def test_binary_uses_one_component() raises:
    var model = OneVsRestClassifier(MarginClassifier())
    var y = labels([4, 9, 4, 9, 4, 9])
    model.fit(X4(), y)
    assert_equal(len(model.estimators_), 1)
    assert_equal(model.n_classes_(), 2)
    var decision = model.decision_function(X4())
    assert_equal(decision.ndim, 1)
    var predicted = model.predict(X4())
    for i in range(6):
        var expected = 9 if decision[Item(i)] > 0.0 else 4
        assert_equal(Int(predicted[Item(i)]), expected)


def test_binary_probability_threshold() raises:
    var model = OneVsRestClassifier(ProbaClassifier())
    model.fit(X4(), labels([0, 1, 1, 0, 0, 1]))
    var proba = model.predict_proba(X4())
    assert_equal(proba.shape[1], 2)
    var predicted = model.predict(X4())
    for i in range(6):
        assert_almost_equal(proba[Item(i, 0)] + proba[Item(i, 1)], 1.0, atol=1e-15)
        assert_equal(Int(predicted[Item(i)]), 1 if proba[Item(i, 1)] > 0.5 else 0)


def test_single_class_constant_component() raises:
    var model = OneVsRestClassifier(MarginProbaClassifier())
    model.fit(X4(), labels([5, 5, 5, 5, 5, 5]))
    assert_equal(len(model.estimators_), 1)
    assert_true(model.estimators_[0].is_constant())
    assert_equal(model.estimators_[0].constant(), 0)
    with assert_raises(contains="wraps no estimator"):
        _ = model.estimators_[0].estimator()
    assert_equal(model.n_classes_(), 1)
    assert_labels(model.predict(X4()), [5, 5, 5, 5, 5, 5])
    var proba = model.predict_proba(X4())
    assert_equal(proba.shape[1], 2)
    assert_equal(proba[Item(0, 0)], 1.0)
    assert_equal(proba[Item(0, 1)], 0.0)
    var decision = model.decision_function(X4())
    assert_equal(decision.ndim, 1)
    assert_equal(decision[Item(0)], 0.0)
    with assert_raises(contains="features"):
        _ = model.predict(matrix([[1.0]]))


def test_ties_choose_first_class() raises:
    # Identical rows give identical component scores for every class.
    var X = matrix([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    var model = OneVsRestClassifier(MarginClassifier())
    model.fit(X, labels([0, 1, 2, 0, 1, 2]))
    var tied = matrix([[1.0, 1.0]])
    var decision = model.decision_function(tied)
    var predicted = model.predict(tied)
    var best = 0
    for k in range(1, 3):
        if decision[Item(0, k)] > decision[Item(0, best)]:
            best = k
    assert_equal(Int(predicted[Item(0)]), best)


def test_float_labels_and_column_target() raises:
    var model = OneVsRestClassifier[MarginClassifier, label=DType.float64](MarginClassifier())
    var y = nm.zeros[DType.float64](Shape(6, 1))
    var values: List[Float64] = [2.0, -1.0, 0.0, 0.0, 2.0, -1.0]
    for i in range(6):
        y[Item(i, 0)] = values[i]
    model.fit(X4(), y)
    assert_equal(model.classes_.size, 3)
    assert_equal(model.classes_[Item(0)], -1.0)
    var predicted = model.predict(X4())
    assert_equal(predicted.ndim, 1)
    assert_equal(model.score(X4(), y), model.score(X4(), y))


def test_invalid_targets_and_inputs() raises:
    var model = OneVsRestClassifier[MarginClassifier, label=DType.float64](MarginClassifier())
    var continuous = nm.zeros[DType.float64](Shape(6))
    for i in range(6):
        continuous[Item(i)] = Float64(i) + 0.5
    with assert_raises(contains="ValueError"):
        model.fit(X4(), continuous)
    # Integral floats beyond int64 range are continuous for sklearn as well.
    var huge = nm.zeros[DType.float64](Shape(6))
    for i in range(6):
        huge[Item(i)] = 2.5e300 if i % 2 else 1.0
    with assert_raises(contains="ValueError"):
        model.fit(X4(), huge)
    var missing = nm.zeros[DType.float64](Shape(6))
    missing[Item(2)] = nan[DType.float64]()
    with assert_raises(contains="ValueError"):
        model.fit(X4(), missing)
    assert_false(model.__sklearn_is_fitted__())
    var ints = OneVsRestClassifier(MarginClassifier())
    var multioutput = nm.zeros[DType.int64](Shape(6, 2))
    for i in range(6):
        multioutput[Item(i, 0)] = Int64(i % 3)
        multioutput[Item(i, 1)] = Int64(i % 2 + 4)
    with assert_raises(contains="Multioutput target data is not supported"):
        ints.fit(X4(), multioutput)
    with assert_raises(contains="inconsistent numbers of samples: [6, 3]"):
        ints.fit(X4(), labels([0, 1, 2]))
    assert_false(ints.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = ints.predict(X4())
    with assert_raises(contains="NotFittedError"):
        _ = ints.decision_function(X4())
    with assert_raises(contains="NotFittedError"):
        _ = ints.n_classes_()
    ints.fit(X4(), labels([0, 1, 2, 0, 1, 2]))
    with assert_raises(contains="features"):
        _ = ints.predict(matrix([[1.0, 2.0, 3.0]]))
    # Like sklearn, parameters are validated when fit starts.
    var noisy = OneVsRestClassifier(MarginClassifier(), verbose=-1)
    with assert_raises(contains="'verbose' parameter"):
        noisy.fit(X4(), labels([0, 1, 2, 0, 1, 2]))
    assert_false(noisy.__sklearn_is_fitted__())


def test_failed_fit_preserves_state() raises:
    var model = OneVsRestClassifier(MarginClassifier())
    model.fit(X4(), labels([0, 1, 2, 0, 1, 2]))
    # A component failure (rank-one X is rejected by the fixture) commits nothing.
    with assert_raises(contains="fixture expects X"):
        model.fit(nm.zeros[DType.float64](Shape(4)), labels([5, 6, 5, 6]))
    assert_equal(model.classes_.size, 3)
    assert_equal(model.n_features_in_.value(), 2)
    assert_equal(len(model.estimators_), 3)


def test_refit_replaces_state() raises:
    var model = OneVsRestClassifier(MarginClassifier())
    model.fit(X4(), labels([0, 1, 2, 0, 1, 2]))
    model.fit(X4(), labels([8, 9, 8, 9, 8, 9]))
    assert_labels(model.classes_, [8, 9])
    assert_equal(len(model.estimators_), 1)
    assert_equal(model.estimators_[0].estimator().fits, 1)


def test_clone_and_params() raises:
    var model = OneVsRestClassifier(MarginClassifier(shift=1.5), n_jobs=4, verbose=2)
    model.fit(X4(), labels([0, 1, 2, 0, 1, 2]))
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    assert_equal(len(copy.estimators_), 0)
    assert_equal(copy.estimator.shift, 1.5)
    assert_equal(copy.n_jobs.value(), 4)
    assert_equal(copy.verbose, 2)
    var params = model.get_params()
    assert_equal(params.estimator.shift, 1.5)
    copy.set_params(OneVsRestParams[ShiftParams](ShiftParams(-2.0), None, 0))
    assert_equal(copy.estimator.shift, -2.0)
    assert_false(Bool(copy.n_jobs))
    copy.set_params(verbose=3, n_jobs=Optional[Int](2))
    assert_equal(copy.verbose, 3)
    assert_equal(copy.n_jobs.value(), 2)
    copy.set_params(estimator=MarginClassifier(shift=9.0))
    assert_equal(copy.estimator.shift, 9.0)
    # The original fitted model is untouched.
    assert_equal(model.estimator.shift, 1.5)
    assert_true(model.__sklearn_is_fitted__())


def test_weighted_score() raises:
    var model = OneVsRestClassifier(MarginClassifier())
    var y = labels([0, 1, 2, 0, 1, 2])
    model.fit(X4(), y)
    var predicted = model.predict(X4())
    var weights = nm.zeros[DType.float64](Shape(6))
    var correct: Float64 = 0.0
    var total: Float64 = 0.0
    for i in range(6):
        weights[Item(i)] = Float64(i + 1)
        total += Float64(i + 1)
        if predicted[Item(i)] == y[Item(i)]:
            correct += Float64(i + 1)
    assert_almost_equal(model.score(X4(), y, weights.copy()), correct / total, atol=1e-15)
    with assert_raises(contains="sample_weight"):
        _ = model.score(X4(), y, nm.zeros[DType.float64](Shape(2)))
    with assert_raises(contains="Weights sum to zero"):
        _ = model.score(X4(), y, nm.zeros[DType.float64](Shape(6)))


def test_regressor_threshold_and_lifecycle() raises:
    var X = matrix([[0.0], [0.0], [0.0], [0.0]])
    var y = labels([-3, 7, -3, 7])
    var model = OneVsRestClassifier(ScoreRegressor(shift=0.5), n_jobs=2)
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(X)
    model.fit(X, y)
    assert_labels(model.predict(X), [-3, -3, -3, -3])
    assert_equal(model.score(X, y), 0.5)
    assert_false(model.estimator.__sklearn_is_fitted__())
    assert_true(model.estimators_[0].estimator().__sklearn_is_fitted__())
    var copied = clone(model)
    assert_false(copied.__sklearn_is_fitted__())
    assert_equal(copied.get_params().estimator.shift, 0.5)
    copied.set_params(OneVsRestParams[ShiftParams](ShiftParams(0.75), None, 0))
    copied.fit(X, y)
    assert_labels(copied.predict(X), [7, 7, 7, 7])
    assert_labels(model.predict(X), [-3, -3, -3, -3])
    with assert_raises(contains="fixture expects X"):
        model.fit(nm.zeros[DType.float64](Shape(4)), y)
    assert_labels(model.predict(X), [-3, -3, -3, -3])
    with assert_raises(contains="features"):
        _ = model.predict(X4())
    model.fit(X4(), labels([2, 0, 1, 2, 0, 1]))
    assert_equal(len(model.estimators_), 3)
    model.fit(X, labels([9, 9, 9, 9]))
    assert_equal(len(model.estimators_), 1)
    assert_true(model.estimators_[0].is_constant())
    assert_labels(model.predict(X), [9, 9, 9, 9])


def test_regressor_prediction_precedes_decision() raises:
    var X = matrix([[0.0], [0.0], [0.0], [0.0]])
    var y = labels([-3, 7, -3, 7])
    var model = OneVsRestClassifier(DecisionRegressor(shift=0.25))
    model.fit(X, y)
    assert_labels(model.predict(X), [-3, -3, -3, -3])
    var decision = model.decision_function(X)
    assert_equal(decision.ndim, 1)
    for i in range(4):
        assert_equal(decision[Item(i)], 10.25)
    var single = OneVsRestClassifier(Float32Regressor(shift=0.75))
    single.fit(X, y)
    assert_labels(single.predict(X), [7, 7, 7, 7])



def test_multilabel_predictions_probabilities_and_subset_score() raises:
    var X = matrix([[0.0], [1.0], [2.0], [3.0]])
    var y = matrix([[0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]]).astype[DType.int64]()
    var model = OneVsRestClassifier(MarginProbaClassifier())
    model.fit(X, y)
    assert_true(model.multilabel_())
    assert_labels(model.classes_, [0, 1, 2])
    assert_equal(model.n_classes_(), 3)
    assert_equal(len(model.estimators_), 3)
    assert_true(model.estimators_[1].is_constant())
    assert_equal(model.estimators_[1].constant(), 1)
    var predicted = model.predict(X)
    var decision = model.decision_function(X)
    var proba = model.predict_proba(X)
    assert_equal(predicted.ndim, 2)
    assert_equal(predicted.shape[0], 4)
    assert_equal(predicted.shape[1], 3)
    for i in range(4):
        assert_equal(proba[Item(i, 1)], 1.0)
        assert_true(proba[Item(i, 0)] + proba[Item(i, 1)] + proba[Item(i, 2)] > 1.0)
        for k in range(3):
            assert_equal(Int(predicted[Item(i, k)]), 1 if decision[Item(i, k)] > 0.0 else 0)
    # Deliberately make two rows wrong, with two errors in one of those rows.
    var truth = predicted.copy()
    truth[Item(0, 0)] = 1 - truth[Item(0, 0)]
    truth[Item(0, 1)] = 0
    truth[Item(2, 2)] = 1 - truth[Item(2, 2)]
    assert_equal(model.score(X, truth), 0.5)
    var weights = matrix([[1.0, 2.0, 3.0, 4.0]]).reshape(Shape(4))
    assert_almost_equal(model.score(X, truth, weights.copy()), 0.6, atol=1e-15)
    with assert_raises(contains="sample_weight"):
        _ = model.score(X, truth, nm.zeros[DType.float64](Shape(4, 1)))
    with assert_raises(contains="Weights sum to zero"):
        _ = model.score(X, truth, nm.zeros[DType.float64](Shape(4)))
    with assert_raises(contains="shape"):
        _ = model.score(X, nm.zeros[DType.int64](Shape(4, 2)))
    with assert_raises(contains="shape"):
        _ = model.score(X, nm.zeros[DType.int64](Shape(3, 3)))
    with assert_raises(contains="shape"):
        _ = model.score(X, labels([0, 1, 0, 1]))


def test_multilabel_constants_refits_clone_and_failures() raises:
    var X = X4()
    var y = nm.zeros[DType.int64](Shape(6, 3))
    for i in range(6):
        y[Item(i, 1)] = 1
        y[Item(i, 2)] = Int64(i % 2)
    var model = OneVsRestClassifier(MarginClassifier(shift=0.25), n_jobs=2)
    with assert_raises(contains="NotFittedError"):
        _ = model.multilabel_()
    model.fit(X, labels([2, 4, 2, 4, 2, 4]))
    model.fit(X, y)
    assert_true(model.multilabel_())
    assert_true(model.estimators_[0].is_constant())
    assert_equal(model.estimators_[0].constant(), 0)
    assert_equal(model.estimators_[2].estimator().fits, 1)
    assert_equal(model.estimator.fits, 0)
    var copied = clone(model)
    assert_false(copied.__sklearn_is_fitted__())
    assert_equal(len(copied.estimators_), 0)
    assert_equal(copied.estimator.shift, 0.25)
    assert_equal(copied.n_jobs.value(), 2)
    with assert_raises(contains="NotFittedError"):
        _ = copied.predict(X)
    var saved = model.predict(X)
    with assert_raises(contains="fixture expects X"):
        model.fit(nm.zeros[DType.float64](Shape(6)), y)
    assert_true(model.multilabel_())
    var again = model.predict(X)
    for i in range(6):
        for k in range(3):
            assert_equal(saved[Item(i, k)], again[Item(i, k)])
    model.set_params(estimator=MarginClassifier(shift=0.75))
    model.fit(X, y)
    assert_equal(model.estimators_[2].estimator().shift, 0.75)
    assert_equal(model.estimators_[2].estimator().fits, 1)
    model.fit(X, nm.zeros[DType.int64](Shape(6, 2)))
    assert_equal(len(model.estimators_), 2)
    assert_true(model.multilabel_())
    model.fit(X, labels([9, 9, 9, 9, 9, 9]))
    assert_false(model.multilabel_())
    assert_labels(model.predict(X), [9, 9, 9, 9, 9, 9])
    model.fit(X, y)
    with assert_raises(contains="features"):
        _ = model.predict(matrix([[1.0]]))
    with assert_raises(contains="inconsistent"):
        model.fit(matrix([[1.0, 2.0]]), y)
    with assert_raises(contains="ValueError"):
        model.fit(X, nm.zeros[DType.int64](Shape(0, 2)))
    with assert_raises(contains="ValueError"):
        model.fit(X, nm.zeros[DType.int64](Shape(6, 0)))
    var invalid = nm.zeros[DType.int64](Shape(6, 2))
    invalid[Item(0, 0)] = 2
    with assert_raises(contains="only 0 or 1"):
        model.fit(X, invalid)
    var floats = OneVsRestClassifier[MarginClassifier, label=DType.float32](MarginClassifier())
    floats.fit(X, y.astype[DType.float32]())
    assert_true(floats.multilabel_())
    assert_equal(floats.predict(X).shape[1], 3)


def test_multilabel_threshold_uses_first_component() raises:
    var X = matrix([[0.0], [0.0], [0.0], [0.0]])
    var y = matrix([[0, 0], [0, 1], [0, 0], [0, 1]]).astype[DType.int64]()
    var model = OneVsRestClassifier(MarginClassifier(shift=0.25))
    model.fit(X, y)
    var predicted = model.predict(X)
    for i in range(4):
        # First constant component makes sklearn use 0.5 even for margins.
        assert_equal(predicted[Item(i, 1)], 0)
    y[Item(0, 0)] = 1
    model.fit(X, y)
    predicted = model.predict(X)
    for i in range(4):
        assert_equal(predicted[Item(i, 1)], 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
