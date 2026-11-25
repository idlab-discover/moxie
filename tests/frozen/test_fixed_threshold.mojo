"""Native contracts and deterministic sklearn input for fixed thresholds."""

from moxie.base import ClassifierMixin, clone
from moxie.discriminant_analysis import LinearDiscriminantAnalysis
from moxie.frozen import FrozenEstimator
from moxie.linear_model import LogisticRegression
from moxie.model_selection import FixedThresholdClassifier
from numojo import Item, NDArray, NDArrayStrides, Shape
from std.testing import assert_equal, assert_false, assert_raises, assert_true


def fixture[dtype: DType](values: List[Scalar[dtype]], matrix: Bool = False) raises -> NDArray[dtype]:
    var size = len(values)
    var result = NDArray[dtype](Shape(size, 1) if matrix else Shape(size))
    for i in range(size):
        result.unsafe_set(i, values[i])
    return result^


def as_list(x: NDArray[DType.int64]) raises -> List[Int]:
    assert_equal(x.ndim, 1)
    var result = List[Int](capacity=x.size)
    for i in range(x.size):
        result.append(Int(x.item(i)))
    return result^


def logistic_cases[dtype: DType]() raises:
    var X = fixture[dtype]([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0], matrix=True)
    var y = fixture[DType.int64]([-3, -3, -3, -3, 7, 7, 7, 7])
    var flipped = fixture[DType.int64]([7, 7, 7, 7, -3, -3, -3, -3])
    var test = fixture[dtype]([-3.0, -1.0, 0.0, 1.0, 3.0], matrix=True)
    var truth = fixture[DType.int64]([-3, -3, -3, 7, 7])
    var weights = fixture[dtype]([1.0, 1.0, 5.0, 2.0, 1.0])

    var base = LogisticRegression[dtype](fit_intercept=False)
    _ = base.fit(X, y)
    var frozen = FrozenEstimator(base^)
    var model = FixedThresholdClassifier(frozen.copy())
    comptime assert conforms_to(type_of(model), ClassifierMixin)
    assert_true(model.__sklearn_is_fitted__())
    assert_false(Bool(model.estimator_))
    assert_equal(model.get_n_features_in(), 1)
    assert_equal(as_list(model.classes_()), [-3, 7])
    var predicted = as_list(model.predict(test))
    assert_equal(predicted[2], 7)
    print("default", String(dtype), predicted)
    var backing = NDArray[dtype](Shape(10))
    var strided = backing.view_with_layout(Shape(5), NDArrayStrides(-2), 9)
    for row in range(5):
        strided[Item(row)] = weights[Item(row)]
    print("strided-score", String(dtype), model.score(test, truth, strided^))
    print("weighted-score", String(dtype), model.score(test, truth, weights^))

    _ = model.set_params(threshold=0.9)
    print("probability", String(dtype), as_list(model.predict(test)))
    _ = model.set_params(pos_label=Int64(-3), response_method=String("predict_proba"))
    print("first-probability", String(dtype), as_list(model.predict(test)))
    _ = model.set_params(threshold=1.2, response_method=String("decision_function"))
    print("first-decision", String(dtype), as_list(model.predict(test)))

    var params = model.get_params()
    params.threshold = None
    _ = model.set_params(params)
    var expected = as_list(model.predict(test))
    assert_equal(expected[2], -3)
    print("first-decision-auto", String(dtype), expected)
    _ = model.fit(X, flipped)
    assert_true(Bool(model.estimator_))
    var refitted = as_list(model.predict(test))
    var cloned = as_list(clone(model).predict(test))
    assert_equal(refitted, expected)
    assert_equal(cloned, expected)
    print("frozen-refit", String(dtype), refitted)
    print("frozen-clone", String(dtype), cloned)

    var probabilities = model.predict_proba(test)
    var log_probabilities = model.predict_log_proba(test)
    var decision = model.decision_function(test)
    var expected_probabilities = frozen.predict_proba(test)
    var expected_log = frozen.predict_log_proba(test)
    var expected_decision = frozen.decision_function(test)
    for i in range(probabilities.size):
        assert_equal(probabilities.unsafe_get(i), expected_probabilities.unsafe_get(i))
        assert_equal(log_probabilities.unsafe_get(i), expected_log.unsafe_get(i))
    for i in range(decision.size):
        assert_equal(decision.unsafe_get(i), expected_decision.unsafe_get(i))

    _ = model.set_params(pos_label=Int64(9))
    with assert_raises():
        _ = model.predict(test)
    _ = model.set_params(response_method=String("unknown"))
    with assert_raises():
        _ = model.fit(X, y)


def lifecycle() raises:
    var X = fixture[DType.float64]([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], matrix=True)
    var y = fixture[DType.int64]([-3, -3, -3, 7, 7, 7])
    var flipped = fixture[DType.int64]([7, 7, 7, -3, -3, -3])
    var multiclass = fixture[DType.int64]([-3, -3, 0, 0, 7, 7])
    var weights = fixture[DType.float64]([2.0, 1.0, 3.0, 2.0, 1.0, 4.0])
    var model = FixedThresholdClassifier(LogisticRegression(fit_intercept=False), threshold=0.78)
    with assert_raises():
        _ = model.predict(X)
    with assert_raises():
        _ = model.classes_()
    _ = model.fit(X, y, sample_weight=weights)
    assert_false(model.estimator.__sklearn_is_fitted__())
    assert_false(clone(model).__sklearn_is_fitted__())
    var expected = as_list(model.predict(X))
    print("weighted-fit", "float64", expected)
    with assert_raises():
        _ = model.fit(X, multiclass)
    assert_equal(as_list(model.predict(X)), expected)
    with assert_raises():
        _ = model.fit(X, fixture[DType.int64]([7]))
    _ = model.fit(X, flipped, sample_weight=weights)
    print("repeat-fit", "float64", as_list(model.predict(X)))

    var lda = LinearDiscriminantAnalysis()
    var listed: List[Int] = [-3, -3, -3, 7, 7, 7]
    _ = lda.fit(X, listed)
    var frozen_lda = FixedThresholdClassifier(FrozenEstimator(lda^), threshold=0.9)
    assert_equal(frozen_lda.classes_(), [-3, 7])
    assert_equal(frozen_lda.get_n_features_in(), 1)
    print("lda-threshold", "float64", frozen_lda.predict(X))
    print("lda-score", "float64", frozen_lda.score(X, listed))

    var multiclass_base = LogisticRegression()
    _ = multiclass_base.fit(X, multiclass)
    var invalid = FixedThresholdClassifier(multiclass_base^)
    assert_equal(as_list(invalid.classes_()), [-3, 0, 7])
    with assert_raises():
        _ = invalid.predict(X)
    assert_equal(invalid.predict_proba(X).shape[1], 3)


def main() raises:
    logistic_cases[DType.float32]()
    logistic_cases[DType.float64]()
    lifecycle()
