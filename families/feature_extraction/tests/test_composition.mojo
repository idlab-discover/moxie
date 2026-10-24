# Composition of the dense dictionary vectorizer with Moxie meta-estimators.
# Expected values come from the pinned scikit-learn reference:
#   make_pipeline(DictVectorizer(sparse=False), RidgeClassifier()).fit(X, y)
# on the rows below (sklearn 4358542a1bd39af2e7e1cbace8b9fed067473c5b).
from std.testing import assert_equal, assert_raises, assert_true
from numojo import NDArray, Shape, zeros
from moxie.base import clone
from moxie.feature_extraction import DictVectorizer
from moxie.feature_extraction._dict_vectorizer import FeatureBatch, FeatureEntry, FeatureKey, FeatureRow, FeatureValue
from moxie.linear_model import RidgeClassifier
from moxie.pipeline import Pipeline


def number(key: String, value: Int) -> FeatureEntry:
    return FeatureEntry(FeatureKey(key), FeatureValue(Int64(value)))


def category(key: String, value: String) -> FeatureEntry:
    return FeatureEntry(FeatureKey(key), FeatureValue(value))


def row(var a: FeatureEntry) -> FeatureRow:
    var r = FeatureRow()
    r.append(a^)
    return r^


def row(var a: FeatureEntry, var b: FeatureEntry) -> FeatureRow:
    var r = row(a^)
    r.append(b^)
    return r^


def row(var a: FeatureEntry, var b: FeatureEntry, var c: FeatureEntry) -> FeatureRow:
    var r = row(a^, b^)
    r.append(c^)
    return r^


def main() raises:
    var X = FeatureBatch()
    X.append(row(number("a", 1), number("b", 2)))
    X.append(row(number("b", 1), number("c", 3)))
    X.append(row(number("a", 2), number("c", 1)))
    X.append(row(number("a", 1), number("b", 1), number("c", 1)))
    X.append(row(category("b", "x")))
    X.append(row(category("b", "y"), number("a", 3)))
    var y = zeros[DType.int64](Shape(6))
    var labels: List[Int] = [0, 1, 0, 1, 2, 2]
    for i in range(6):
        y.unsafe_set(i, Int64(labels[i]))
    var Q = FeatureBatch()
    Q.append(row(number("a", 1)))
    Q.append(row(number("c", 2), number("z", 5)))
    Q.append(row(category("b", "x")))

    var pipe = Pipeline(DictVectorizer(sparse=False), RidgeClassifier())
    _ = pipe.fit(X, y)
    var predicted = pipe.predict(Q)
    var expected_labels: List[Int] = [2, 1, 2]
    for i in range(3):
        assert_equal(Int(predicted.unsafe_get(i)), expected_labels[i])
    # Closed-form ridge on a 6x5 count matrix; 1e-12 absorbs solver rounding.
    var expected: List[Float64] = [
        -0.30232558139534893, -0.8139534883720928, 0.11627906976744201,
        -0.9572093023255814, 0.3767441860465117, -0.41953488372093,
        -0.7972093023255814, -0.8232558139534882, 0.6204651162790701,
    ]
    var decision = pipe.decision_function(Q)
    assert_equal(decision.shape[0], 3)
    assert_equal(decision.shape[1], 3)
    for i in range(9):
        assert_true(abs(decision.unsafe_get(i) - expected[i]) <= 1e-12)
    var names = pipe.transformer.get_feature_names_out()
    var expected_names: List[String] = ["a", "b", "b=x", "b=y", "c"]
    assert_equal(len(names), 5)
    for i in range(5):
        assert_equal(names[i], expected_names[i])

    # Nested parameter change then refit replaces the learned vocabulary.
    _ = pipe.transformer.set_params(separator=":")
    _ = pipe.fit(X, y)
    assert_equal(pipe.transformer.get_feature_names_out()[2], "b:x")
    assert_equal(pipe.transformer.get_feature_names_out()[3], "b:y")

    # Cloning the pipeline yields unfitted steps with the same configuration.
    var fresh = clone(pipe)
    assert_true(not fresh.transformer.__sklearn_is_fitted__())
    assert_equal(fresh.transformer.get_params().separator, ":")
    with assert_raises(contains="NotFittedError"):
        _ = fresh.predict(Q)
    print("PASS: DictVectorizer pipeline composition, nested set_params and clone")
