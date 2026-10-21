"""Wrapper behavior with production Ridge, RidgeClassifier and RandomForest components."""
import numojo as nm
from numojo import NDArray, Shape
from std.math import nan
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.ensemble import RandomForestClassifier
from moxie.linear_model import Ridge, RidgeClassifier
from moxie.multioutput import MultiOutputRegressor, MultiOutputClassifier, RegressorChain, ClassifierChain


comptime ROWS = 12
comptime FEATURES = 3


def features[dtype: DType]() raises -> NDArray[dtype]:
    var X = nm.zeros[dtype](Shape(ROWS, FEATURES))
    for i in range(ROWS):
        for j in range(FEATURES):
            X.unsafe_set(i * FEATURES + j, Scalar[dtype](Float64((i * (j + 3) + 2 * j) % 7) - 3.0 + 0.1 * Float64(i)))
    return X^


def targets[dtype: DType]() raises -> NDArray[dtype]:
    var X = features[DType.float64]()
    var Y = nm.zeros[dtype](Shape(ROWS, 3))
    for i in range(ROWS):
        for k in range(3):
            var value = 0.5 * Float64(i % 3)
            for j in range(FEATURES):
                value += X.unsafe_get(i * FEATURES + j) * Float64(k + 1 - j)
            Y.unsafe_set(i * 3 + k, Scalar[dtype](value))
    return Y^


def labels[label: DType](multiclass: Bool) raises -> NDArray[label]:
    var X = features[DType.float64]()
    var L = nm.zeros[label](Shape(ROWS, 3))
    for i in range(ROWS):
        L.unsafe_set(i * 3, Scalar[label](1 if X.unsafe_get(i * FEATURES) + X.unsafe_get(i * FEATURES + 1) > 0 else 0))
        L.unsafe_set(i * 3 + 1, Scalar[label](i % 2))
        if multiclass:
            L.unsafe_set(i * 3 + 2, Scalar[label](i % 3))
        else:
            L.unsafe_set(i * 3 + 2, Scalar[label](1 if X.unsafe_get(i * FEATURES + 2) > X.unsafe_get(i * FEATURES) else 0))
    return L^


def weights() raises -> NDArray[DType.float64]:
    var w = nm.zeros[DType.float64](Shape(ROWS))
    for i in range(ROWS):
        w.unsafe_set(i, Float64(i % 4 + 1))
    return w^


def parity[dtype: DType](name: String, values: NDArray[dtype]) raises:
    var line = "PARITY " + name
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            line += " " + String(Float64(values.unsafe_get(i * Int(values.strides[0]) + j * Int(values.strides[1]))))
    print(line)


def assert_same[dtype: DType](a: NDArray[dtype], b: NDArray[dtype]) raises:
    assert_equal(a.shape[0], b.shape[0])
    assert_equal(a.shape[1], b.shape[1])
    for i in range(a.shape[0] * a.shape[1]):
        assert_equal(a.unsafe_get(i), b.unsafe_get(i))


def main() raises:
    var X = features[DType.float64]()
    var Y = targets[DType.float64]()
    var w = weights()

    var regressor = MultiOutputRegressor(Ridge(0.5))
    with assert_raises(contains="NotFittedError"):
        _ = regressor.predict(X)
    var first = regressor.fit(X, Y).predict(X)
    parity("ComponentsRidge", first)
    # Repeat fits are deterministic and replace earlier state.
    assert_same(first, regressor.fit(X, Y).predict(X))
    print("PARITY ComponentsRidgeScore", regressor.score(X, Y))
    parity("ComponentsRidgeWeighted", regressor.fit(X, Y, sample_weight=w).predict(X))
    print("PARITY ComponentsRidgeWeightedScore", regressor.score(X, Y, w.copy()))
    var two = nm.zeros[DType.float64](Shape(ROWS, 2))
    for i in range(ROWS * 2):
        two.unsafe_set(i, Y.unsafe_get((i // 2) * 3 + i % 2))
    _ = regressor.fit(X, two)
    assert_equal(len(regressor.estimators_), 2)
    assert_equal(regressor.predict(X).shape[1], 2)

    # Nested parameter edits reach the component and survive clone.
    var params = regressor.get_params()
    params.estimator.alpha = Float64(10)
    _ = regressor.set_params(params)
    var edited = clone(regressor)
    assert_false(edited.__sklearn_is_fitted__())
    assert_equal(edited.get_params().estimator.alpha[Float64], Float64(10))
    var direct = MultiOutputRegressor(Ridge(10.0))
    assert_same(regressor.fit(X, Y).predict(X), direct.fit(X, Y).predict(X))
    parity("ComponentsRidgeAlpha10", direct.predict(X))

    # Float64 wrapper weights are narrowed for Float32 components.
    var X32 = features[DType.float32]()
    var Y32 = targets[DType.float32]()
    var narrow = MultiOutputRegressor[Ridge[DType.float32], DType.float32](Ridge[DType.float32](0.5))
    parity("ComponentsRidge32Weighted", narrow.fit(X32, Y32, sample_weight=w).predict(X32))
    var chain32 = RegressorChain[Ridge[DType.float32], DType.float32](Ridge[DType.float32](0.5), order=List[Int]([2, 0, 1]))
    parity("ComponentsRegressorChain32Weighted", chain32.fit(X32, Y32, sample_weight=w).predict(X32))

    var chain = RegressorChain(Ridge(0.5), order=List[Int]([2, 0, 1]), verbose=True)
    parity("ComponentsRegressorChain", chain.fit(X, Y).predict(X))
    print("PARITY ComponentsRegressorChainScore", chain.score(X, Y))
    var wrong = nm.zeros[DType.float64](Shape(2, FEATURES + 1))
    with assert_raises(contains="feature count"):
        _ = chain.predict(wrong)
    var missing = X.copy()
    missing.unsafe_set(4, nan[DType.float64]())
    var rejecting = RegressorChain(Ridge(0.5))
    with assert_raises(contains="NaN"):
        _ = rejecting.fit(missing, Y)

    var binary = labels[DType.int64](False)
    var decision_chain = ClassifierChain(RidgeClassifier(), order=List[Int]([1, 0, 2]), chain_method="decision_function")
    with assert_raises(contains="NotFittedError"):
        _ = decision_chain.decision_function(X)
    _ = decision_chain.fit(X, binary)
    assert_equal(decision_chain.chain_method_, "decision_function")
    parity("ComponentsDecisionChainLabels", decision_chain.predict(X))
    parity("ComponentsDecisionChain", decision_chain.decision_function(X))
    print("PARITY ComponentsDecisionChainScore", decision_chain.score(X, binary, w.copy()))

    var multiclass = labels[DType.int64](True)
    var label_chain = ClassifierChain(RidgeClassifier(), order=List[Int]([2, 0, 1]))
    _ = label_chain.fit(X, multiclass)
    # classes_ follows chain order: the first link models output 2 (three classes).
    assert_equal(label_chain.classes_[0].shape[0], 3)
    assert_equal(label_chain.classes_[1].shape[0], 2)
    parity("ComponentsLabelChainLabels", label_chain.predict(X))
    var independent = MultiOutputClassifier(RidgeClassifier())
    _ = independent.fit(X, multiclass)
    assert_equal(independent.classes_[2].shape[0], 3)
    parity("ComponentsIndependentLabels", independent.predict(X))
    print("PARITY ComponentsIndependentScore", independent.score(X, multiclass))
    var random_chain = ClassifierChain(RidgeClassifier(), random_order=True, random_state=3)
    _ = random_chain.fit(X, multiclass)
    print("PARITY ComponentsRandomOrder", random_chain.order_[0], random_chain.order_[1], random_chain.order_[2])
    parity("ComponentsRandomLabels", random_chain.predict(X))

    # Float64 forest probabilities are converted to the Float32 feature dtype.
    var forest = MultiOutputClassifier[RandomForestClassifier[], DType.float32, DType.float64](RandomForestClassifier(n_estimators=5, random_state=0))
    var forest_labels = labels[DType.float64](True)
    _ = forest.fit(X32, forest_labels)
    var probabilities = forest.predict_proba(X32)
    assert_equal(len(probabilities), 3)
    assert_equal(probabilities[2].shape[1], 3)
    for j in range(3):
        for i in range(ROWS):
            var total = Float64(0)
            for c in range(probabilities[j].shape[1]):
                var p = Float64(probabilities[j].unsafe_get(i * probabilities[j].shape[1] + c))
                assert_true(p >= 0 and p <= 1)
                total += p
            assert_true(abs(total - 1) < 1e-6)
    var forest_chain = ClassifierChain[RandomForestClassifier[], DType.float32, DType.float64](RandomForestClassifier(n_estimators=5, random_state=0), chain_method="predict_proba")
    var forest_binary = labels[DType.float64](False)
    var chained = forest_chain.fit(X32, forest_binary).predict_proba(X32)
    # Probabilities stay Float64 (sklearn's dtype) although features are Float32.
    parity("ComponentsForestProba2", probabilities[2])
    parity("ComponentsForestChainProba", chained)
    for i in range(ROWS * 3):
        assert_true(chained.unsafe_get(i) >= 0 and chained.unsafe_get(i) <= 1)
    print("PASS: production Ridge, RidgeClassifier and RandomForest components")
