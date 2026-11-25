"""Wrapper behavior with production linear models and RandomForestClassifier."""
import numojo as nm
from numojo import NDArray, Shape
from std.math import nan
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.ensemble import RandomForestClassifier
from moxie.linear_model import LogisticRegression, Ridge, RidgeClassifier
from moxie.model_selection import IndexSplit, KFold, StratifiedKFold
from moxie.multioutput import ChainFolds, ChainSplitter, MultiOutputRegressor, MultiOutputClassifier, RegressorChain, ClassifierChain


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


def logistic_components() raises:
    var X = features[DType.float64]()
    var binary = labels[DType.int64](False)
    var multiclass = labels[DType.int64](True)
    var base = LogisticRegression(tol=1e-10, max_iter=1000)
    var independent = MultiOutputClassifier(clone(base))
    _ = independent.fit(X, multiclass)
    var predictions = independent.predict(X)
    var probabilities = independent.predict_proba(X)
    comptime assert type_of(predictions) == NDArray[DType.int64]
    comptime assert type_of(probabilities) == List[NDArray[DType.float64]]
    assert_equal(len(probabilities), 3)
    assert_equal(probabilities[2].shape, Shape(ROWS, 3))
    for i in range(3):
        assert_equal(independent.classes_[2].item(i), Int64(i))
    parity("ComponentsLogisticIndependentLabels", predictions)
    parity("ComponentsLogisticIndependentProba", probabilities[2])

    var chain = ClassifierChain(
        clone(base), order=List[Int]([2, 0, 1]), chain_method="predict_proba",
    )
    _ = chain.fit(X, binary)
    parity("ComponentsLogisticChainLabels", chain.predict(X))
    parity("ComponentsLogisticChainProba", chain.predict_proba(X))
    _ = chain.fit(X, binary, cv=3)
    parity("ComponentsLogisticChainCVLabels", chain.predict(X))
    parity("ComponentsLogisticChainCVProba", chain.predict_proba(X))


@fieldwise_init
struct PerLinkKFold[label: DType](ChainSplitter):
    comptime Input = NDArray[DType.float64]
    comptime Target = NDArray[Self.label]
    comptime Splits = KFold.Splits

    var seed: Int
    var threshold: Float64

    def split(self, X: Self.Input, y: Self.Target) raises -> Self.Splits:
        if self.seed < 0:
            raise Error("ValueError: invalid splitter seed")
        if y.ndim != 1 or X.shape[1] >= FEATURES + 2:
            raise Error("ValueError: expected a target column before the final link")
        var folds = 2 if Float64(y.unsafe_get(0)) > self.threshold else 3
        return KFold(folds, shuffle=True, random_state=self.seed + X.shape[1]).split(X.shape[0])


def custom_chain_cv() raises:
    var X = features[DType.float64]()
    var Y = targets[DType.float64]()
    var order = List[Int]([2, 0, 1])
    var regressor = RegressorChain(Ridge(0.5), order=order^, cv=PerLinkKFold[DType.float64](7, -6.0))
    var prediction = regressor.fit(X, Y).predict(X)
    parity("ComponentsRegressorChainCustomCV", prediction)
    var params = regressor.get_params()
    assert_equal(params.cv.seed, 7)
    params.cv.seed = -1
    _ = regressor.set_params(params)
    with assert_raises(contains="invalid splitter seed"):
        _ = regressor.fit(X, Y)
    assert_same(regressor.predict(X), prediction)
    params.cv.seed = 7
    _ = regressor.set_params(params)
    var regressor_clone = clone(regressor)
    assert_false(regressor_clone.__sklearn_is_fitted__())
    assert_same(regressor_clone.fit(X, Y).predict(X), prediction)

    var binary = labels[DType.int64](False)
    for i in range(ROWS):
        binary.unsafe_set(i * 3, Int64(1 if i < 4 or i == 7 else 0))
    var classifier = ClassifierChain(
        LogisticRegression(tol=1e-10, max_iter=1000), chain_method="predict_proba",
        cv=PerLinkKFold[DType.int64](7, 0.5),
    )
    var proba = classifier.fit(X, binary).predict_proba(X)
    parity("ComponentsLogisticChainCustomCVProba", proba)
    parity("ComponentsLogisticChainCustomCVLabels", classifier.predict(X))
    var override = classifier.fit(X, binary, cv=3).predict_proba(X)
    assert_true(abs(override.unsafe_get(1) - proba.unsafe_get(1)) > 1e-5)
    assert_same(classifier.fit(X, binary).predict_proba(X), proba)


def chain_cv() raises:
    var X = features[DType.float64]()
    var Y = targets[DType.float64]()
    var order = List[Int]([2, 0, 1])
    var regressor = RegressorChain(Ridge(0.5), order=order.copy(), cv=KFold(3))
    var kfold_prediction = regressor.fit(X, Y).predict(X)
    parity("ComponentsRegressorChainKFold", kfold_prediction)
    var integer_prediction = regressor.fit(X, Y, cv=3).predict(X)
    parity("ComponentsRegressorChainIntegerCV", integer_prediction)
    assert_same(kfold_prediction, integer_prediction)
    with assert_raises(contains="at least 2"):
        _ = regressor.fit(X, Y, cv=1)
    with assert_raises(contains="exceeds sample count"):
        _ = regressor.fit(X, Y, cv=ROWS + 1)
    assert_same(regressor.predict(X), integer_prediction)
    var cloned_regressor = clone(regressor)
    assert_false(cloned_regressor.__sklearn_is_fitted__())
    assert_equal(cloned_regressor.get_params().cv[KFold].n_splits, 3)
    assert_same(cloned_regressor.fit(X, Y).predict(X), kfold_prediction)
    var without_cv = RegressorChain(Ridge(0.5), order=order.copy())
    assert_same(regressor.fit(X, Y, cv=None).predict(X), without_cv.fit(X, Y).predict(X))
    assert_same(regressor.fit(X, Y).predict(X), kfold_prediction)
    parity("ComponentsRegressorChainCVWeighted", regressor.fit(X, Y, sample_weight=weights()).predict(X))

    var shuffled = KFold(3, shuffle=True, random_state=7)
    var cursor = shuffled.split(ROWS)
    var reordered = List[IndexSplit]()
    for _ in range(3):
        var split = cursor.next()
        reordered.insert(0, split.take())
    var fixed_regressor = RegressorChain(Ridge(0.5), order=order.copy(), cv=ChainFolds(reordered))
    var fixed_prediction = fixed_regressor.fit(X, Y).predict(X)
    parity("ComponentsRegressorChainFixedCV", fixed_prediction)
    assert_same(fixed_prediction, regressor.fit(X, Y, cv=shuffled.copy()).predict(X))
    assert_true(abs(fixed_prediction.unsafe_get(0) - kfold_prediction.unsafe_get(0)) > 1e-4)

    # A snapshot remains valid when the caller edits the source folds.
    reordered[0].test.unsafe_set(0, -1)
    with assert_raises(contains="sample index out of bounds"):
        _ = fixed_regressor.fit(X, Y, cv=ChainFolds(reordered))
    reordered[0].test.unsafe_set(0, reordered[0].test.unsafe_get(1))
    with assert_raises(contains="test indices to partition samples"):
        _ = fixed_regressor.fit(X, Y, cv=ChainFolds(reordered))
    assert_same(fixed_regressor.predict(X), fixed_prediction)
    assert_same(fixed_regressor.fit(X, Y).predict(X), fixed_prediction)

    var binary = labels[DType.int64](False)
    for i in range(ROWS):
        binary.unsafe_set(i * 3, Int64(1 if i < 4 or i == 7 else 0))
    var classifier_order = List[Int]([0, 2, 1])
    var label_chain = ClassifierChain(RidgeClassifier(), order=classifier_order.copy(), cv=3)
    parity("ComponentsClassifierChainIntegerCVLabels", label_chain.fit(X, binary).predict(X))
    var classifier = ClassifierChain(RidgeClassifier(), order=classifier_order^, chain_method="decision_function", cv=3)
    var fixed_params = classifier.get_params()
    fixed_params.cv = ChainFolds(shuffled.split(ROWS))
    var fixed_classifier = clone(classifier)
    _ = fixed_classifier.set_params(fixed_params)
    var cloned_fixed = clone(fixed_classifier)
    assert_false(cloned_fixed.__sklearn_is_fitted__())
    assert_true(cloned_fixed.get_params().cv.isa[ChainFolds]())
    var fixed_decision = cloned_fixed.fit(X, binary).decision_function(X)
    parity("ComponentsDecisionChainFixedCV", fixed_decision)
    assert_same(fixed_decision, fixed_classifier.fit(X, binary, cv=shuffled^).decision_function(X))
    var kfold_decision = classifier.fit(X, binary, cv=KFold(3)).decision_function(X)
    parity("ComponentsDecisionChainKFold", kfold_decision)
    var integer_decision = classifier.fit(X, binary).decision_function(X)
    parity("ComponentsDecisionChainIntegerCV", integer_decision)
    var splitter_effect = Float64(0)
    for i in range(integer_decision.size):
        splitter_effect = max(splitter_effect, abs(integer_decision.unsafe_get(i) - kfold_decision.unsafe_get(i)))
    assert_true(splitter_effect > 0.1)
    assert_same(integer_decision, classifier.fit(X, binary, cv=StratifiedKFold(3)).decision_function(X))
    var classifier_params = classifier.get_params()
    assert_equal(classifier_params.cv[Int], 3)
    classifier_params.cv = StratifiedKFold(3, shuffle=True, random_state=7)
    _ = classifier.set_params(classifier_params)
    var shuffled_decision = classifier.fit(X, binary).decision_function(X)
    parity("ComponentsDecisionChainShuffledCV", shuffled_decision)
    var cloned_classifier = clone(classifier)
    assert_false(cloned_classifier.__sklearn_is_fitted__())
    var cloned_cv = cloned_classifier.get_params().cv[StratifiedKFold].copy()
    assert_true(cloned_cv.shuffle)
    assert_equal(cloned_cv.random_state.or_else(-1), 7)
    assert_same(cloned_classifier.fit(X, binary).decision_function(X), shuffled_decision)
    classifier_params.cv = 1
    _ = classifier.set_params(classifier_params)
    with assert_raises(contains="at least 2"):
        _ = classifier.fit(X, binary)
    assert_same(classifier.decision_function(X), shuffled_decision)

    var X32 = features[DType.float32]()
    var forest_target = labels[DType.float64](False)
    # A rare positive label is absent from one training fold. Use non-code
    # labels so the full-data estimator cannot be confused with a CV clone.
    for i in range(ROWS):
        forest_target.unsafe_set(i * 3, 9.0 if i == 0 else -5.0)
    var forest = ClassifierChain[RandomForestClassifier[], DType.float32, DType.float64](
        RandomForestClassifier(n_estimators=5, random_state=0), chain_method="predict_proba", cv=3,
    )
    parity("ComponentsForestChainKFold", forest.fit(X32, forest_target, cv=KFold(3)).predict_proba(X32))
    parity("ComponentsForestChainIntegerCV", forest.fit(X32, forest_target).predict_proba(X32))
    assert_equal(forest.classes_[0].unsafe_get(0), -5.0)
    assert_equal(forest.classes_[0].unsafe_get(1), 9.0)

    var one_target = nm.zeros[DType.float64](Shape(ROWS, 1))
    # The final link is always fitted on the full data and never split.
    var regressor_params = regressor.get_params()
    regressor_params.order = None
    regressor_params.cv = ROWS + 1
    _ = regressor.set_params(regressor_params)
    _ = regressor.fit(X, one_target)
    assert_equal(len(regressor.estimators_), 1)


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
    chain_cv()
    custom_chain_cv()
    logistic_components()
    print("PASS: production linear models and RandomForestClassifier components")
