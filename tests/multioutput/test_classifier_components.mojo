"""Dense classifier components with list probabilities or list weights."""
import numojo as nm
from numojo import NDArray, Shape
from std.testing import assert_equal, assert_raises
from moxie.base import clone
from moxie.discriminant_analysis import LinearDiscriminantAnalysis
from moxie.dummy import DummyClassifier, Label
from moxie.naive_bayes import GaussianNB, NumericDomain
from moxie.neighbors import KNeighborsClassifier
from moxie.svm import SVC
from moxie.tree import DecisionTreeClassifier
from moxie.multioutput import ClassifierChain, MultiOutputClassifier


def features() raises -> NDArray[DType.float64]:
    var X = nm.zeros[DType.float64](Shape(12, 1))
    for i in range(12):
        X.unsafe_set(i, Float64(i * i) + 0.1 * Float64(i))
    return X^


def targets(binary: Bool) raises -> NDArray[DType.float64]:
    var y = nm.zeros[DType.float64](Shape(12, 2))
    for i in range(12):
        y.unsafe_set(2 * i, -3.0 if i < 6 else 4.0)
        y.unsafe_set(2 * i + 1, 8.0 if i < 4 else (11.0 if binary or i < 8 else 19.0))
    return y^


def parity[name: StaticString, dtype: DType](values: NDArray[dtype]) raises:
    var line = "PARITY " + String(name)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            line += " " + String(Float64(values.item(i, j)))
    print(line)


def main() raises:
    var X = features()
    var multiclass = targets(False)
    var binary = targets(True)
    comptime Tree = DecisionTreeClassifier[]
    var tree = MultiOutputClassifier[Tree, DType.float32, DType.float64](
        Tree(max_depth=2, random_state=17),
    )
    var X32 = X.astype[DType.float32]()
    with assert_raises(contains="NotFittedError"):
        _ = tree.predict_proba(X32)
    _ = tree.fit(X32, multiclass)
    var tree_proba = tree.predict_proba(X32)
    assert_equal(len(tree_proba), 2)
    assert_equal(tree_proba[0].shape, Shape(12, 2))
    assert_equal(tree_proba[1].shape, Shape(12, 3))
    assert_equal(tree.classes_[1].item(2), 19.0)
    parity["ComponentsTreeLabels"](tree.predict(X32))
    parity["ComponentsTreeProba"](tree_proba[1])

    var tree_chain = ClassifierChain[Tree, DType.float32, DType.float64](
        Tree(max_depth=2, random_state=17), chain_method="predict_proba", cv=3,
    )
    parity["ComponentsTreeChainCVProba"](tree_chain.fit(X32, binary).predict_proba(X32))
    parity["ComponentsTreeChainCVLabels"](tree_chain.predict(X32))
    var rare = binary.copy()
    for i in range(12):
        rare.unsafe_set(2 * i, 4.0 if i == 11 else -3.0)
    # One fold trains on a single class, so its sole probability column must
    # be aligned to the full vocabulary before selecting the positive class.
    parity["ComponentsTreeChainMissingClassProba"](tree_chain.fit(X32, rare).predict_proba(X32))

    var constant = nm.zeros[DType.float32](Shape(12, 1))
    var log_chain = ClassifierChain[Tree, DType.float32, DType.float64](
        Tree(max_depth=2, random_state=17), chain_method="predict_log_proba", cv=3,
    )
    parity["ComponentsTreeChainLogResponseProba"](
        log_chain.fit(constant, binary).predict_proba(constant),
    )

    comptime KNN = KNeighborsClassifier[]
    var neighbors = MultiOutputClassifier[KNN, DType.float64, DType.float64](
        KNN(n_neighbors=3, algorithm="brute"),
    )
    _ = neighbors.fit(X, multiclass)
    parity["ComponentsKNNLabels"](neighbors.predict(X))
    parity["ComponentsKNNProba"](neighbors.predict_proba(X)[1])

    comptime NB = GaussianNB[NumericDomain[DType.float64]]
    var bayes = MultiOutputClassifier[NB, DType.float64, DType.float64](NB())
    var weights = nm.zeros[DType.float64](Shape(12))
    for i in range(12):
        weights.unsafe_set(i, Float64(i % 3 + 1))
    _ = bayes.fit(X, multiclass, sample_weight=weights)
    parity["ComponentsGaussianNBLabels"](bayes.predict(X))
    parity["ComponentsGaussianNBProba"](bayes.predict_proba(X)[1])
    var bad_weights = nm.zeros[DType.float64](Shape(12, 1))
    with assert_raises(contains="sample weights must be a vector"):
        _ = bayes.fit(X, multiclass, sample_weight=bad_weights)
    assert_equal(bayes.predict(X).item(0, 1), 8.0)

    var bayes_chain = ClassifierChain[NB, DType.float64, DType.float64](
        clone(bayes.estimator), chain_method="predict_proba",
    )
    _ = bayes_chain.fit(X, binary, sample_weight=weights)
    parity["ComponentsGaussianNBChainProba"](bayes_chain.predict_proba(X))

    var integer_targets = multiclass.astype[DType.int64]()
    var list_bayes = MultiOutputClassifier(GaussianNB())
    with assert_raises(contains="first partial_fit requires classes"):
        _ = list_bayes.partial_fit(X, integer_targets)
    _ = list_bayes.fit(X, integer_targets, sample_weight=weights)
    assert_equal(list_bayes.classes_[1].item(2), Int64(19))
    parity["ComponentsGaussianNBListLabels"](list_bayes.predict(X))
    parity["ComponentsGaussianNBListProba"](list_bayes.predict_proba(X)[1])

    var incremental = clone(list_bayes)
    var first_X = nm.zeros[DType.float64](Shape(6, 1))
    var second_X = first_X.copy()
    var first_y = nm.zeros[DType.int64](Shape(6, 2))
    var second_y = first_y.copy()
    for i in range(6):
        first_X.unsafe_set(i, X.unsafe_get(i))
        second_X.unsafe_set(i, X.unsafe_get(6 + i))
        for j in range(2):
            first_y.unsafe_set(2 * i + j, integer_targets.item(i, j))
            second_y.unsafe_set(2 * i + j, integer_targets.item(6 + i, j))
    _ = incremental.partial_fit(first_X, first_y, classes=list_bayes.classes_.copy())
    _ = incremental.partial_fit(second_X, second_y)
    parity["ComponentsGaussianNBListIncrementalLabels"](incremental.predict(X))
    parity["ComponentsGaussianNBListIncrementalProba"](incremental.predict_proba(X)[1])

    var list_chain = ClassifierChain(GaussianNB(), chain_method="predict_proba", cv=3)
    var integer_binary = binary.astype[DType.int64]()
    _ = list_chain.fit(X, integer_binary)
    parity["ComponentsGaussianNBListChainCVLabels"](list_chain.predict(X))
    parity["ComponentsGaussianNBListChainCVProba"](list_chain.predict_proba(X))

    comptime LDA = LinearDiscriminantAnalysis[DType.float64, Float64]
    var lda = MultiOutputClassifier[LDA, DType.float64, DType.float64](LDA())
    _ = lda.fit(X, multiclass)
    parity["ComponentsLDAListLabels"](lda.predict(X))
    parity["ComponentsLDAListProba"](lda.predict_proba(X)[1])

    var dummy = MultiOutputClassifier(DummyClassifier())
    _ = dummy.fit(X, integer_targets, sample_weight=weights)
    assert_equal(dummy.estimators_[1].classes_[2] == Label(19), True)
    parity["ComponentsDummyWeightedLabels"](dummy.predict(X))
    parity["ComponentsDummyWeightedProba"](dummy.predict_proba(X)[1])
    var unfitted_dummy = clone(dummy)
    with assert_raises(contains="NotFittedError"):
        _ = unfitted_dummy.predict(X)

    # Adjacent integers beyond Float64's exact range must stay distinct.
    var wide_labels = nm.full[DType.int64](Shape(12, 1), Int64(9007199254740992))
    wide_labels.unsafe_set(0, Int64(9007199254740993))
    var constant_dummy = MultiOutputClassifier(
        DummyClassifier(strategy="constant", constant=Label(9007199254740993)),
    )
    _ = constant_dummy.fit(X, wide_labels)
    assert_equal(constant_dummy.classes_[0].size, 2)
    assert_equal(constant_dummy.predict(X).item(0, 0), Int64(9007199254740993))

    var dummy_chain = ClassifierChain[DummyClassifier[], DType.float64, DType.float64](
        DummyClassifier(), order=List[Int]([1, 0]), chain_method="predict_proba", cv=3,
    )
    _ = dummy_chain.fit(X, binary)
    assert_equal(dummy_chain.estimators_[0].classes_[1] == Label(11.0), True)
    parity["ComponentsDummyChainCVLabels"](dummy_chain.predict(X))
    parity["ComponentsDummyChainCVProba"](dummy_chain.predict_proba(X))

    comptime Dummy32 = DummyClassifier[False, DType.float32]
    var dummy32 = MultiOutputClassifier[Dummy32, DType.float32, DType.float32](Dummy32())
    var targets32 = multiclass.astype[DType.float32]()
    _ = dummy32.fit(X32, targets32, sample_weight=weights)
    parity["ComponentsDummy32WeightedLabels"](dummy32.predict(X32))
    parity["ComponentsDummy32WeightedProba"](dummy32.predict_proba(X32)[1])

    var svc = MultiOutputClassifier(SVC[Int](kernel="linear"))
    _ = svc.fit(X, integer_targets)
    parity["ComponentsSVCLabels"](svc.predict(X))
    var svc_chain = ClassifierChain(
        SVC[Int](kernel="linear"), chain_method="decision_function", cv=3,
    )
    _ = svc_chain.fit(X, integer_binary)
    parity["ComponentsSVCChainCVLabels"](svc_chain.predict(X))
    parity["ComponentsSVCChainCVDecision"](svc_chain.decision_function(X))
