"""String-domain contracts; independent sklearn comparisons in string_labels_parity.py."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises, TestSuite
from numojo import NDArray, Shape, Item
import numojo as nm
from moxie.base import clone
from moxie.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier, StringLabels, StringTarget
from ovr_fixtures import MarginProbaClassifier, DecisionRegressor
from ovr_incremental_fixtures import OnlineClassifier
from test_ovr import X4, matrix


def assert_strings(y: StringTarget, values: List[String]) raises:
    assert_equal(y.rank(), 1)
    assert_equal(y.rows(), len(values))
    for i in range(len(values)):
        assert_equal(y.get(i), values[i])


def test_fit_order_copy_score_and_refit() raises:
    var y = StringTarget(["猫", "", "é", "é", "猫", ""])
    var ovr = OneVsRestClassifier[MarginProbaClassifier, kind=StringLabels](MarginProbaClassifier(), n_jobs=2)
    var ovo = OneVsOneClassifier[MarginProbaClassifier, kind=StringLabels](MarginProbaClassifier(), n_jobs=2)
    var ecoc = OutputCodeClassifier[DecisionRegressor, kind=StringLabels](DecisionRegressor(), random_state=0, n_jobs=2)
    with assert_raises(contains="NotFittedError"):
        _ = ovr.predict(X4())
    with assert_raises(contains="NotFittedError"):
        _ = ovo.predict(X4())
    with assert_raises(contains="NotFittedError"):
        _ = ecoc.predict(X4())
    ovr.fit(X4(), y)
    ovo.fit(X4(), y)
    ecoc.fit(X4(), y)
    for i in range(3):
        assert_equal(ovr.classes_[i], ["", "é", "猫"][i])
        assert_equal(ovo.classes_[i], ovr.classes_[i])
        assert_equal(ecoc.classes_[i], ovr.classes_[i])
    var unseen = StringTarget(["unseen", "unseen", "unseen", "unseen", "unseen", "unseen"])
    assert_equal(ovr.score(X4(), unseen), 0.0)
    assert_equal(ovo.score(X4(), unseen), 0.0)
    assert_equal(ecoc.score(X4(), unseen), 0.0)
    var prediction = ovr.predict(X4())
    assert_equal(ovr.score(X4(), prediction), 1.0)
    assert_equal(ovo.score(X4(), ovo.predict(X4())), 1.0)
    assert_equal(ecoc.score(X4(), ecoc.predict(X4())), 1.0)
    var weights = nm.full[DType.float64](Shape(6), 1.0)
    assert_equal(ovr.score(X4(), prediction, weights.copy()), 1.0)
    y.set(0, "changed")
    assert_equal(ovr.classes_[2], "猫")
    assert_equal(ovo.classes_[2], "猫")
    assert_equal(ecoc.classes_[2], "猫")
    var fresh = clone(ovr)
    var fresh_ovo = clone(ovo)
    var fresh_ecoc = clone(ecoc)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_false(fresh_ovo.__sklearn_is_fitted__())
    assert_false(fresh_ecoc.__sklearn_is_fitted__())
    var column = StringTarget(["b", "a", "b", "a", "b", "a"], 6, 1)
    ovr.fit(X4(), column)
    ovo.fit(X4(), column)
    ecoc.fit(X4(), column)
    assert_equal(ovr.n_classes_(), 2)
    assert_equal(ovo.n_classes_(), 2)
    assert_equal(len(ecoc.classes_), 2)
    assert_equal(ovr.decision_function(X4()).ndim, 1)
    assert_equal(ovr.predict_proba(X4()).shape[1], 2)
    assert_equal(ovo.decision_function(X4()).ndim, 1)
    var bad = StringTarget(["a", "b", "a", "b", "a", "b"], 3, 2)
    with assert_raises(contains="1d"):
        ovr.fit(X4(), bad)
    with assert_raises(contains="1d"):
        ovo.fit(X4(), bad)
    with assert_raises(contains="1d"):
        ecoc.fit(X4(), bad)
    assert_equal(ovr.n_classes_(), 2)
    var empty = StringTarget(List[String]())
    with assert_raises():
        ovr.fit(X4(), empty)
    with assert_raises():
        ovo.fit(X4(), empty)
    with assert_raises():
        ecoc.fit(X4(), empty)
    with assert_raises(contains="sample_weight"):
        _ = ecoc.score(X4(), column, nm.zeros[DType.float64](Shape(1)))


def test_constant_ties_and_incremental_domains() raises:
    var X = nm.zeros[DType.float64](Shape(3, 2))
    var y = StringTarget(["z", "a", "m"])
    var ovr = OneVsRestClassifier[MarginProbaClassifier, kind=StringLabels](MarginProbaClassifier())
    var ovo = OneVsOneClassifier[MarginProbaClassifier, kind=StringLabels](MarginProbaClassifier())
    var ecoc = OutputCodeClassifier[DecisionRegressor, kind=StringLabels](DecisionRegressor(), random_state=17, code_size=2)
    ovr.fit(X, y)
    ovo.fit(X, y)
    ecoc.fit(X, y)
    assert_strings(ovr.predict(X), ["a", "a", "a"])
    assert_strings(ovo.predict(X), ["a", "a", "a"])
    assert_strings(ecoc.predict(X), ["a", "a", "a"])
    var one = StringTarget(["only", "only", "only"])
    ovr.fit(X, one)
    ecoc.fit(X, one)
    assert_strings(ovr.predict(X), ["only", "only", "only"])
    assert_strings(ecoc.predict(X), ["only", "only", "only"])
    with assert_raises(contains="one class"):
        ovo.fit(X, one)
    var online = OneVsRestClassifier[OnlineClassifier, kind=StringLabels](OnlineClassifier(), n_jobs=2)
    var pairs = OneVsOneClassifier[OnlineClassifier, kind=StringLabels](OnlineClassifier(), n_jobs=2)
    var classes = StringTarget(["z", "m", "a", "z"])
    online.partial_fit(X, y, classes.copy())
    pairs.partial_fit(X, y, classes.copy())
    online.partial_fit(X, y)
    pairs.partial_fit(X, y)
    assert_equal(online.n_classes_(), 3)
    assert_equal(pairs.n_classes_(), 3)
    assert_equal(online.estimators_[0].estimator().updates, 2)
    assert_equal(pairs.estimators_[0].updates, 2)
    with assert_raises(contains="outside classes"):
        online.partial_fit(X, one)
    with assert_raises(contains="outside classes"):
        pairs.partial_fit(X, one)
    with assert_raises(contains="not the same"):
        online.partial_fit(X, y, StringTarget(["a", "m"]))
    with assert_raises(contains="not the same"):
        pairs.partial_fit(X, y, StringTarget(["a", "m"]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
