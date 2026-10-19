"""Consumer coverage for public imports and fluent mutations."""
from std.testing import assert_equal, assert_true
from numojo import Shape, Item
import numojo as nm
from moxie.multiclass import (
    OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier,
    OneVsRestParams, OneVsOneParams, OutputCodeParams,
    PairwiseInput, pairwise_input, StringLabels, StringTarget,
)
from ovr_incremental_fixtures import OnlineClassifier
from ovr_fixtures import MarginProbaClassifier


def main() raises:
    var X = nm.zeros[DType.float64](Shape(6, 2))
    var y = nm.zeros[DType.int64](Shape(6))
    var classes = nm.zeros[DType.int64](Shape(3))
    for i in range(6):
        X[Item(i, 0)] = Float64(i * i)
        X[Item(i, 1)] = Float64(i)
        y[Item(i)] = Int64(i // 2)
    for i in range(3):
        classes[Item(i)] = Int64(i)
    var ovr = OneVsRestClassifier(OnlineClassifier())
    _ = ovr.set_params(ovr.get_params()).partial_fit(X, y, classes.copy()).partial_fit(X, y)
    assert_equal(ovr.predict(X).shape[0], 6)
    assert_equal(ovr.predict_proba(X).shape[1], 3)
    assert_equal(ovr.decision_function(X).shape[1], 3)
    assert_true(not pairwise_input(ovr))
    var ovo = OneVsOneClassifier(OnlineClassifier())
    _ = ovo.set_params(ovo.get_params()).partial_fit(X, y, classes.copy()).partial_fit(X, y)
    assert_equal(ovo.predict(X).shape[0], 6)
    assert_equal(ovo.decision_function(X).shape[1], 3)
    var code = OutputCodeClassifier(MarginProbaClassifier(), random_state=7)
    assert_equal(code.set_params(code.get_params()).fit(X, y).predict(X).shape[0], 6)
    var labels = StringTarget(["c", "c", "a", "a", "b", "b"])
    var names = StringTarget(["c", "a", "b"])
    var string_ovr = OneVsRestClassifier[OnlineClassifier, kind=StringLabels](OnlineClassifier())
    _ = string_ovr.partial_fit(X, labels, names.copy()).partial_fit(X, labels)
    assert_equal(string_ovr.predict(X).rows(), 6)
    assert_equal(string_ovr.predict_proba(X).shape[1], 3)
    var string_ovo = OneVsOneClassifier[OnlineClassifier, kind=StringLabels](OnlineClassifier())
    _ = string_ovo.partial_fit(X, labels, names.copy()).partial_fit(X, labels)
    assert_equal(string_ovo.predict(X).rows(), 6)
    assert_equal(string_ovo.decision_function(X).shape[1], 3)
    var string_code = OutputCodeClassifier[MarginProbaClassifier, kind=StringLabels](MarginProbaClassifier(), random_state=7)
    assert_equal(string_code.fit(X, labels).predict(X).rows(), 6)
    print("PASS: isolated multiclass public consumer")
