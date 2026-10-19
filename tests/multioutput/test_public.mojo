"""Fluent contracts, fitted copies and production regression composition."""
import numojo as nm
from moxie.base import BaseEstimator, SupervisedEstimator, Predictor, clone
from moxie.dummy import DummyRegressor
from moxie.multioutput import (
    MultiOutputRegressor, MultiOutputClassifier, RegressorChain, ClassifierChain,
    IndependentParams, ChainParams, ChainResponses,
    MultiOutputClassifierAPI, MultiOutputRegressorAPI, MultiOutputProbability,
)
from test_estimators import Nearest, Params
from std.testing import assert_equal, assert_false


def fluent[E: SupervisedEstimator & Predictor](
    mut model: E, X: E.Input, y: E.Target, params: E.Params,
) raises -> E.Prediction:
    return model.set_params(params).fit(X, y).predict(X)


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(3, 1))
    var y = nm.ones[DType.float64](nm.Shape(3, 2))
    for i in range(3):
        X.unsafe_set(i, Float64(i))
    var reg = MultiOutputRegressor(Nearest[DType.float64]())
    var prediction = fluent(reg, X, y, reg.get_params())
    assert_equal(prediction.shape[1], 2)
    var fitted = reg.copy()
    fitted.estimators_[0].y.unsafe_set(0, 7.0)
    assert_equal(Float64(reg.predict(X).unsafe_get(0)), 1.0)
    assert_equal(Float64(fitted.predict(X).unsafe_get(0)), 7.0)
    assert_equal(Float64(fitted.predict(X).unsafe_get(1)), 1.0)
    var fresh = clone(fitted)
    assert_false(fresh.__sklearn_is_fitted__())
    var rc = RegressorChain(Nearest[DType.float64]())
    _ = fluent(rc, X, y, rc.get_params())
    var copied_chain = rc.copy()
    copied_chain.order_[0] = 1
    assert_equal(rc.order_[0], 0)
    var labels = nm.zeros[DType.int64](nm.Shape(3, 2))
    labels.unsafe_set(2, 1)
    labels.unsafe_set(3, 1)
    var clf = MultiOutputClassifier(Nearest[DType.int64]())
    _ = fluent(clf, X, labels, clf.get_params())
    var copied_clf = clf.copy()
    copied_clf.estimators_[0].y.unsafe_set(0, 1)
    assert_equal(clf.predict(X).unsafe_get(0), Int64(0))
    assert_equal(copied_clf.predict(X).unsafe_get(0), Int64(1))
    var cc = ClassifierChain(Nearest[DType.int64]())
    _ = fluent(cc, X, labels, cc.get_params())
    var independent: IndependentParams[Params] = clf.get_params()
    var chain: ChainParams[Params] = cc.get_params()
    assert_equal(independent.estimator.dummy, 0)
    assert_equal(chain.chain_method, "predict")
    print("PASS: public records, generic fluent composition and independent fitted copies")
