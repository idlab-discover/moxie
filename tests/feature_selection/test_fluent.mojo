"""Fluent calls borrow the fitted selector, including generic trait dispatch."""
import numojo as nm
from std.testing import assert_equal, assert_false, assert_raises
from moxie.base import UnsupervisedEstimator, TransformerMixin, clone
from moxie.feature_selection import (
    VarianceThreshold, SelectKBest, SelectPercentile, SelectFpr, SelectFdr,
    SelectFwe, GenericUnivariateSelect, BuiltinScorer, SelectionParameter,
)


def generic_chain[E: UnsupervisedEstimator & TransformerMixin](
    mut estimator: E, X: E.Input, y: Optional[E.Target]
) raises -> E.Transformed:
    var params = estimator.get_params()
    return estimator.set_params(params).fit(X, y).transform(X)


def check[dtype: DType]() raises:
    var X = nm.zeros[dtype](nm.Shape(8, 3))
    var y = nm.zeros[dtype](nm.Shape(8))
    for i in range(8):
        X.unsafe_set(3 * i, 7)
        X.unsafe_set(3 * i + 1, SIMD[dtype, 1](i))
        X.unsafe_set(3 * i + 2, SIMD[dtype, 1](i % 2))
        y.unsafe_set(i, SIMD[dtype, 1](i))
    var variance = VarianceThreshold[dtype]()
    var reduced = variance.set_params(threshold=0).fit(X).transform(X)
    assert_equal(reduced.shape[1], 2)
    assert_equal(variance.n_features_in_(), 3)
    # Mutating through the returned borrow changes the original object.
    _ = variance.fit(X).set_params(threshold=1)
    assert_equal(variance.threshold, 1)
    assert_equal(variance.transform(X).shape[1], 1)
    assert_equal(generic_chain(variance, X, None).shape[1], 1)
    var fresh = clone(variance)
    assert_false(fresh.__sklearn_is_fitted__())
    with assert_raises():
        _ = variance.set_params(threshold=-1).fit(X)
    assert_equal(variance.threshold, -1)
    assert_equal(variance.n_features_in_(), 3)

    var best = SelectKBest(BuiltinScorer[dtype]("f_regression"), k=SelectionParameter(1))
    assert_equal(best.fit(X, y.copy()).transform(X).shape[1], 1)
    assert_equal(best.get_support(True)[List[Int]][0], 1)
    assert_equal(generic_chain(best, X, y.copy()).shape[1], 1)
    var percentile = SelectPercentile(BuiltinScorer[dtype]("f_regression"), percentile=50)
    assert_equal(generic_chain(percentile, X, y.copy()).shape[1], 1)
    var fpr = SelectFpr(BuiltinScorer[dtype]("f_regression"))
    assert_equal(generic_chain(fpr, X, y.copy()).shape[1], 1)
    var fdr = SelectFdr(BuiltinScorer[dtype]("f_regression"))
    assert_equal(generic_chain(fdr, X, y.copy()).shape[1], 1)
    var fwe = SelectFwe(BuiltinScorer[dtype]("f_regression"))
    assert_equal(generic_chain(fwe, X, y.copy()).shape[1], 1)
    var generic = GenericUnivariateSelect(BuiltinScorer[dtype]("f_regression"), mode="k_best", param=SelectionParameter(1))
    assert_equal(generic_chain(generic, X, y.copy()).shape[1], 1)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("fluent dense selector contracts passed")
