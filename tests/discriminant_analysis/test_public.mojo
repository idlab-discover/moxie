"""External consumers use both estimators and current fluent shared contracts."""
from std.testing import assert_equal, assert_false, assert_true, assert_raises
from numojo import Shape, zeros, NDArray
from moxie.base import ClassifierMixin, clone
from moxie.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


def fit_predict[E: ClassifierMixin](mut model: E, params: E.Params, X: E.Input, y: E.Target) raises -> E.Prediction:
    return model.set_params(params).fit(X, y).predict(X)


def check[dtype: DType]() raises:
    var X = zeros[dtype](Shape(4, 1))
    X.unsafe_set(0, -2.0)
    X.unsafe_set(1, -1.0)
    X.unsafe_set(2, 1.0)
    X.unsafe_set(3, 2.0)
    var y: List[Int] = [0, 0, 1, 1]
    var lda = LinearDiscriminantAnalysis[dtype]()
    assert_equal(fit_predict(lda, lda.get_params(), X, y), y)
    assert_true(lda.__sklearn_is_fitted__())
    assert_equal(lda.transform(X).shape[1], 1)
    assert_equal(lda.set_params(solver="lsqr").fit(X, y).predict(X), y)
    assert_equal(lda.solver, "lsqr")
    assert_equal(lda.set_params(solver="eigen").fit(X, y).score(X, y), 1.0)
    var fresh = clone(lda)
    assert_false(fresh.__sklearn_is_fitted__())
    var invalid = zeros[dtype](Shape(0, 1))
    with assert_raises():
        _ = lda.fit(invalid, y)
    assert_false(lda.__sklearn_is_fitted__())
    var qda = QuadraticDiscriminantAnalysis[dtype]()
    assert_equal(fit_predict(qda, qda.get_params(), X, y), y)
    assert_true(qda.__sklearn_is_fitted__())
    assert_equal(qda.set_params(solver="eigen").fit(X, y).predict(X), y)
    assert_equal(qda.solver, "eigen")
    var fresh_qda = clone(qda)
    assert_false(fresh_qda.__sklearn_is_fitted__())
    var words: List[String] = ["left", "left", "right", "right"]
    var text_lda = LinearDiscriminantAnalysis[dtype, String]()
    var text_qda = QuadraticDiscriminantAnalysis[dtype, String]()
    assert_equal(fit_predict(text_lda, text_lda.get_params(), X, words), words)
    assert_equal(fit_predict(text_qda, text_qda.get_params(), X, words), words)


def main() raises:
    check[DType.float64]()
    check[DType.float32]()
    print("Public LDA/QDA consumers: solvers, fluent contracts, dtypes, labels and cloning passed")
