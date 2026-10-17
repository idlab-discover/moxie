"""Lifecycle contracts: params, clone independence, repeat fit, failed-fit reset, estimator copies."""
from std.testing import assert_true, assert_false, assert_equal, assert_raises, TestSuite
from std.math import abs
from numojo import NDArray
from moxie.base import clone
from moxie.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    Shrinkage,
)
from support.fixtures import FixtureCovariance, matrix, assert_close

comptime F64 = DType.float64


def toy6() raises -> NDArray[F64]:
    return matrix[F64](9, 2, [0.0, 0, -2, -2, -2, -1, -1, -1, -1, -2, 1, 3, 1, 2, 2, 1, 2, 2])


def y6() -> List[Int]:
    return [1, 1, 1, 1, 1, 2, 2, 2, 2]


def y7() -> List[Int]:
    return [1, 2, 3, 2, 3, 1, 2, 3, 1]


def test_defaults_match_sklearn() raises:
    var lda = LinearDiscriminantAnalysis()
    var p = lda.get_params()
    assert_equal(p.solver, "svd")
    assert_true(p.shrinkage.is_none())
    assert_false(Bool(p.priors))
    assert_false(Bool(p.n_components))
    assert_false(p.store_covariance)
    assert_equal(p.tol, 1e-4)
    assert_false(Bool(p.covariance_estimator))
    var qda = QuadraticDiscriminantAnalysis()
    var q = qda.get_params()
    assert_equal(q.solver, "svd")
    assert_equal(q.reg_param, 0.0)
    assert_equal(q.tol, 1e-4)
    assert_false(q.store_covariance)


def test_set_params_typed_and_keyword() raises:
    var lda = LinearDiscriminantAnalysis()
    lda.set_params(solver=String("eigen"), shrinkage=Shrinkage("auto"), n_components=Optional[Int](1))
    assert_equal(lda.solver, "eigen")
    assert_true(lda.shrinkage.is_auto())
    lda.fit(toy6(), y7())
    assert_equal(lda.transform(toy6()).shape[1], 1)
    var params = lda.get_params()
    params.solver = "lsqr"
    params.shrinkage = Shrinkage()
    lda.set_params(params^)
    assert_equal(lda.solver, "lsqr")
    assert_true(lda.shrinkage.is_none())
    # Parameter changes do not touch learned state until the next fit.
    assert_true(lda.__sklearn_is_fitted__())
    lda.fit(toy6(), y7())
    assert_false(Bool(lda.scalings_))


def test_clone_is_unfitted_and_independent() raises:
    var priors: List[Float64] = [0.3, 0.7]
    var lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.2, priors=priors^)
    lda.fit(toy6(), y6())
    var copy = clone(lda)
    assert_false(copy.__sklearn_is_fitted__())
    assert_equal(copy.solver, "lsqr")
    assert_equal(copy.priors.value()[0], 0.3)
    copy.fit(toy6(), y6())
    assert_close(copy.coef_, lda.coef_, 0.0, "clone refit")
    copy.priors.value()[0] = 0.9
    assert_equal(lda.priors.value()[0], 0.3)
    var qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
    qda.fit(toy6(), y6())
    var qcopy = clone(qda)
    assert_false(qcopy.__sklearn_is_fitted__())
    assert_equal(qcopy.reg_param, 0.1)


def test_repeat_fit_replaces_state() raises:
    var lda = LinearDiscriminantAnalysis()
    lda.fit(toy6(), y7())
    assert_equal(len(lda.classes_), 3)
    lda.fit(toy6(), y6())
    assert_equal(len(lda.classes_), 2)
    assert_equal(lda.coef_.shape[0], 1)
    var first = lda.coef_.copy()
    lda.fit(toy6(), y6())
    assert_close(lda.coef_, first, 0.0, "deterministic refit")
    var one = matrix[F64](9, 1, [-3.0, -2, -1, -1, 0, 1, 1, 2, 3])
    lda.fit(one, y6())
    assert_equal(lda.n_features_in_, 1)
    with assert_raises(contains="expecting 1 features"):
        _ = lda.predict(toy6())


def test_failed_fit_resets_to_unfitted() raises:
    var qda = QuadraticDiscriminantAnalysis()
    qda.fit(toy6(), y6())
    assert_true(qda.__sklearn_is_fitted__())
    var x2 = matrix[F64](9, 2, [-3.0, 0, -2, 0, -1, 0, -1, 0, 0, 0, 1, 0, 1, 0, 2, 0, 3, 0])
    with assert_raises(contains="LinAlgError"):
        qda.fit(x2, y6())
    assert_false(qda.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = qda.predict(x2)
    var lda = LinearDiscriminantAnalysis(n_components=5)
    with assert_raises(contains="n_components"):
        lda.fit(toy6(), y7())
    assert_false(lda.__sklearn_is_fitted__())


def test_covariance_estimator_is_copied_not_mutated() raises:
    var estimator = FixtureCovariance("shrunk", 0.5)
    var lda = LinearDiscriminantAnalysis[F64, Int, FixtureCovariance](
        solver="lsqr", covariance_estimator=estimator.copy()
    )
    lda.fit(toy6(), y7())
    assert_equal(lda.covariance_estimator.value().fit_calls, 0)
    assert_false(Bool(lda.covariance_estimator.value().fitted))
    # Equivalent to shrinkage=0.5 (sklearn test_lda_shrinkage).
    var reference = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.5)
    reference.fit(toy6(), y7())
    assert_close(lda.covariance_.value(), reference.covariance_.value(), 1e-12, "shrunk estimator")
    assert_close(lda.coef_, reference.coef_, 1e-10, "shrunk estimator coef")
    # Standardized Ledoit-Wolf estimator equals shrinkage='auto' (test_lda_ledoitwolf / test_qda_ledoitwolf).
    var lw = QuadraticDiscriminantAnalysis[F64, Int, FixtureCovariance](
        solver="eigen", store_covariance=True, covariance_estimator=FixtureCovariance("stdlw")
    )
    lw.fit(toy6(), y6())
    var auto = QuadraticDiscriminantAnalysis(solver="eigen", store_covariance=True, shrinkage="auto")
    auto.fit(toy6(), y6())
    for c in range(2):
        assert_close(lw.covariance_.value()[c], auto.covariance_.value()[c], 1e-12, "stdlw")


def test_fitted_arrays_do_not_alias_input() raises:
    var X = toy6()
    var lda = LinearDiscriminantAnalysis()
    lda.fit(X, y6())
    var means = lda.means_.copy()
    X.unsafe_set(0, 100.0)
    assert_close(lda.means_, means, 0.0, "input mutation")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
