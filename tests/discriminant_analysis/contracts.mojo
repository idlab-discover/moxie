"""Native contracts: invalid parameters/input, unfitted errors, shapes, labels, capabilities."""
from std.testing import assert_true, assert_false, assert_equal, assert_raises, TestSuite
from std.math import abs, exp
from numojo import NDArray, Shape, zeros
from moxie.base import ClassifierMixin, SupervisedEstimator
from moxie.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    DiscriminantClassifier,
    LinearDiscriminantTransformer,
    Shrinkage,
)
from support.fixtures import FixtureCovariance, matrix

comptime F64 = DType.float64


def toy() raises -> NDArray[F64]:
    return matrix[F64](6, 2, [-2.0, -1, -1, -1, -1, -2, 1, 1, 1, 2, 2, 1])


def toy6() raises -> NDArray[F64]:
    return matrix[F64](9, 2, [0.0, 0, -2, -2, -2, -1, -1, -1, -1, -2, 1, 3, 1, 2, 2, 1, 2, 2])


def y2() -> List[Int]:
    return [1, 1, 1, 2, 2, 2]


def y6() -> List[Int]:
    return [1, 1, 1, 1, 1, 2, 2, 2, 2]


def y7() -> List[Int]:
    return [1, 2, 3, 2, 3, 1, 2, 3, 1]


def fit_it[E: SupervisedEstimator & Deinitable](var model: E, X: E.Input, y: E.Target) raises:
    model.fit(X, y)


def generic_labels[E: ClassifierMixin](model: E, X: E.Input) raises -> E.Prediction:
    return model.predict(X)


def decision_of[E: DiscriminantClassifier](model: E, X: E.Input) raises -> E.Decision:
    return model.decision_function(X)


def transform_of[E: LinearDiscriminantTransformer](model: E, X: E.Input) raises -> E.Transformed:
    return model.transform(X)


def test_trait_dispatch() raises:
    var lda = LinearDiscriminantAnalysis()
    lda.fit(toy(), y2())
    assert_equal(len(generic_labels(lda, toy())), 6)
    assert_equal(decision_of(lda, toy()).ndim, 1)
    assert_equal(transform_of(lda, toy()).shape[1], 1)
    var qda = QuadraticDiscriminantAnalysis()
    qda.fit(toy6(), y6())
    assert_equal(len(generic_labels(qda, toy6())), 9)


def test_unfitted_errors() raises:
    var lda = LinearDiscriminantAnalysis()
    assert_false(lda.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = lda.predict(toy())
    with assert_raises(contains="NotFittedError"):
        _ = lda.transform(toy())
    with assert_raises(contains="NotFittedError"):
        _ = lda.get_feature_names_out()
    var qda = QuadraticDiscriminantAnalysis()
    with assert_raises(contains="NotFittedError"):
        _ = qda.predict_proba(toy())
    # lsqr transform is rejected before the fitted check, like sklearn.
    var lsqr = LinearDiscriminantAnalysis(solver="lsqr")
    with assert_raises(contains="NotImplementedError: transform not implemented for 'lsqr'"):
        _ = lsqr.transform(toy())


def test_invalid_parameters() raises:
    with assert_raises(contains="'solver' parameter"):
        fit_it(LinearDiscriminantAnalysis(solver="cholesky"), toy(), y2())
    with assert_raises(contains="'shrinkage' parameter"):
        fit_it(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="full"), toy(), y2())
    with assert_raises(contains="'shrinkage' parameter"):
        fit_it(LinearDiscriminantAnalysis(solver="lsqr", shrinkage=-0.1), toy(), y2())
    with assert_raises(contains="'n_components' parameter"):
        fit_it(LinearDiscriminantAnalysis(n_components=0), toy(), y2())
    with assert_raises(contains="'tol' parameter"):
        fit_it(LinearDiscriminantAnalysis(tol=-1.0), toy(), y2())
    with assert_raises(contains="NotImplementedError: shrinkage not supported with 'svd'"):
        fit_it(LinearDiscriminantAnalysis(shrinkage="auto"), toy(), y2())
    with assert_raises(contains="covariance estimator is not supported with svd"):
        fit_it(LinearDiscriminantAnalysis[F64, Int, FixtureCovariance](covariance_estimator=FixtureCovariance("lw")), toy(), y2())
    with assert_raises(contains="Only one of the two can be set"):
        fit_it(LinearDiscriminantAnalysis[F64, Int, FixtureCovariance](
            solver="lsqr", shrinkage=0.1, covariance_estimator=FixtureCovariance("shrunk", 0.5)
        ), toy(), y2())
    with assert_raises(contains="'solver' parameter"):
        fit_it(QuadraticDiscriminantAnalysis(solver="lsqr"), toy6(), y6())
    with assert_raises(contains="'reg_param' parameter"):
        fit_it(QuadraticDiscriminantAnalysis(reg_param=1.5), toy6(), y6())
    with assert_raises(contains="NotImplementedError"):
        fit_it(QuadraticDiscriminantAnalysis(shrinkage="auto"), toy6(), y6())
    with assert_raises(contains="covariance_estimator is not supported with solver='svd'"):
        fit_it(QuadraticDiscriminantAnalysis[F64, Int, FixtureCovariance](covariance_estimator=FixtureCovariance("lw")), toy6(), y6())


def test_invalid_data() raises:
    var lda = LinearDiscriminantAnalysis()
    with assert_raises(contains="The number of samples must be more than the number of classes"):
        lda.fit(matrix[F64](2, 2, [0.5, 0.6, 0.6, 0.5]), [0, 1])
    with assert_raises(contains="minimum of 2 is required"):
        lda.fit(matrix[F64](1, 2, [0.5, 0.6]), [0])
    with assert_raises(contains="inconsistent numbers of samples"):
        lda.fit(toy(), [1, 2])
    var bad = toy()
    bad.unsafe_set(3, Float64("nan"))
    with assert_raises(contains="NaN or infinity"):
        lda.fit(bad, y2())
    with assert_raises(contains="expected a 2D array"):
        lda.fit(zeros[F64](Shape(6)), y2())
    with assert_raises(contains="priors must be non-negative"):
        var priors: List[Float64] = [0.5, -0.5]
        fit_it(LinearDiscriminantAnalysis(priors=priors^), toy(), y2())
    with assert_raises(contains="priors has 3 entries"):
        var priors: List[Float64] = [0.2, 0.3, 0.5]
        fit_it(LinearDiscriminantAnalysis(priors=priors^), toy(), y2())
    with assert_raises(contains="n_components cannot be larger"):
        fit_it(LinearDiscriminantAnalysis(n_components=2), toy(), y2())
    lda.fit(toy(), y2())
    with assert_raises(contains="features, but LinearDiscriminantAnalysis is expecting 2"):
        _ = lda.predict(matrix[F64](1, 3, [1.0, 2, 3]))
    var qda = QuadraticDiscriminantAnalysis()
    with assert_raises(contains="The number of classes has to be greater than one"):
        qda.fit(toy6(), [1, 1, 1, 1, 1, 1, 1, 1, 1])
    with assert_raises(contains="y has only 1 sample in class 2"):
        qda.fit(toy6(), [1, 1, 1, 1, 1, 1, 1, 1, 2])
    var x2 = matrix[F64](9, 2, [-3.0, 0, -2, 0, -1, 0, -1, 0, 0, 0, 1, 0, 1, 0, 2, 0, 3, 0])
    with assert_raises(contains="LinAlgError: The covariance matrix of class 1 is not full rank. Increase the value of `reg_param`"):
        qda.fit(x2, y6())
    with assert_raises(contains="Increase the value of `shrinkage`"):
        fit_it(QuadraticDiscriminantAnalysis(solver="eigen"), x2, y6())
    with assert_raises(contains="When using `solver='svd'`"):
        qda.fit(matrix[F64](5, 3, [1.0, 2, 3, 0, 1, 1, 2, 0, 3, 5, 1, 2, 1, 1, 0]), [1, 1, 2, 2, 2])
    # Regularization repairs the constant-feature case (sklearn test_qda_regularization).
    var reg = QuadraticDiscriminantAnalysis(reg_param=0.01)
    reg.fit(x2, y6())
    var pred = reg.predict(x2)
    for i in range(9):
        assert_equal(pred[i], y6()[i])
    var shrunk = QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=0.01)
    shrunk.fit(x2, y6())
    with assert_raises(contains="LinAlgError"):
        fit_it(LinearDiscriminantAnalysis(solver="eigen"), x2, y6())


def test_shapes_binary_and_multiclass() raises:
    for solver in ["svd", "lsqr", "eigen"]:
        var lda = LinearDiscriminantAnalysis(solver=solver)
        lda.fit(toy6(), y6())
        assert_equal(lda.coef_.shape[0], 1)
        assert_equal(lda.coef_.shape[1], 2)
        assert_equal(lda.intercept_.shape[0], 1)
        assert_equal(lda.decision_function(toy6()).ndim, 1)
        var proba = lda.predict_proba(toy6())
        assert_equal(proba.shape[1], 2)
        var log_proba = lda.predict_log_proba(toy6())
        for i in range(proba.size):
            assert_true(abs(exp(Float64(log_proba.unsafe_get(i))) - Float64(proba.unsafe_get(i))) < 1e-6)
        lda.fit(toy6(), y7())
        assert_equal(lda.coef_.shape[0], 3)
        assert_equal(lda.intercept_.shape[0], 3)
        var decision = lda.decision_function(toy6())
        assert_equal(decision.ndim, 2)
        assert_equal(decision.shape[1], 3)
        proba = lda.predict_proba(toy6())
        for i in range(9):
            var total: Float64 = 0
            for c in range(3):
                total += Float64(proba.unsafe_get(i * 3 + c))
            assert_true(abs(total - 1.0) < 1e-12)
        if solver == "lsqr":
            assert_false(Bool(lda.scalings_))
            assert_false(Bool(lda.explained_variance_ratio_))
        else:
            assert_equal(lda.transform(toy6()).shape[1], 2)
            assert_equal(lda.explained_variance_ratio_.value().shape[0], 2)
        assert_equal(Bool(lda.xbar_), solver == "svd")
        assert_equal(Bool(lda.covariance_), solver != "svd")
        assert_equal(len(lda.get_feature_names_out()), 2)
        assert_equal(lda.get_feature_names_out()[1], "lineardiscriminantanalysis1")
    var qda = QuadraticDiscriminantAnalysis()
    qda.fit(toy6(), y7())
    assert_equal(qda.decision_function(toy6()).shape[1], 3)
    assert_equal(len(qda.rotations_), 3)
    assert_equal(qda.rotations_[0].shape[0], 2)
    assert_false(Bool(qda.covariance_))
    qda.fit(toy6(), y6())
    assert_equal(qda.decision_function(toy6()).ndim, 1)


def test_n_components_limits_transform() raises:
    var lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(toy6(), y7())
    assert_equal(lda.transform(toy6()).shape[1], 1)
    assert_equal(lda.explained_variance_ratio_.value().shape[0], 1)
    var eigen = LinearDiscriminantAnalysis(solver="eigen", n_components=1)
    var out = eigen.fit_transform(toy6(), y7())
    assert_equal(out.shape[1], 1)
    with assert_raises(contains="requires y"):
        _ = eigen.fit_transform(toy6())


def test_string_labels_and_order() raises:
    var labels: List[String] = ["pear", "apple", "pear", "fig", "apple", "fig", "pear", "apple", "fig"]
    var lda = LinearDiscriminantAnalysis[F64, String]()
    lda.fit(toy6(), labels)
    assert_equal(lda.classes_[0], "apple")
    assert_equal(lda.classes_[1], "fig")
    assert_equal(lda.classes_[2], "pear")
    assert_equal(len(lda.predict(toy6())), 9)
    var qda = QuadraticDiscriminantAnalysis[F64, String]()
    var two: List[String] = ["b", "b", "b", "b", "b", "a", "a", "a", "a"]
    qda.fit(toy6(), two)
    assert_equal(qda.classes_[0], "a")
    assert_equal(qda.predict(toy6())[0], "b")


def test_priors_and_renormalization() raises:
    var priors: List[Float64] = [0.5, 0.6]
    var lda = LinearDiscriminantAnalysis(priors=priors^)
    lda.fit(toy(), y2())
    assert_true(abs(Float64(lda.priors_.unsafe_get(0)) - 0.5 / 1.1) < 1e-12)
    var tiny: List[Float64] = [1e-30, 1.0 - 1e-30]
    var qda = QuadraticDiscriminantAnalysis(priors=tiny^)
    qda.fit(toy(), y2())
    var base = QuadraticDiscriminantAnalysis()
    base.fit(toy(), y2())
    var count_tiny = 0
    var count_base = 0
    var test = matrix[F64](3, 2, [0.0, 0.0, -0.3, 0.1, 0.2, -0.4])
    for p in qda.predict(test):
        if p == 2:
            count_tiny += 1
    for p in base.predict(test):
        if p == 2:
            count_base += 1
    assert_true(count_tiny >= count_base)


def test_score_with_sample_weight() raises:
    var lda = LinearDiscriminantAnalysis()
    lda.fit(toy(), y2())
    var wrong: List[Int] = [1, 1, 1, 2, 2, 1]
    assert_true(abs(lda.score(toy(), wrong) - 5.0 / 6.0) < 1e-15)
    var w = matrix[F64](1, 6, [1.0, 1, 1, 1, 1, 5]).reshape(Shape(6))
    assert_true(abs(lda.score(toy(), wrong, w^) - 5.0 / 10.0) < 1e-15)


def test_lda_invalid_score_weights() raises:
    var model = LinearDiscriminantAnalysis()
    model.fit(toy(), y2())
    var zero = zeros[F64](Shape(6))
    with assert_raises(contains="ValueError: Sample weights must contain at least one non-zero number"):
        _ = model.score(toy(), y2(), zero^)
    var cancel = matrix[F64](1, 6, [1.0, -1, 0, 0, 0, 0]).reshape(Shape(6))
    with assert_raises(contains="ZeroDivisionError: Weights sum to zero"):
        _ = model.score(toy(), y2(), cancel^)
    var invalid = matrix[F64](1, 6, [Float64("nan"), 1, 1, 1, 1, 1]).reshape(Shape(6))
    with assert_raises(contains="ValueError: Input sample_weight contains NaN"):
        _ = model.score(toy(), y2(), invalid^)
    var infinite = matrix[F64](1, 6, [Float64("inf"), 1, 1, 1, 1, 1]).reshape(Shape(6))
    with assert_raises(contains="ValueError: Input sample_weight contains infinity"):
        _ = model.score(toy(), y2(), infinite^)
    var signed = matrix[F64](1, 6, [1.0, 1, 1, 1, 1, -1]).reshape(Shape(6))
    var wrong: List[Int] = [1, 1, 1, 2, 2, 1]
    assert_true(abs(model.score(toy(), wrong, signed^) - 1.25) < 1e-15)


def test_qda_invalid_score_weights() raises:
    var model = QuadraticDiscriminantAnalysis()
    model.fit(toy(), y2())
    var zero = zeros[F64](Shape(6))
    with assert_raises(contains="ValueError: Sample weights must contain at least one non-zero number"):
        _ = model.score(toy(), y2(), zero^)
    var cancel = matrix[F64](1, 6, [1.0, -1, 0, 0, 0, 0]).reshape(Shape(6))
    with assert_raises(contains="ZeroDivisionError: Weights sum to zero"):
        _ = model.score(toy(), y2(), cancel^)
    var invalid = matrix[F64](1, 6, [Float64("nan"), 1, 1, 1, 1, 1]).reshape(Shape(6))
    with assert_raises(contains="ValueError: Input sample_weight contains NaN"):
        _ = model.score(toy(), y2(), invalid^)
    var infinite = matrix[F64](1, 6, [Float64("inf"), 1, 1, 1, 1, 1]).reshape(Shape(6))
    with assert_raises(contains="ValueError: Input sample_weight contains infinity"):
        _ = model.score(toy(), y2(), infinite^)
    var signed = matrix[F64](1, 6, [1.0, 1, 1, 1, 1, -1]).reshape(Shape(6))
    var wrong: List[Int] = [1, 1, 1, 2, 2, 1]
    assert_true(abs(model.score(toy(), wrong, signed^) - 1.25) < 1e-15)



def test_float32_outputs_keep_dtype() raises:
    var X = matrix[DType.float32](6, 2, [-2.0, -1, -1, -1, -1, -2, 1, 1, 1, 2, 2, 1])
    var lda = LinearDiscriminantAnalysis[DType.float32](solver="eigen", shrinkage="auto")
    lda.fit(X, y2())
    var proba = lda.predict_proba(X)
    assert_true(proba.dtype == DType.float32)
    var qda = QuadraticDiscriminantAnalysis[DType.float32]()
    qda.fit(matrix[DType.float32](9, 2, [0.0, 0, -2, -2, -2, -1, -1, -1, -1, -2, 1, 3, 1, 2, 2, 1, 2, 2]), y6())
    assert_equal(qda.scalings_[0].shape[0], 2)


def test_shrinkage_representation() raises:
    assert_true(Shrinkage().is_none())
    assert_true(Shrinkage(None).is_none())
    assert_true(Shrinkage("auto").is_auto())
    assert_true(Shrinkage(0).is_zero())
    assert_equal(String(Shrinkage(0.25)), "0.25")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
