"""Native Ridge / RidgeClassifier / ridge_regression contracts."""
import numojo as nm
from numojo import NDArray, Shape
from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal
from moxie.base import clone
from moxie.linear_model import Ridge, RidgeClassifier, ridge_regression, ClassWeight, RidgeParams
from moxie.linear_model._ridge import ScalarOrVector


def matrix[dtype: DType](rows: Int, cols: Int, values: List[Float64]) raises -> NDArray[dtype]:
    var a = nm.zeros[dtype](Shape(rows, cols))
    for i in range(rows * cols):
        a.unsafe_set(i, values[i].cast[dtype]())
    return a^


def vector[dtype: DType](values: List[Float64]) raises -> NDArray[dtype]:
    var a = nm.zeros[dtype](Shape(len(values)))
    for i in range(len(values)):
        a.unsafe_set(i, values[i].cast[dtype]())
    return a^


def data[dtype: DType]() raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    var X = matrix[dtype](6, 2, [0.0, 1.0, 1.0, 0.5, 2.0, 2.0, 3.0, 1.0, 4.0, 3.0, 5.0, 2.5])
    var y = vector[dtype]([1.0, 2.5, 5.0, 5.5, 8.5, 9.0])
    return (X^, y^)


def solvers_agree[dtype: DType]() raises:
    var d = data[dtype]()
    var reference = Ridge[dtype](alpha=0.5, solver="cholesky")
    reference.fit(d[0], d[1])
    assert_equal(reference.solver_, "cholesky")
    assert_false(Bool(reference.n_iter_))
    var names: List[String] = ["svd", "lsqr", "sparse_cg", "auto"]
    for name in names:
        var model = Ridge[dtype](alpha=0.5, solver=name, tol=1e-10)
        model.fit(d[0], d[1])
        for j in range(2):
            assert_almost_equal(Float64(model.coef_.unsafe_get(j)), Float64(reference.coef_.unsafe_get(j)), atol=1e-3)
        assert_almost_equal(Float64(model.intercept_.unsafe_get(0)), Float64(reference.intercept_.unsafe_get(0)), atol=1e-3)
        if name == "lsqr":
            assert_true(model.n_iter_.value()[0] >= 1)
    var scalar = Ridge[dtype](alpha=0.5)
    scalar.fit(d[0], d[1], sample_weight=2.0)
    var repeated = Ridge[dtype](alpha=0.5)
    repeated.fit(d[0], d[1], sample_weight=vector[dtype]([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]))
    assert_almost_equal(Float64(scalar.coef_.unsafe_get(0)), Float64(repeated.coef_.unsafe_get(0)), atol=1e-6)
    # alpha -> 0 approaches least squares; large alpha shrinks toward zero.
    var big = Ridge[dtype](alpha=1e6)
    big.fit(d[0], d[1])
    assert_true(abs(Float64(big.coef_.unsafe_get(0))) < 1e-3)
    # Positive (L-BFGS-B): coefficients are nonnegative.
    var negative = vector[dtype]([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    var positive = Ridge[dtype](positive=True)
    positive.fit(d[0], negative)
    assert_equal(positive.solver_, "lbfgs")
    for j in range(2):
        assert_true(Float64(positive.coef_.unsafe_get(j)) >= 0.0)


def shapes_and_alphas[dtype: DType]() raises:
    var d = data[dtype]()
    var model = Ridge[dtype]()
    model.fit(d[0], d[1])
    assert_equal(model.coef_.ndim, 1)
    assert_equal(model.predict(d[0]).ndim, 1)
    assert_true(model.score(d[0], d[1]) > 0.9)
    # One-column target is raveled like sklearn (coef_ (p,), predict (n,)).
    var Y1 = matrix[dtype](6, 1, [1.0, 2.5, 5.0, 5.5, 8.5, 9.0])
    model.fit(d[0], Y1)
    assert_equal(model.coef_.ndim, 1)
    assert_equal(model.predict(d[0]).ndim, 1)
    # Per-target alphas.
    var Y = matrix[dtype](6, 2, [1.0, 0.0, 2.5, 1.0, 5.0, 2.0, 5.5, 3.0, 8.5, 4.0, 9.0, 5.0])
    var two: List[Float64] = [0.1, 100.0]
    var per = Ridge[dtype](ScalarOrVector(two^))
    per.fit(d[0], Y)
    assert_equal(per.coef_.shape[0], 2)
    assert_equal(per.predict(d[0]).shape[1], 2)
    var three: List[Float64] = [0.1, 1.0, 2.0]
    var wrong = Ridge[dtype](ScalarOrVector(three^))
    with assert_raises(contains="penalties"):
        wrong.fit(d[0], Y)
    # Weighted fit equals repeating a sample.
    var w = Ridge[dtype]()
    w.fit(d[0], d[1], sample_weight=vector[dtype]([2.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    var r = Ridge[dtype]()
    r.fit(matrix[dtype](7, 2, [0.0, 1.0, 0.0, 1.0, 1.0, 0.5, 2.0, 2.0, 3.0, 1.0, 4.0, 3.0, 5.0, 2.5]), vector[dtype]([1.0, 1.0, 2.5, 5.0, 5.5, 8.5, 9.0]))
    assert_almost_equal(Float64(w.coef_.unsafe_get(0)), Float64(r.coef_.unsafe_get(0)), atol=1e-4)
    # Wide data takes the kernel (dual) Cholesky path.
    var wide = Ridge[dtype]()
    wide.fit(matrix[dtype](2, 3, [1.0, 0.0, 2.0, 0.0, 1.0, 1.0]), vector[dtype]([1.0, 2.0]))
    assert_equal(wide.coef_.shape[0], 3)


def errors_and_lifecycle() raises:
    var d = data[DType.float64]()
    var fresh = Ridge()
    with assert_raises(contains="NotFittedError"):
        _ = fresh.predict(d[0])
    with assert_raises(contains="solver"):
        var bad = Ridge(solver="newton")
        bad.fit(d[0], d[1])
    with assert_raises(contains="positive"):
        var bad = Ridge(solver="svd", positive=True)
        bad.fit(d[0], d[1])
    with assert_raises(contains="lbfgs"):
        var bad = Ridge(solver="lbfgs")
        bad.fit(d[0], d[1])
    with assert_raises(contains="NotImplementedError"):
        var bad = Ridge(solver="sag")
        bad.fit(d[0], d[1])
    with assert_raises(contains="alpha"):
        var bad = Ridge(-1.0)
        bad.fit(d[0], d[1])
    with assert_raises(contains="max_iter"):
        var bad = Ridge(max_iter=Optional(0))
        bad.fit(d[0], d[1])
    var model = Ridge(2.0, fit_intercept=False, solver="svd")
    model.fit(d[0], d[1])
    assert_equal(model.intercept_.unsafe_get(0), 0.0)
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    assert_equal(copy.get_params().solver, "svd")
    copy.set_params(solver=Optional(String("lsqr")))
    assert_equal(model.solver, "svd")
    with assert_raises(contains="features"):
        _ = model.predict(matrix[DType.float64](1, 3, [1.0, 2.0, 3.0]))
    # ridge_regression: no intercept; lsqr reports iterations only on request.
    var result = ridge_regression(d[0], d[1], 1.0, solver="lsqr", return_n_iter=True)
    assert_true(result.n_iter.value()[0] >= 1)
    var silent = ridge_regression(d[0], d[1], 1.0, solver="lsqr")
    assert_false(Bool(silent.n_iter))
    with assert_raises(contains="sag"):
        _ = ridge_regression(d[0], d[1], 1.0, return_intercept=True, solver="svd")


def labels_of(values: List[Int]) raises -> NDArray[DType.int64]:
    var a = nm.zeros[DType.int64](Shape(len(values)))
    for i in range(len(values)):
        a.unsafe_set(i, Int64(values[i]))
    return a^


def classifier() raises:
    var X = matrix[DType.float64](6, 2, [0.0, 0.0, 0.1, 0.2, 0.2, 0.1, 3.0, 3.0, 3.1, 2.9, 2.9, 3.2])
    var y = labels_of([7, 7, 7, -2, -2, -2])
    var model = RidgeClassifier()
    model.fit(X, y)
    assert_equal(model.classes_.unsafe_get(0), -2)
    assert_equal(model.classes_.unsafe_get(1), 7)
    assert_equal(model.coef_.ndim, 1)  # binary: (features,)
    assert_equal(model.decision_function(X).ndim, 1)
    assert_equal(model.score(X, y), 1.0)
    assert_equal(model.predict(X).unsafe_get(0), 7)
    var column = y.reshape(Shape(6, 1))
    var column_model = RidgeClassifier()
    column_model.fit(X, column)
    assert_equal(column_model.predict(X).ndim, 1)
    assert_equal(column_model.score(X, column), 1.0)
    var scalar_model = RidgeClassifier()
    scalar_model.fit(X, column, sample_weight=2.0)
    var array_model = RidgeClassifier()
    array_model.fit(X, y, sample_weight=vector[DType.float64]([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]))
    assert_almost_equal(scalar_model.coef_.unsafe_get(0), array_model.coef_.unsafe_get(0), atol=1e-12)
    with assert_raises():
        _ = model.score(X, y, Optional(vector[DType.float64]([1.0])))
    with assert_raises():
        _ = model.score(X, y, Optional(vector[DType.float64]([1.0, 1.0, 1.0, 1.0, 1.0, Float64("nan")])))
    with assert_raises():
        model.fit(X, nm.zeros[DType.int64](Shape(6, 0)))
    assert_equal(model.score(X, y), 1.0)  # failed refit preserves state
    # Multiclass: one column per sorted class.
    var X3 = matrix[DType.float64](6, 2, [0.0, 0.0, 0.1, 0.0, 5.0, 0.0, 5.1, 0.0, 0.0, 5.0, 0.0, 5.1])
    var y3 = labels_of([1, 1, 2, 2, 3, 3])
    var multi = RidgeClassifier(0.1)
    multi.fit(X3, y3)
    assert_equal(multi.coef_.shape[0], 3)
    assert_equal(multi.decision_function(X3).shape[1], 3)
    assert_equal(multi.score(X3, y3), 1.0)
    # class_weight variants and multilabel indicators.
    var balanced = RidgeClassifier(class_weight=ClassWeight.balanced_weights())
    balanced.fit(X3, y3)
    var explicit = Dict[Int, Float64]()
    explicit[1] = 3.0
    var weighted = RidgeClassifier(class_weight=ClassWeight.explicit(explicit^))
    weighted.fit(X3, y3)
    var unknown = Dict[Int, Float64]()
    unknown[9] = 1.0
    var bad = RidgeClassifier(class_weight=ClassWeight.explicit(unknown^))
    with assert_raises(contains="class_weight"):
        bad.fit(X3, y3)
    var indicator = nm.zeros[DType.int64](Shape(6, 2))
    for i in range(6):
        indicator.unsafe_set(i * 2, Int64(1 if i < 4 else 0))
        indicator.unsafe_set(i * 2 + 1, Int64(1 if i >= 2 else 0))
    var ml = RidgeClassifier(0.01)
    ml.fit(X3, indicator)
    assert_equal(ml.predict(X3).ndim, 2)
    assert_true(ml.score(X3, indicator) > 0.3)
    var badml = nm.zeros[DType.int64](Shape(6, 2))
    badml.unsafe_set(0, 5)
    with assert_raises(contains="indicator"):
        ml.fit(X3, badml)
    with assert_raises(contains="NotFittedError"):
        _ = RidgeClassifier().predict(X3)
    assert_false(clone(multi).__sklearn_is_fitted__())


def main() raises:
    solvers_agree[DType.float32]()
    solvers_agree[DType.float64]()
    shapes_and_alphas[DType.float32]()
    shapes_and_alphas[DType.float64]()
    errors_and_lifecycle()
    classifier()
    print("Ridge contracts passed: solver agreement, positive, shapes, per-target alpha, weights, kernel path, errors, lifecycle, classifier")
