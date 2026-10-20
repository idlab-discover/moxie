"""Ridge regression with several solvers, a RidgeClassifier and ridge_regression."""
import numojo as nm
from numojo import Shape
from moxie.linear_model import Ridge, RidgeClassifier, ridge_regression


def main() raises:
    var X = nm.zeros[DType.float64](Shape(6, 2))
    var values: List[Float64] = [0, 1, 1, 0.5, 2, 2, 3, 1, 4, 3, 5, 2.5]
    for i in range(12):
        X.unsafe_set(i, values[i])
    var y = nm.zeros[DType.float64](Shape(6))
    for i in range(6):
        y.unsafe_set(i, 2 * values[2 * i] + values[2 * i + 1] + 1)
    var solvers: List[String] = ["auto", "svd", "lsqr", "sparse_cg"]
    for solver in solvers:
        var model = Ridge(alpha=0.1, solver=solver)
        model.fit(X, y)
        print(solver, "->", model.solver_, "coef_:", model.coef_.unsafe_get(0), model.coef_.unsafe_get(1), "intercept_:", model.intercept_.unsafe_get(0))
    var positive = Ridge(positive=True)
    positive.fit(X, y)
    print("positive (lbfgs) coef_:", positive.coef_.unsafe_get(0), positive.coef_.unsafe_get(1))
    var labels = nm.zeros[DType.int64](Shape(6))
    for i in range(3, 6):
        labels.unsafe_set(i, 1)
    var classifier = RidgeClassifier()
    classifier.fit(X, labels)
    print("RidgeClassifier accuracy:", classifier.score(X, labels))
    var result = ridge_regression(X, y, 1.0, solver="lsqr", return_n_iter=True)
    print("ridge_regression coef:", result.coef.unsafe_get(0), "lsqr iterations:", result.n_iter.value()[0])
