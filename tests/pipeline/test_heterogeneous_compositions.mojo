"""Check held-out predictions, probability columns, and refits for typed finals."""
from numojo import Item, NDArray, Shape, zeros
from std.testing import assert_equal, assert_false, assert_raises

from moxie.base import ClassifierMixin, clone
from moxie.naive_bayes import GaussianNB, NumericDomain
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler
from moxie.tree import DecisionTreeClassifier
from parity_io import ResultLine

comptime Labels = NDArray[DType.int64]


def dataset() raises -> Tuple[NDArray[DType.float64], Labels, NDArray[DType.float64], Labels]:
    var X = NDArray[DType.float64](Shape(12, 2))
    var y = Labels(Shape(12))
    for i in range(12):
        var group = i % 2
        X[Item(i, 0)] = Float64(i // 2 + group * 2)
        X[Item(i, 1)] = Float64(10 + (i * 7) % 19 + group * 3)
        y[Item(i)] = 7 if group == 0 else -2
    var query = NDArray[DType.float64](Shape(4, 2))
    var expected = Labels(Shape(4))
    for i in range(4):
        query[Item(i, 0)] = Float64(i * 2) + 0.5
        query[Item(i, 1)] = Float64(12 + (i * 5) % 11)
        expected[Item(i)] = 7 if i % 2 == 0 else -2
    return (X^, y^, query^, expected^)


def check_unfitted[E: ClassifierMixin & Deinitable](
    estimator: E, X: E.Input,
) raises where conforms_to(E.Prediction, Deinitable):
    assert_false(estimator.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = estimator.predict(X)


def check_tree(
    X: NDArray[DType.float32], y: Labels, query: NDArray[DType.float32], expected: Labels,
) raises:
    var pipe = make_pipeline(
        StandardScaler[DType.float32](),
        DecisionTreeClassifier[DType.int64](random_state=0, max_depth=2),
    )
    check_unfitted(pipe, query)
    _ = pipe.fit(X, y)
    with assert_raises(contains="ValueError"):
        _ = pipe.predict(zeros[DType.float32](Shape(1, 1)))
    var fresh = clone(pipe)
    check_unfitted(fresh, query)
    assert_equal(fresh.get_params().estimator.max_depth.value(), 2)

    var result = ResultLine("tree")
    var probabilities = pipe.predict_proba(query)
    assert_equal(len(probabilities), 1)
    assert_equal(len(pipe.estimator.classes_), 1)
    result.add("classes", pipe.estimator.classes_[0])
    result.add("prediction", pipe.predict(query))
    result.add("probability", probabilities[0])
    result.scalar("score", pipe.score(query, expected))

    var params = pipe.get_params()
    params.estimator.max_depth = 1
    _ = pipe.set_params(params).fit(X, y + 10)
    result.add("refit_classes", pipe.estimator.classes_[0])
    result.add("refit_prediction", pipe.predict(query))
    result.add("refit_probability", pipe.predict_proba(query)[0])
    result.emit()


def check_gaussian(
    X: NDArray[DType.float64], y: Labels, query: NDArray[DType.float64], expected: Labels,
) raises:
    var pipe = make_pipeline(StandardScaler(), GaussianNB[NumericDomain[DType.int64]]())
    check_unfitted(pipe, query)
    _ = pipe.fit(X, y)
    with assert_raises(contains="ValueError"):
        _ = pipe.predict(zeros[DType.float64](Shape(1, 1)))
    var fresh = clone(pipe)
    check_unfitted(fresh, query)
    assert_equal(fresh.get_params().estimator.var_smoothing, 1e-9)

    var result = ResultLine("gaussian")
    result.add("classes", pipe.estimator.classes_)
    result.add("prediction", pipe.predict(query))
    result.add("probability", pipe.predict_proba(query))
    result.scalar("score", pipe.score(query, expected))

    var params = pipe.get_params()
    params.estimator.var_smoothing = 0.05
    _ = pipe.set_params(params).fit(X, y + 10)
    result.add("refit_classes", pipe.estimator.classes_)
    result.add("refit_prediction", pipe.predict(query))
    result.add("refit_probability", pipe.predict_proba(query))
    result.emit()


def main() raises:
    var data = dataset()
    ref X = data[0]
    ref y = data[1]
    ref query = data[2]
    ref expected = data[3]
    check_tree(X.astype[DType.float32](), y, query.astype[DType.float32](), expected)
    check_gaussian(X, y, query, expected)
    print("PASS: dense tree and GaussianNB pipeline contracts")
