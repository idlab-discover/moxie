"""Class labels through direct and nested classifier pipelines."""
from std.testing import assert_equal, assert_raises
from numojo import NDArray, Shape

from moxie.base import NumericClassifierClasses, clone
from moxie.dummy import DummyRegressor
from moxie.linear_model import LogisticRegression
from moxie.pipeline import Pipeline, make_pipeline
from moxie.preprocessing import StandardScaler
from parity_io import ResultLine


def ordered_classes[E: NumericClassifierClasses](
    estimator: E,
) raises -> E.Target:
    comptime assert E.ClassLabelDType == DType.int64
    return estimator.get_classes()


def main() raises:
    comptime assert not conforms_to(
        Pipeline[StandardScaler[], DummyRegressor[]], NumericClassifierClasses
    )
    var X = NDArray[DType.float64](Shape(6, 1))
    var y = NDArray[DType.int64](Shape(6))
    for i in range(6):
        X.unsafe_set(i, Float64(i))
        y.unsafe_set(i, Int64(7 if i < 3 else -2))

    var pipe = make_pipeline(
        StandardScaler(), make_pipeline(StandardScaler(), LogisticRegression()),
    )
    with assert_raises(contains="NotFittedError"):
        _ = pipe.classes_()
    _ = pipe.fit(X, y)
    var classes = pipe.classes_()
    assert_equal(classes.unsafe_get(0), -2)
    assert_equal(classes.unsafe_get(1), 7)
    classes.unsafe_set(0, 0)

    var result = ResultLine("logistic")
    result.add("classes", pipe.estimator.classes_())
    result.add("nested_classes", ordered_classes(pipe))
    result.add("probabilities", pipe.predict_proba(X))

    var fresh = clone(pipe)
    with assert_raises(contains="NotFittedError"):
        _ = ordered_classes(fresh)
    _ = fresh.fit(X, y + 10)
    result.add("clone_classes", fresh.classes_())
    result.add("original_classes", pipe.classes_())
    _ = pipe.fit(X, y + 20)
    result.add("refit_classes", pipe.classes_())
    result.emit()
