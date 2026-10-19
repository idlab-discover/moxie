from evaluation import Adapter, Scorer, Dense
from moxie.dummy import DummyRegressor
from moxie.model_selection import LearningCurveDisplay, ValidationCurveDisplay, KFold, IndexSplit, IndexArray
from numojo import Shape, Item, zeros
from std.testing import assert_equal, assert_true

def main() raises:
    var X = zeros[DType.float64](Shape(8, 2))
    var y = Dense(Shape(8))
    for i in range(8):
        y[Item(i)] = Float64(i)
    var cursor = KFold(4).split(8)
    var cv = List[IndexSplit]()
    while True:
        var split = cursor.next()
        if not split:
            break
        cv.append(split.value().copy())
    var sizes = IndexArray(Shape(2))
    sizes[Item(0)] = 2
    sizes[Item(1)] = 6
    var estimator = DummyRegressor[]()
    var learning = LearningCurveDisplay.from_estimator[Adapter, DType.int64](estimator, X, y, Adapter(), Scorer(False), cv, None, None, train_sizes=sizes.copy())
    assert_equal(learning.train_scores.shape[0], 2)
    assert_true(learning.figure_)
    var x = Dense(Shape(2))
    x[Item(0)] = 1
    x[Item(1)] = 2
    var validation = ValidationCurveDisplay.from_estimator[Adapter](estimator, X, y, Adapter(), Scorer(False), cv, None, None, param_name="strategy", param_range=List[String](["mean", "median"]), x_values=x)
    assert_equal(validation.test_scores.shape[1], 4)
    assert_true(validation.figure_)
    assert_true(not estimator.__sklearn_is_fitted__())
    print("PASS display factories with real DummyRegressor and original isolation")
