from moxie.model_selection import LearningCurveDisplay, ValidationCurveDisplay, IndexArray
from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_true, assert_raises

comptime Dense = NDArray[DType.float64]

def main() raises:
    var x = Dense(Shape(3))
    var sizes = IndexArray(Shape(3))
    var train = Dense(Shape(3, 2))
    var test = Dense(Shape(3, 2))
    for i in range(3):
        x[Item(i)] = Float64(i + 1)
        sizes[Item(i)] = Int64(i + 1)
        for j in range(2):
            train[Item(i, j)] = Float64(i + j)
            test[Item(i, j)] = Float64(i + 2 * j)
    var display = LearningCurveDisplay(train_sizes=sizes, train_scores=train, test_scores=test, score_name=String("a<&b"))
    train[Item(0, 0)] = 99
    display.plot()
    assert_equal(len(display.lines_.value()), 2)
    assert_equal(display.lines_.value()[0].mean[Item(0)], Float64(0.5))
    assert_true(display.figure_.value().svg.find("a&lt;&amp;b") >= 0)
    display.figure_.value().savefig("builds/learning.svg")
    for i in range(3):
        for a in range(2):
            var artist = display.lines_.value()[a].copy()
            print("display", a, i, artist.mean[Item(i)], artist.std[Item(i)])
    display.plot(negate_score=True, score_type="test", std_display_style=String("errorbar"))
    assert_true(not display.lines_)
    assert_equal(display.errorbar_.value()[0].mean[Item(1)], Float64(-2))
    display.plot(std_display_style=None)
    assert_true(not display.fill_between_)
    with assert_raises():
        display.plot(score_type="invalid")
    x[Item(2)] = 100
    var validation = ValidationCurveDisplay(param_name="alpha", param_range=x, train_scores=test, test_scores=test)
    validation.plot()
    assert_equal(validation.ax_.value().xscale, "log")
    validation.figure_.value().savefig("builds/validation.svg")
    print("PASS display ownership, moments, styles, SVG and invalid arguments")
