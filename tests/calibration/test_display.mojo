from std.testing import assert_equal, assert_true, assert_raises
import numojo as nm
from moxie.calibration import (
    CalibrationDisplay,
    CalibrationAxes,
    CalibratedClassifierCV,
)
from moxie._calibration_algorithms import Array
from calibration_centroid import Centroid


def values() raises -> Tuple[Array, Array]:
    var y = nm.zeros[DType.float64](nm.Shape(6))
    var p = nm.zeros[DType.float64](nm.Shape(6))
    for i in range(6):
        y[nm.Item(i)] = 1 if i > 2 else 0
        p[nm.Item(i)] = Float64(i) / 5
    return (y^, p^)


def test_plot() raises:
    var input = values()
    var display = CalibrationDisplay.from_predictions(
        input[0], input[1], n_bins=3, name="Centroid", alpha=0.8
    )
    assert_equal(display.estimator_name.value(), "Centroid")
    assert_equal(display.pos_label.value(), 1)
    assert_equal(display.prob_true.size, 3)
    assert_equal(display.line_.value().alpha, 0.8)
    assert_equal(display.line_.value().marker, "s")
    assert_true(display.ax_.value().reference_line)
    assert_equal(
        display.ax_.value().xlabel,
        "Mean predicted probability (Positive class: 1)",
    )
    # Constructed without pos_label, the reference labels end in a space.
    var bare = CalibrationDisplay(display.prob_true, display.prob_pred, input[1])
    bare.plot()
    assert_equal(bare.ax_.value().xlabel, "Mean predicted probability ")
    assert_equal(bare.ax_.value().ylabel, "Fraction of positives ")
    var axes = CalibrationAxes()
    display.plot(ax=axes, name="Before <fit>")
    display.plot(
        ax=axes,
        name="After & fit",
        color="#d97706",
        marker="o",
        linestyle="--",
        ref_line=False,
    )
    assert_equal(len(axes.lines), 2)
    assert_equal(len(axes.legend_labels()), 3)
    assert_equal(display.line_.value().label.value(), "After & fit")
    assert_equal(display.estimator_name.value(), "Centroid")
    display.figure_.value().savefig("builds/calibration-display.svg")
    var svg = axes.to_svg()
    assert_true("Before &lt;fit&gt;" in svg)
    assert_true("After &amp; fit" in svg)
    with assert_raises():
        display.plot(alpha=Float64("nan"))
    with assert_raises():
        display.plot(marker="unsupported")
    with assert_raises():
        display.figure_.value().savefig("builds/not-a-png.png")
    var direct = CalibrationDisplay(
        display.prob_true, display.prob_pred, input[1]
    )
    assert_true(not direct.line_)
    direct.plot(ref_line=False)
    assert_equal(len(direct.ax_.value().legend_labels()), 0)
    var overlay = CalibrationDisplay.from_predictions(
        input[0], input[1], ax=axes, name="Third"
    )
    assert_equal(len(axes.lines), 3)
    assert_equal(len(overlay.ax_.value().lines), 3)
    assert_equal(len(display.figure_.value().axes.lines), 2)
    input[1][nm.Item(0)] = 1
    assert_equal(display.y_prob.item(0), 0)


def test_estimator() raises:
    var X = nm.zeros[DType.float64](nm.Shape(18, 2))
    var y = nm.zeros[DType.float64](nm.Shape(18))
    for i in range(18):
        X[nm.Item(i, 0)] = Float64(i)
        X[nm.Item(i, 1)] = Float64(i % 5)
        y[nm.Item(i)] = -2 if i % 2 == 0 else 3
    var model = CalibratedClassifierCV(Centroid(), cv=3)
    with assert_raises():
        _ = CalibrationDisplay.from_estimator(model, X, y)
    _ = model.fit(X, y)
    var probabilities = model.predict_proba(X)
    var axes = CalibrationAxes()
    var display = CalibrationDisplay.from_estimator(model, X, y, ax=axes)
    assert_equal(display.estimator_name.value(), "CalibratedClassifierCV")
    assert_equal(display.pos_label.value(), 3)
    for i in range(18):
        assert_equal(display.y_prob.item(i), probabilities.item(i, 1))
    var negative = CalibrationDisplay.from_estimator(
        model, X, y, pos_label=Float64(-2)
    )
    for i in range(18):
        assert_equal(negative.y_prob.item(i), probabilities.item(i, 0))
    with assert_raises():
        _ = CalibrationDisplay.from_estimator(model, X, y, pos_label=Float64(5))
    y[nm.Item(0)] = 0.5
    with assert_raises():
        _ = CalibrationDisplay.from_predictions(
            y, display.y_prob, pos_label=Float64(3)
        )


def main() raises:
    test_plot()
    test_estimator()
    print("PASS: display computation, attributes, overlays and SVG contracts")
