from std.testing import assert_equal, assert_raises
import numojo as nm
from moxie.calibration import calibration_curve


def main() raises:
    var y = nm.zeros[DType.float32](nm.Shape(4))
    var p = nm.zeros[DType.float32](nm.Shape(4))
    y[nm.Item(2)] = 1
    y[nm.Item(3)] = 1
    p[nm.Item(1)] = 0.25
    p[nm.Item(2)] = 0.75
    p[nm.Item(3)] = 1
    var result = calibration_curve(y, p, n_bins=2)
    assert_equal(result[0].item(0), 0)
    assert_equal(result[0].item(1), 1)
    assert_equal(result[1].item(0), 0.125)
    assert_equal(result[1].item(1), 0.875)
    var labels: List[String] = ["no", "no", "yes", "yes"]
    var string_result = calibration_curve(
        labels, p, n_bins=2, pos_label=String("yes")
    )
    assert_equal(string_result[0].item(0), 0)
    assert_equal(string_result[0].item(1), 1)
    with assert_raises():
        _ = calibration_curve(labels, p)
    labels[0] = "third"
    with assert_raises():
        _ = calibration_curve(labels, p, pos_label=String("yes"))
    print("PASS: Float32 and string-label calibration curves")
