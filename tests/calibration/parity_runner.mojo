"""Execute externally generated oracle cases in a native process."""
from std.sys import argv
from std.math import isfinite
import numojo as nm
from moxie.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
    CalibrationCV,
    CalibrationSplit,
    CalibrationDisplay,
)
from moxie._calibration_algorithms import (
    Array,
    _SigmoidCalibration,
    _IsotonicCalibration,
    _TemperatureScaling,
)
from calibration_centroid import Centroid


def array(text: String) raises -> Array:
    var fields = text.split(":")
    var n = Int(String(fields[0]))
    var c = Int(String(fields[1]))
    var values = fields[2].split(",")
    if c == 0:
        var out = nm.zeros[DType.float64](nm.Shape(n))
        for i in range(n):
            out[nm.Item(i)] = Float64(String(values[i]))
        return out^
    var out = nm.zeros[DType.float64](nm.Shape(n, c))
    for i in range(n):
        for j in range(c):
            out[nm.Item(i, j)] = Float64(String(values[i * c + j]))
    return out^


def compare(
    actual: Array, expected: Array, tolerance: Float64, context: String
) raises:
    if actual.ndim != expected.ndim or actual.shape[0] != expected.shape[0]:
        raise Error(context + ": shape mismatch")
    var columns = actual.shape[1] if actual.ndim == 2 else 1
    if actual.ndim == 2 and actual.shape[1] != expected.shape[1]:
        raise Error(context + ": column mismatch")
    for i in range(actual.shape[0]):
        for j in range(columns):
            var a = actual.item(i, j) if actual.ndim == 2 else actual.item(i)
            var b = expected.item(
                i, j
            ) if expected.ndim == 2 else expected.item(i)
            if (
                not isfinite(a)
                or not isfinite(b)
                or abs(a - b) > tolerance * (1 + abs(b))
            ):
                raise Error(
                    context + ": actual=" + String(a) + " expected=" + String(b)
                )


def main() raises:
    var args = argv()
    var data: String
    with open(args[1], "r") as f:
        data = f.read()
    var count = 0
    for line in data.split("\n"):
        if line == "":
            continue
        var fields = line.split("|")
        var kind = String(fields[0])
        var x = array(String(fields[1]))
        var y = array(String(fields[2]))
        var w: Optional[Array] = None
        if String(fields[3]) != "-":
            w = array(String(fields[3]))
        var q = array(String(fields[4]))
        var expected = array(String(fields[5]))
        var context = kind + " case " + String(count)
        if kind == "sigmoid":
            var model = _SigmoidCalibration()
            _ = model.fit(x, y, w)
            compare(model.predict(q), expected, 5e-6, context)
        elif kind == "isotonic":
            var model = _IsotonicCalibration()
            _ = model.fit(x, y, w)
            compare(model.predict(q), expected, 2e-12, context)
        elif kind == "temperature":
            var model = _TemperatureScaling()
            _ = model.fit(x, y, w)
            compare(model.predict(q), expected, 5e-6, context)
        elif kind == "curve":
            var result = calibration_curve(
                y,
                x,
                n_bins=Int(String(fields[6])),
                strategy=String(fields[7]),
                pos_label=Float64(String(fields[8])),
            )
            compare(result[0], expected, 2e-12, context)
            compare(result[1], q, 2e-12, context)
        elif kind == "curve_float32":
            var y32 = y.astype[DType.float32]()
            var x32 = x.astype[DType.float32]()
            var result = calibration_curve(
                y32,
                x32,
                n_bins=Int(String(fields[6])),
                strategy=String(fields[7]),
            )
            compare(result[0], expected, 2e-12, context)
            compare(result[1], q, 2e-12, context)
        elif kind == "curve_string":
            var labels = List[String]()
            for i in range(y.size):
                labels.append("cat" if y.item(i) == -3 else "dog")
            var result = calibration_curve(
                labels,
                x,
                n_bins=Int(String(fields[6])),
                strategy=String(fields[7]),
                pos_label=String(fields[8]),
            )
            compare(result[0], expected, 2e-12, context)
            compare(result[1], q, 2e-12, context)
        elif kind == "display" or kind == "display_estimator":
            var pos: Optional[Float64] = None
            if String(fields[8]) != "-":
                pos = Float64(String(fields[8]))
            var display: CalibrationDisplay
            if kind == "display":
                display = CalibrationDisplay.from_predictions(
                    y,
                    x,
                    n_bins=Int(String(fields[6])),
                    strategy=String(fields[7]),
                    pos_label=pos,
                    name=String(fields[9]),
                    alpha=0.8,
                )
            else:
                var estimator = Centroid()
                estimator.fit(x, y)
                display = CalibrationDisplay.from_estimator(
                    estimator,
                    x,
                    y,
                    n_bins=Int(String(fields[6])),
                    strategy=String(fields[7]),
                    pos_label=pos,
                    alpha=0.8,
                )
            compare(display.prob_true, expected, 2e-12, context)
            compare(display.prob_pred, q, 2e-12, context)
            compare(display.y_prob, array(String(fields[10])), 2e-12, context)
            if display.ax_.value().xlabel != String(
                fields[11]
            ) or display.ax_.value().ylabel != String(fields[12]):
                raise Error(context + ": axis labels differ")
            if (
                display.estimator_name.value() != String(fields[9])
                or display.line_.value().alpha != 0.8
            ):
                raise Error(context + ": display metadata differs")
        elif kind == "cv" or kind == "cv_splits" or kind == "cv_default":
            var cv: CalibrationCV = 3
            if kind == "cv_default":
                cv = None
            elif kind == "cv_splits":
                var splits = List[CalibrationSplit]()
                for split in fields[11].split(";"):
                    var parts = split.split(":")
                    var train = List[Int]()
                    var test = List[Int]()
                    for index in parts[0].split(","):
                        train.append(Int(String(index)))
                    for index in parts[1].split(","):
                        test.append(Int(String(index)))
                    splits.append(CalibrationSplit(train^, test^))
                cv = splits^
            var estimator = Centroid(String(fields[8]) == "true")
            if String(fields[9]) == "true":
                estimator.fit(x, y, w)
                estimator.frozen = True
            var model = CalibratedClassifierCV(
                estimator^,
                cv=cv^,
                method=String(fields[6]),
                ensemble=String(fields[7]) == "true",
            )
            _ = model.fit(x, y, w)
            compare(model.predict_proba(q), expected, 5e-6, context)
            compare(model.predict(q), array(String(fields[10])), 0, context)
        else:
            raise Error("unknown parity case")
        count += 1
    print("PASS:", count, "scikit-learn parity cases")
