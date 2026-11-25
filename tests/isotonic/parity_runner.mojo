"""Read oracle fixtures produced by check_parity.py; no Python runtime."""
from std.sys import argv
from std.testing import assert_equal, assert_raises
from numojo import array
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler
from std.math import isfinite, isnan
from moxie.isotonic import (
    IsotonicRegression,
    Increasing,
    isotonic_regression,
    check_increasing,
)


def vector[dtype: DType](text: String) raises -> List[Scalar[dtype]]:
    var result = List[Scalar[dtype]]()
    if text == "-":
        return result^
    for part in text.split(","):
        result.append(Scalar[dtype](Float64(String(part))))
    return result^


def optional_number(text: String) raises -> Optional[Float64]:
    if text == "-":
        return None
    return Float64(text)


def optional_vector[dtype: DType](text: String) raises -> Optional[List[Scalar[dtype]]]:
    if text == "-":
        return None
    return vector[dtype](text)


def compare[dtype: DType](
    actual: List[Scalar[dtype]],
    expected: List[Scalar[dtype]],
    context: String,
    tolerance: Float64 = 2e-6 if dtype == DType.float32 else 2e-12,
) raises:
    if len(actual) != len(expected):
        raise Error(
            context
            + ": shape mismatch "
            + String(len(actual))
            + " != "
            + String(len(expected))
        )
    for i in range(len(actual)):
        if isnan(expected[i]) and isnan(actual[i]):
            continue
        if actual[i] == expected[i]:
            continue
        if (
            not isfinite(expected[i])
            or not isfinite(actual[i])
            or abs(Float64(actual[i]) - Float64(expected[i])) > tolerance * (1 + abs(Float64(expected[i])))
        ):
            raise Error(
                context
                + ": index "
                + String(i)
                + " actual="
                + String(actual[i])
                + " expected="
                + String(expected[i])
            )


def run_cases[dtype: DType]() raises:
    # The comparator must reject finite/NaN mismatches in either direction.
    with assert_raises():
        compare([1.0], [Float64("nan")], "comparator")
    with assert_raises():
        compare([Float64("nan")], [1.0], "comparator")
    with assert_raises():
        compare([1.0], [Float64("inf")], "comparator")
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
        var context = kind + " case " + String(count)
        if kind == "kernel":
            var y = vector[dtype](String(fields[1]))
            var result = isotonic_regression(
                y,
                sample_weight=optional_vector[dtype](String(fields[2])),
                y_min=optional_number(String(fields[3])),
                y_max=optional_number(String(fields[4])),
                increasing=fields[5] == "true",
            )
            compare(result, vector[dtype](String(fields[6])), context)
        elif kind == "direction":
            var result = check_increasing(
                vector[dtype](String(fields[1])), vector[dtype](String(fields[2]))
            )
            if result != (fields[3] == "true"):
                raise Error(context + ": direction mismatch")
        elif kind == "pipeline":
            var x = vector[dtype](String(fields[1]))
            var y = vector[dtype](String(fields[2]))
            var query = vector[dtype](String(fields[3]))
            var pipe = make_pipeline(
                StandardScaler[dtype](),
                IsotonicRegression[dtype](out_of_bounds="clip"),
            )
            _ = pipe.fit(array(x, [len(x), 1]), array(y, [len(y)]))
            var predictions = pipe.predict(array(query, [len(query), 1]))
            assert_equal(predictions.ndim, 1)
            compare(predictions.tolist(), vector[dtype](String(fields[4])), context)
        else:
            var X = vector[dtype](String(fields[1]))
            var y = vector[dtype](String(fields[2]))
            var direction = Increasing(fields[6] == "true")
            if fields[6] == "auto":
                direction = Increasing(String("auto"))
            var model = IsotonicRegression[dtype](
                y_min=optional_number(String(fields[4])),
                y_max=optional_number(String(fields[5])),
                increasing=direction^,
                out_of_bounds=String(fields[7]),
            )
            var x_array = array(X, [len(X), 1])
            var target = array(y, [len(y)])
            if fields[3] == "-":
                _ = model.fit(x_array, target)
            else:
                var weights = vector[dtype](String(fields[3]))
                _ = model.fit(x_array, target, sample_weight=array(weights, [len(weights)]))
            compare(
                model.X_thresholds_,
                vector[dtype](String(fields[8])),
                context + " X thresholds",
                0.0,
            )
            compare(
                model.y_thresholds_,
                vector[dtype](String(fields[9])),
                context + " y thresholds",
            )
            var query = vector[dtype](String(fields[10]))
            var query_array = array(query, [len(query)])
            var predictions = model.predict(query_array)
            assert_equal(predictions.ndim, 1)
            compare(
                predictions.tolist(),
                vector[dtype](String(fields[11])),
                context + " predict",
            )
            compare(
                model.transform(query),
                vector[dtype](String(fields[11])),
                context + " transform",
            )
            if model.increasing_ != (fields[12] == "true"):
                raise Error(context + ": fitted direction mismatch")
            compare(
                [model.X_min_, model.X_max_],
                vector[dtype](String(fields[13])),
                context + " domain",
                0.0,
            )
            if kind == "score":
                var score_target = vector[dtype](String(fields[14]))
                var score_weights = vector[dtype](String(fields[15]))
                var score = model.score(
                    query_array,
                    array(score_target, [len(score_target)]),
                    array(score_weights, [len(score_weights)]),
                )
                compare([score], vector[DType.float64](String(fields[16])), context + " R2", 2e-6 if dtype == DType.float32 else 2e-12)
        count += 1
    print("Reference parity checks passed:", count, "cases; dtype", dtype)


def main() raises:
    var args = argv()
    if len(args) > 2 and args[2] == "float32":
        run_cases[DType.float32]()
    else:
        run_cases[DType.float64]()
