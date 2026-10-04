"""Compare weighted KernelRidge.score with the pinned regression metric."""
from std.sys import argv
from std.math import isfinite, isnan
from moxie.kernel_ridge import KernelRidge, ScalarOrVector
from parity_runner import array, numbers


def run_score[dtype: DType](line: String, index: Int) raises:
    var fields = line.split("|")
    var X = array[dtype](String(fields[1]))
    var y = array[dtype](String(fields[2]))
    var model = KernelRidge[dtype](
        Float64(String(fields[3])), kernel="precomputed"
    )
    _ = model.fit(X, y)
    var weight: Optional[ScalarOrVector] = None
    if String(fields[4]) != "none":
        weight = ScalarOrVector(numbers(String(fields[4])))
    var expected = String(fields[5])
    var failed = False
    var actual: Float64 = 0
    try:
        actual = model.score(X, y, weight)
    except e:
        failed = True
        if expected != "error":
            raise Error("score case " + String(index) + ": " + String(e))
    if expected == "error":
        if not failed:
            raise Error("expected score error: " + String(index))
        return
    var reference = Float64(expected)
    if isnan(reference):
        if not isnan(actual):
            raise Error("expected NaN score: " + String(index))
        return
    var tolerance: Float64 = 2e-5 if dtype == DType.float32 else 1e-12
    if not isfinite(actual) or abs(actual - reference) > tolerance * (
        1 + abs(reference)
    ):
        raise Error(
            "score case " + String(index) + " actual=" + String(actual)
            + " expected=" + expected
        )


def main() raises:
    var args = argv()
    var text: String
    with open(args[1], "r") as handle:
        text = handle.read()
    var count = 0
    for line in text.splitlines():
        if line.byte_length() == 0:
            continue
        count += 1
        if line.startswith("32|"):
            run_score[DType.float32](String(line), count)
        else:
            run_score[DType.float64](String(line), count)
    print("Weighted score parity cases passed:", count)
