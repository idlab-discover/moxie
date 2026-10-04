"""Text fixture protocol for the pinned Python oracle; no Python runtime."""
import numojo as nm
from std.sys import argv
from std.math import isfinite
from moxie._kernel_ridge import (
    pairwise,
    pairwise_callable,
    RowKernel,
    solve_kernel,
    least_squares,
)


@fieldwise_init
struct Asymmetric(RowKernel):
    def evaluate[
        dtype: DType
    ](
        self, X: nm.NDArray[dtype], i: Int, Y: nm.NDArray[dtype], j: Int
    ) raises -> SIMD[dtype, 1]:
        return (
            X.unsafe_get(i * Int(X.strides[0]))
            + 2 * Y.unsafe_get(j * Int(Y.strides[0]))
            + 2
        )


def numbers[dtype: DType](text: String) raises -> List[SIMD[dtype, 1]]:
    var result = List[SIMD[dtype, 1]]()
    for value in text.split(","):
        result.append(SIMD[dtype, 1](Float64(String(value))))
    return result^


def array[dtype: DType](text: String) raises -> nm.NDArray[dtype]:
    var fields = text.split(":")
    var rows = Int(String(fields[0]))
    var cols = Int(String(fields[1]))
    var data = numbers[dtype](String(fields[2]))
    var result = nm.zeros[dtype](nm.Shape(rows, cols))
    for i in range(len(data)):
        result.unsafe_set(i, data[i])
    return result^


def run_case[dtype: DType](line: String, index: Int) raises:
    var f = line.split("|")
    var X = array[dtype](String(f[2]))
    var Y = array[dtype](String(f[3]))
    var result = nm.zeros[dtype](nm.Shape(1, 1))
    var failed = False
    try:
        if f[1] == "solve" or f[1] == "error":
            result = solve_kernel(
                X, Y, numbers[dtype](String(f[4])), numbers[dtype](String(f[5]))
            )
        elif f[1] == "callable":
            result = pairwise_callable(X, Y, Asymmetric(), f[7] == "1")
        elif f[1] == "lstsq":
            result = least_squares(X, Y)
        else:
            var gamma: Optional[Float64] = None
            if String(f[4]) != "none":
                gamma = Float64(String(f[4]))
            result = pairwise(
                X,
                Y,
                String(f[1]),
                gamma,
                Float64(String(f[5])),
                Float64(String(f[6])),
                f[7] == "1",
            )
    except e:
        failed = True
        if String(f[1]) != "error":
            raise Error("case " + String(index) + " " + String(e))
    if f[1] == "error":
        if not failed:
            raise Error(
                "expected factorization failure in case " + String(index)
            )
        return
    var expected = array[dtype](String(f[8]))
    var tolerance = Float64(String(f[9]))
    if result.shape != expected.shape:
        raise Error("shape mismatch")
    for i in range(result.size):
        var a = Float64(result.unsafe_get(i))
        var b = Float64(expected.unsafe_get(i))
        if not isfinite(a) or abs(a - b) > tolerance * (1 + abs(b)):
            raise Error(
                "case "
                + String(index)
                + " "
                + String(f[1])
                + " element "
                + String(i)
                + " actual="
                + String(a)
                + " expected="
                + String(b)
            )


def main() raises:
    var args = argv()
    var data: String
    with open(args[1], "r") as file:
        data = file.read()
    var n = 0
    for line in data.split("\n"):
        if line == "":
            continue
        if line.startswith("32|"):
            run_case[DType.float32](String(line), n)
        else:
            run_case[DType.float64](String(line), n)
        n += 1
    print("PASS parity cases:", n)
