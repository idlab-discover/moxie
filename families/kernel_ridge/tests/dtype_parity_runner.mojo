"""Assert dtype arms, shapes, and values against the uncast sklearn oracle."""
import numojo as nm
from std.sys import argv
from std.math import isfinite
from moxie.kernel_ridge import (
    KernelRidge,
    RowKernel,
    FloatingArray,
    ScalarOrVector,
)
from parity_runner import array, numbers, compare


@fieldwise_init
struct Dot(Copyable, RowKernel):
    def evaluate[
        dtype: DType
    ](
        self, X: nm.NDArray[dtype], i: Int, Y: nm.NDArray[dtype], j: Int
    ) raises -> SIMD[dtype, 1]:
        var value: SIMD[dtype, 1] = 0
        for k in range(X.shape[1]):
            value += X.unsafe_get(
                i * Int(X.strides[0]) + k * Int(X.strides[1])
            ) * Y.unsafe_get(j * Int(Y.strides[0]) + k * Int(Y.strides[1]))
        return value


def check_dtype(actual: FloatingArray, expected: String, label: String) raises:
    if actual.isa[nm.NDArray[DType.float32]]() != (expected == "float32"):
        raise Error(label + " dtype mismatch; expected " + expected)


def run_case[xd: DType, yd: DType, zd: DType](line: String, index: Int) raises:
    var f = line.split("|")
    var X = array[xd](String(f[6]))
    var y = array[yd](String(f[7]))
    var Z = array[zd](String(f[8]))
    var metric: Optional[Dot] = None
    var kernel = String(f[0])
    if kernel == "callable":
        kernel = "linear"
        metric = Dot()
    # Default estimator parameter deliberately differs from some feature dtypes.
    var model = KernelRidge[DType.float64, Dot](
        ScalarOrVector(numbers(String(f[5]))),
        kernel=kernel,
        gamma=0.2,
        native_kernel=metric^,
    )
    var label = "dtype case " + String(index) + " " + String(f[0])
    var failed = False
    try:
        if f[4] == "none":
            _ = model.fit(X, y)
        elif f[4] == "list":
            _ = model.fit(
                X,
                y,
                sample_weight=ScalarOrVector(List[Float64]([0.5, 1.0, 2.0])),
            )
        else:
            _ = model.fit(X, y, sample_weight=Float64(String(f[4])))
    except e:
        failed = True
        if String(f[9]) != "error":
            raise Error(label + " " + String(e))
    if f[9] == "error":
        if not failed:
            raise Error(label + " expected error")
        return
    if not model.X_fit_.isa[nm.NDArray[xd]]():
        raise Error(label + " X_fit_ dtype mismatch")
    var predicted = model.predict(Z)
    check_dtype(model.dual_coef_, String(f[9]), label + " coefficients")
    check_dtype(predicted, String(f[10]), label + " predictions")
    var tol = Float64(String(f[13]))
    var score_y = y.view_with_layout(
        nm.Shape(2) if y.ndim == 1 else nm.Shape(2, y.shape[1]),
        y.strides,
        0,
    )
    var score = model.score(Z, score_y)
    var expected_score = Float64(String(f[14]))
    if not isfinite(score) or abs(score - expected_score) > 5 * tol * (
        1 + abs(expected_score)
    ):
        raise Error(label + " score mismatch")
    compare(
        model.dual_coef_,
        array[DType.float64](String(f[11])),
        tol,
        label + " coefficients",
    )
    compare(
        predicted,
        array[DType.float64](String(f[12])),
        tol,
        label + " predictions",
    )


def main() raises:
    var args = argv()
    var text: String
    with open(args[1], "r") as handle:
        text = handle.read()
    comptime types = (DType.float32, DType.float64, DType.int32, DType.int64)
    comptime names = ("float32", "float64", "int32", "int64")
    var count = 0
    for line in text.splitlines():
        var f = line.split("|")
        count += 1
        comptime for x in range(4):
            if f[1] == names[x]:
                comptime for y in range(4):
                    if f[2] == names[y]:
                        comptime for z in range(4):
                            if f[3] == names[z]:
                                run_case[types[x], types[y], types[z]](
                                    String(line), count
                                )
    print("Dense dtype parity cases passed:", count)
