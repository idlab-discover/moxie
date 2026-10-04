"""End-to-end estimator parity; Python is fixture tooling only."""
from moxie.kernel_ridge import KernelParameters, KernelParameterValue
import numojo as nm
from std.sys import argv
from std.math import isfinite, isnan
from moxie.kernel_ridge import KernelRidge, ScalarOrVector, FloatingArray
from moxie._kernel_ridge import RowKernel


@fieldwise_init
struct Asymmetric(Copyable, RowKernel):
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

    def evaluate_with_params[
        dtype: DType
    ](
        self,
        X: nm.NDArray[dtype],
        i: Int,
        Y: nm.NDArray[dtype],
        j: Int,
        params: Optional[KernelParameters],
    ) raises -> SIMD[dtype, 1]:
        var scale: Float64 = 1
        if params:
            scale = params.value()["scale"][Float64]
            if params.value()["enabled"][Bool] and params.value()["mode"][String] == "scaled":
                scale *= Float64(params.value()["factor"][Int])
        return SIMD[dtype, 1](scale) * self.evaluate(X, i, Y, j)


def numbers(text: String) raises -> List[Float64]:
    var values = List[Float64]()
    for value in text.split(","):
        values.append(Float64(String(value)))
    return values^


def array[dtype: DType](text: String) raises -> nm.NDArray[dtype]:
    var fields = text.split(":")
    var rows = Int(String(fields[0]))
    var cols = Int(String(fields[1]))
    var values = numbers(String(fields[2]))
    var a = nm.zeros[dtype](nm.Shape(rows, max(cols, 1)))
    for i in range(len(values)):
        a.unsafe_set(i, SIMD[dtype, 1](values[i]))
    if cols == 0:
        return a.reshape(nm.Shape(rows))
    return a^


def compare[
    dtype: DType
](
    actual: nm.NDArray[dtype],
    expected: nm.NDArray[dtype],
    tol: Float64,
    label: String,
) raises:
    if actual.shape != expected.shape:
        raise Error(label + " shape mismatch")
    for i in range(actual.size):
        var a = Float64(actual.unsafe_get(i))
        var b = Float64(expected.unsafe_get(i))
        if not isfinite(a) or abs(a - b) > tol * (1 + abs(b)):
            raise Error(
                label
                + " element "
                + String(i)
                + " actual="
                + String(a)
                + " expected="
                + String(b)
            )


def run_case[dtype: DType](line: String, index: Int) raises:
    var f = line.split("|")
    var X = array[dtype](String(f[2]))
    var y = array[dtype](String(f[3]))
    var Z = array[dtype](String(f[4]))
    var yt = array[dtype](String(f[5]))
    var gamma: Optional[Float64] = None
    if String(f[8]) != "none":
        gamma = Float64(String(f[8]))
    var metric: Optional[Asymmetric] = None
    var name = String(f[1])
    if name == "callable" or name == "callable_params":
        metric = Asymmetric()
        name = "linear"
    var model = KernelRidge[dtype, Asymmetric](
        ScalarOrVector(numbers(String(f[6]))),
        kernel=name,
        gamma=gamma,
        degree=Float64(String(f[9])),
        coef0=Float64(String(f[10])),
        native_kernel=metric^,
    )
    if String(f[1]) == "callable_params":
        _ = model.set_params(kernel=Asymmetric())
        var params: KernelParameters = {"scale": 1.75}
        params["factor"] = KernelParameterValue(Int(2))
        params["enabled"] = KernelParameterValue(True)
        params["mode"] = KernelParameterValue(String("scaled"))
        _ = model.set_params(kernel_params=params^)
    var failed = False
    try:
        if String(f[7]) == "none":
            _ = model.fit(X, y)
        else:
            var wf = f[7].split(":")
            if String(wf[0]) == "s":
                _ = model.fit(X, y, sample_weight=Float64(String(wf[1])))
            else:
                _ = model.fit(
                    X, y, sample_weight=ScalarOrVector(numbers(String(wf[1])))
                )
    except e:
        failed = True
        if String(f[15]) != "1":
            raise Error("case " + String(index) + " " + String(e))
    if String(f[15]) == "1":
        if not failed:
            raise Error("expected solve error: " + String(index))
        return
    var tol = Float64(String(f[14]))
    var label = "case " + String(index) + " " + String(f[1])
    if len(f) > 16:
        var params = model.get_params_record()
        if String(f[16]) != "keep":
            params.kernel = String(f[16])
        if String(f[17]) == "none":
            params.gamma = None
        elif String(f[17]) != "keep":
            params.gamma = Float64(String(f[17]))
        if String(f[18]) != "keep":
            params.degree = Float64(String(f[18]))
        if String(f[19]) != "keep":
            params.coef0 = Float64(String(f[19]))
        if String(f[20]) != "keep":
            params.alpha = ScalarOrVector(Float64(String(f[20])))
        if String(f[21]) != "keep":
            var ignored: KernelParameters = {"unused": 19.0}
            params.kernel_params = ignored^
        _ = model.set_params(params)
    compare(model.dual_coef_, array[dtype](String(f[11])), tol, label + " dual")
    compare(
        model.predict(Z),
        array[dtype](String(f[12])),
        tol,
        label + " prediction",
    )
    var actual_score = model.score(Z, yt)
    var expected_score = Float64(String(f[13]))
    if not (isnan(actual_score) and isnan(expected_score)):
        if not isfinite(actual_score) or abs(
            actual_score - expected_score
        ) > 5 * tol * (1 + abs(expected_score)):
            raise Error(
                label
                + " score "
                + String(actual_score)
                + " expected "
                + String(expected_score)
            )


def main() raises:
    var args = argv()
    var text: String
    with open(args[1], "r") as f:
        text = f.read()
    var n = 0
    for line in text.splitlines():
        if line.byte_length() == 0:
            continue
        n += 1
        if line.startswith("32|"):
            run_case[DType.float32](String(line), n)
        else:
            run_case[DType.float64](String(line), n)
    print("Dense estimator parity cases passed:", n)


def compare[
    dtype: DType
](
    actual: FloatingArray,
    expected: nm.NDArray[dtype],
    tol: Float64,
    label: String,
) raises:
    if actual.isa[nm.NDArray[DType.float32]]():
        compare(
            actual[nm.NDArray[DType.float32]].astype[DType.float64](),
            expected.astype[DType.float64](),
            tol,
            label,
        )
    else:
        compare(
            actual[nm.NDArray[DType.float64]],
            expected.astype[DType.float64](),
            tol,
            label,
        )
