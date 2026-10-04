"""Run generated sklearn/SciPy fixtures in a native Mojo process."""
from std.sys import argv
from std.math import isfinite, isnan
from std.testing import assert_raises
import numojo as nm
from moxie.kernel_ridge import KernelRidge, ScalarOrVector

comptime DenseArray = nm.NDArray[DType.float64]
from moxie._kernel_ridge._solve import least_squares
from moxie._kernel_ridge._kernels import pairwise


def numbers(text: String) raises -> List[Float64]:
    var result = List[Float64]()
    for value in text.split(","):
        result.append(Float64(String(value)))
    return result^


def vector(values: List[Float64]) raises -> DenseArray:
    var result = nm.zeros[DType.float64](nm.Shape(len(values)))
    for i in range(len(values)):
        result.unsafe_set(i, values[i])
    return result^


def array(text: String) raises -> DenseArray:
    var fields = text.split(":")
    var values = vector(numbers(String(fields[3])))
    if Int(String(fields[0])) == 1:
        return values^
    return values.reshape(
        nm.Shape(Int(String(fields[1])), Int(String(fields[2])))
    )


def scalar_or_vector(text: String) raises -> ScalarOrVector:
    if text.startswith("s:"):
        var fields = text.split(":")
        return ScalarOrVector(Float64(String(fields[1])))
    return ScalarOrVector(numbers(text))


def optional_number(text: String) raises -> Optional[Float64]:
    if text == "-":
        return None
    return Float64(text)


def optional_weights(text: String) raises -> Optional[ScalarOrVector]:
    if text == "-":
        return None
    return scalar_or_vector(text)


def compare(
    actual: DenseArray,
    expected: DenseArray,
    context: String,
    tolerance: Float64,
) raises:
    if actual.shape != expected.shape:
        raise Error(context + ": shape mismatch")
    for i in range(actual.size):
        var a = actual.unsafe_get(i)
        var e = expected.unsafe_get(i)
        if a == e or (isnan(a) and isnan(e)):
            continue
        if (
            not isfinite(a)
            or not isfinite(e)
            or abs(a - e) > tolerance * (1 + abs(e))
        ):
            raise Error(
                context
                + ": element "
                + String(i)
                + " actual="
                + String(a)
                + " expected="
                + String(e)
            )


def main() raises:
    with assert_raises():
        compare(vector([1.0]), vector([Float64("nan")]), "comparator", 1e-9)
    with assert_raises():
        compare(vector([Float64("nan")]), vector([1.0]), "comparator", 1e-9)
    var args = argv()
    var data: String
    with open(args[1], "r") as file:
        data = file.read()
    var count = 0
    for line in data.split("\n"):
        if line == "":
            continue
        var fields = line.split("|")
        var context = String(fields[0]) + " case " + String(count)
        if fields[0] == "lstsq":
            var result = least_squares(
                array(String(fields[1])), array(String(fields[2]))
            )
            compare(
                result,
                array(String(fields[3])),
                context,
                Float64(String(fields[4])),
            )
        elif fields[0] == "pairwise":
            var result = pairwise(
                array(String(fields[1])),
                array(String(fields[2])),
                String(fields[3]),
                optional_number(String(fields[4])),
                Float64(String(fields[5])),
                Float64(String(fields[6])),
            )
            compare(result, array(String(fields[7])), context, 2e-12)
        else:
            var X = array(String(fields[1]))
            var y = array(String(fields[2]))
            var model = KernelRidge(
                scalar_or_vector(String(fields[3])),
                kernel=String(fields[4]),
                gamma=optional_number(String(fields[5])),
                degree=Float64(String(fields[6])),
                coef0=Float64(String(fields[7])),
            )
            var weights = optional_weights(String(fields[8]))
            if fields[0] == "fit_error":
                with assert_raises():
                    if weights:
                        _ = model.fit(X, y, sample_weight=weights.value())
                    else:
                        _ = model.fit(X, y)
            else:
                if weights:
                    _ = model.fit(X, y, sample_weight=weights.value())
                else:
                    _ = model.fit(X, y)
                var tolerance = Float64(String(fields[9]))
                compare(
                    model.dual_coef_[DenseArray],
                    array(String(fields[10])),
                    context + " dual",
                    tolerance,
                )
                compare(
                    model.X_fit_[DenseArray], X, context + " retained X", 0.0
                )
                if model.n_features_in_ != X.shape[1]:
                    raise Error(context + ": feature count mismatch")
                var query = array(String(fields[11]))
                compare(
                    model.predict(query)[DenseArray],
                    array(String(fields[12])),
                    context + " predict",
                    tolerance,
                )
                var score_y = array(String(fields[13]))
                var score = model.score(
                    query, score_y, optional_weights(String(fields[14]))
                )
                compare(
                    vector([score]),
                    vector([Float64(String(fields[15]))]),
                    context + " score",
                    tolerance * 10,
                )
        count += 1
    print("Kernel ridge reference parity checks passed:", count, "cases")
