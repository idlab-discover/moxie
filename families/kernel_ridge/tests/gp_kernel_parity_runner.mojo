from std.sys import argv
from std.testing import assert_equal, assert_false, assert_raises
import numojo as nm
from moxie.gaussian_process.kernels import ExpSineSquared, WhiteKernel
from moxie.kernel_ridge import (
    FloatingArray,
    GaussianProcessKernel,
    KernelParameters,
    KernelRidge,
    MatrixKernel,
)
from parity_runner import array, compare


comptime Dense64 = nm.NDArray[DType.float64]
comptime PeriodicRidge = KernelRidge[DType.float64, GaussianProcessKernel]


def run_case[dtype: DType](line: String) raises:
    var fields = line.split("|")
    var X = array[dtype](String(fields[2]))
    var y = array[dtype](String(fields[3]))
    var Z = array[dtype](String(fields[4]))
    var periodic = ExpSineSquared(0.9, 4.5)
    var white = WhiteKernel(5.0)
    var name = String(fields[1])
    var model = PeriodicRidge(
        alpha=0.5, kernel=GaussianProcessKernel(ExpSineSquared())
    )
    if name == "sum":
        _ = model.set_params(
            kernel=GaussianProcessKernel(periodic + white.as_kernel())
        )
    elif name == "white":
        _ = model.set_params(kernel=GaussianProcessKernel(white))
    else:
        _ = model.set_params(kernel=GaussianProcessKernel(periodic))
    if y.ndim == 2:
        var weights: List[Float64] = [0.5, 1.0, 2.0, 1.5]
        _ = model.fit(X, y, sample_weight=weights)
    else:
        _ = model.fit(X, y)
    var prediction = model.predict(Z)
    # Reused Float64 GP distances and the ridge solve can differ in rounding
    # from SciPy. The positive penalty keeps these systems well conditioned.
    compare(
        model.dual_coef_[Dense64],
        array[DType.float64](String(fields[5])),
        1e-10,
        name + " dual",
    )
    compare(
        prediction[Dense64],
        array[DType.float64](String(fields[6])),
        1e-10,
        name + " prediction",
    )
    var clone = model.__sklearn_clone__()
    assert_false(clone.__sklearn_is_fitted__())
    assert_equal(
        clone.get_params_record().native_kernel.value().kernel,
        model.get_params_record().native_kernel.value().kernel,
    )
    with assert_raises():
        _ = clone.predict(Z)
    var unsupported: KernelParameters = {"periodicity": 2.0}
    _ = model.set_params(kernel_params=unsupported^)
    with assert_raises():
        _ = model.fit(X, y)
    _ = model.set_params(kernel_params=None)
    compare(
        model.predict(Z)[Dense64], prediction[Dense64], 0.0, name + " failed fit"
    )


@fieldwise_init
struct InvalidCrossKernel(Copyable, MatrixKernel):
    def evaluate_matrix[
        dtype: DType
    ](
        self,
        X: nm.NDArray[dtype],
        var Y: nm.NDArray[dtype],
        same_data: Bool,
        params: Optional[KernelParameters],
    ) raises -> FloatingArray:
        var rows = X.shape[0] if same_data else Y.shape[0]
        return FloatingArray(nm.zeros[DType.float64](nm.Shape(rows, Y.shape[0])))


def check_invalid_prediction_shape() raises:
    var X = nm.zeros[DType.float64](nm.Shape(3, 2))
    var y = nm.zeros[DType.float64](nm.Shape(3))
    var Z = nm.zeros[DType.float64](nm.Shape(1, 2))
    var model = KernelRidge[DType.float64, InvalidCrossKernel](
        kernel=InvalidCrossKernel()
    )
    _ = model.fit(X, y)
    with assert_raises():
        _ = model.predict(Z)


def main() raises:
    var args = argv()
    var cases = 0
    with open(String(args[1]), "r") as file:
        for line in file.read().split("\n"):
            if line.byte_length() == 0:
                continue
            var dtype = line.split("|")[0]
            if dtype == "float32":
                run_case[DType.float32](String(line))
            elif dtype == "float64":
                run_case[DType.float64](String(line))
            elif dtype == "int32":
                run_case[DType.int32](String(line))
            elif dtype == "int64":
                run_case[DType.int64](String(line))
            else:
                raise Error("Unexpected fixture dtype")
            cases += 1
    assert_equal(cases, 24)
    check_invalid_prediction_shape()
    print("GP KernelRidge parity and contracts passed:", cases, "cases")
