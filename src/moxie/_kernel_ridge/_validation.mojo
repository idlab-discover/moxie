"""Kernel-ridge input checks; storage and layout belong to NuMojo."""
import numojo as nm
from std.math import isfinite


def epsilon[dtype: DType]() -> SIMD[dtype, 1]:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    comptime if dtype == DType.float32:
        return 1.1920928955078125e-7
    else:
        return 2.220446049250313e-16


def validate[
    dtype: DType
](a: nm.NDArray[dtype], name: String, matrix: Bool = False) raises:
    comptime assert dtype in (
        DType.float32,
        DType.float64,
        DType.int32,
        DType.int64,
    )
    if a.ndim != 1 and a.ndim != 2:
        raise Error("ValueError: " + name + " must be a vector or matrix")
    if matrix and a.ndim != 2:
        raise Error("ValueError: " + name + " must be a matrix")
    if a.shape[0] == 0 or (a.ndim == 2 and a.shape[1] == 0):
        raise Error("ValueError: empty " + name)
    var row_stride = Int(a.strides[0])
    var col_stride = Int(a.strides[1]) if a.ndim == 2 else 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1] if a.ndim == 2 else 1):
            var offset = i * row_stride
            if a.ndim == 2:
                offset += j * col_stride
            if not isfinite(Float64(a.unsafe_get(offset))):
                raise Error("ValueError: nonfinite " + name)


def matrix_copy[dtype: DType](a: nm.NDArray[dtype]) raises -> nm.NDArray[dtype]:
    # Caller validates rank first. Materialize only logical elements of views.
    var cols = a.shape[1] if a.ndim == 2 else 1
    var result = nm.zeros[dtype](nm.Shape(a.shape[0], cols))
    var row_stride = Int(a.strides[0])
    var col_stride = Int(a.strides[1]) if a.ndim == 2 else 0
    for i in range(a.shape[0]):
        for j in range(cols):
            var offset = i * row_stride
            if a.ndim == 2:
                offset += j * col_stride
            result.unsafe_set(i * cols + j, a.unsafe_get(offset))
    return result^


def check_system[
    dtype: DType
](A: nm.NDArray[dtype], y: nm.NDArray[dtype]) raises:
    validate(A, "kernel", True)
    validate(y, "targets", True)
    if A.shape[0] != A.shape[1] or y.shape[0] != A.shape[0]:
        raise Error("ValueError: incompatible kernel-system shapes")


def floating_dtype(dtype: DType) -> DType:
    return DType.float32 if dtype == DType.float32 else DType.float64


def dense_cast[
    dtype: DType, //, target: DType
](a: nm.NDArray[dtype]) raises -> nm.NDArray[target]:
    # NuMojo astype traverses physical storage, so materialize views first.
    var contiguous = matrix_copy(a)
    var converted = contiguous.astype[target]()
    if a.ndim == 1:
        return converted.reshape(nm.Shape(a.shape[0]))
    return converted^
