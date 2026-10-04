"""Explicit fixture maintenance tool; never run by the test suite."""
import numojo as nm
from moxie.kernel_ridge import KernelRidge, RowKernel
from moxie.kernel_ridge_serialization import (
    KernelRidgeSchema,
    NativeKernelAdapter,
    _NoKernelAdapter,
)
from moxie._kernel_ridge._estimator import NoKernel
from moxie.serialization import BinaryWriter, save_model
from callable_kernel import AffineKernel, AffineKernelAdapter


def write_fixtures[
    dtype: DType,
    Metric: RowKernel & Copyable & Deinitable = NoKernel,
    Adapter: NativeKernelAdapter = _NoKernelAdapter[Metric],
](var model: KernelRidge[dtype, Metric], name: String) raises:
    comptime Schema = KernelRidgeSchema[dtype, Metric, Adapter]
    var prefix = "builds/schema-v2-" + name + "-" + String(dtype)
    save_model[Schema](model, prefix + "-unfitted.moxie")
    var X = nm.arange[dtype](6).reshape(nm.Shape(3, 2))
    _ = model.fit(X, nm.arange[dtype](3))
    save_model[Schema](model, prefix + "-fitted.moxie")
    var query = nm.arange[dtype](4).reshape(nm.Shape(2, 2))
    var predictions = model.predict(query)
    var writer = BinaryWriter()
    for i in range(2):
        writer.scalar[dtype](predictions[nm.NDArray[dtype]].item(i))
    var bytes = writer^.finish()
    with open(prefix + ".predictions", "w") as file:
        file.write_all(bytes)


def main() raises:
    comptime types = (DType.float32, DType.float64)
    comptime for i in range(2):
        comptime dt = types[i]
        write_fixtures[dt](KernelRidge[dt](), "linear")
        write_fixtures[dt, AffineKernel, AffineKernelAdapter](
            KernelRidge[dt, AffineKernel](
                alpha=0.25, native_kernel=AffineKernel(1.5)
            ),
            "affine",
        )
