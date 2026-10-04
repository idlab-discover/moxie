"""A typed native kernel with independently owned configuration."""
import numojo as nm
from moxie.kernel_ridge import KernelRidge, RowKernel
from moxie.base import clone
from std.math import isfinite
from moxie.kernel_ridge_serialization import (
    NativeKernelAdapter,
    KernelRidgeSchema,
)
from moxie.serialization import (
    BinaryWriter,
    BinaryReader,
    save_model,
    load_model,
)


@fieldwise_init
struct AffineKernel(Copyable, RowKernel):
    var offset: Float64

    def evaluate[
        dtype: DType
    ](
        self,
        X: nm.NDArray[dtype],
        i: Int,
        Y: nm.NDArray[dtype],
        j: Int,
    ) raises -> SIMD[dtype, 1]:
        var value = SIMD[dtype, 1](self.offset)
        for k in range(X.shape[1]):
            value += X.unsafe_get(
                i * Int(X.strides[0]) + k * Int(X.strides[1])
            ) * Y.unsafe_get(j * Int(Y.strides[0]) + k * Int(Y.strides[1]))
        return value


struct AffineKernelAdapter(NativeKernelAdapter):
    """Versioned state for this example; only finite offsets are supported."""

    comptime Kernel = AffineKernel

    @staticmethod
    def identifier() -> String:
        return "moxie.examples.AffineKernel"

    @staticmethod
    def state_version() -> Int:
        return 1

    @staticmethod
    def encode(kernel: Self.Kernel) raises -> List[UInt8]:
        if not isfinite(kernel.offset):
            raise Error("ModelSchemaError: affine offset must be finite")
        var writer = BinaryWriter()
        writer.scalar[DType.float64](kernel.offset)
        return writer^.finish()

    @staticmethod
    def decode(state: Span[UInt8, _]) raises -> Self.Kernel:
        var reader = BinaryReader(state)
        var offset = reader.scalar[DType.float64]()
        reader.finish()
        if not isfinite(offset):
            raise Error("ModelSchemaError: affine offset must be finite")
        return AffineKernel(offset)


def main() raises:
    var X = nm.arange[DType.float32](4).reshape(nm.Shape(4, 1))
    var y = nm.arange[DType.float32](4)
    var model = KernelRidge[DType.float32, AffineKernel](
        alpha=0.1, native_kernel=AffineKernel(1.0)
    )
    _ = model.fit(X, y)
    comptime Schema = KernelRidgeSchema[
        DType.float32, AffineKernel, AffineKernelAdapter
    ]
    save_model[Schema](model, "builds/callable-kernel.moxie")
    var restored = load_model[Schema]("builds/callable-kernel.moxie")
    print("Restored callable predictions:", restored.predict(X))
    var fresh = clone(model)
    _ = fresh.set_params(native_kernel=AffineKernel(2.0))
    _ = fresh.fit(X, y)
    print("Updated callable predictions:", fresh.predict(X))
