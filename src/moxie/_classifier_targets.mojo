# SPDX-License-Identifier: BSD-3-Clause
"""Native numeric List storage for classifier component targets."""


struct NumericListTarget[Target: Movable]:
    comptime dtype = (
        DType.float64 if Self.Target == List[Float64] else
        DType.float32 if Self.Target == List[Float32] else DType.int64
    )
    comptime Label = Int if Self.Target == List[Int] else Scalar[Self.dtype]
    comptime supported = Self.Target == List[Self.Label]

    @staticmethod
    def from_array_label[dtype: DType](label: Scalar[dtype]) -> Self.Label:
        """Convert a label after the caller has checked its domain."""
        comptime if Self.Target == List[Int]:
            return rebind[Self.Label](Int(label.cast[DType.int64]()))
        else:
            return rebind[Self.Label](label.cast[Self.dtype]())

    @staticmethod
    def as_float64(label: Self.Label) -> Float64:
        comptime if Self.Target == List[Int]:
            return Float64(rebind[Int](label))
        else:
            return rebind[Scalar[Self.dtype]](label).cast[DType.float64]()
