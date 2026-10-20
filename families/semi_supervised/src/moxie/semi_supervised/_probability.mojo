"""Owned row normalization with explicit zero-mass semantics."""
from numojo import NDArray, Shape, zeros
from moxie._dense import sum_axis2d
from ._contracts import ProbabilityNormalizer, ZeroMassPolicy

@fieldwise_init
struct RowNormalizer(ProbabilityNormalizer):
    def normalize(self, values: NDArray[DType.float64], zero_mass: Int) raises -> NDArray[DType.float64]:
        if zero_mass != ZeroMassPolicy.PRESERVE and zero_mass != ZeroMassPolicy.IEEE_DIVIDE:
            raise Error("ValueError: invalid zero-mass policy")
        var sums = sum_axis2d(values, 1)
        var result = zeros[DType.float64](Shape(values.shape[0], values.shape[1]))
        for i in range(values.shape[0]):
            var denominator = sums.unsafe_get(i)
            if denominator == 0 and zero_mass == ZeroMassPolicy.PRESERVE:
                denominator = 1
            for j in range(values.shape[1]):
                result.unsafe_set(i * values.shape[1] + j, values.unsafe_get(i * Int(values.strides[0]) + j * Int(values.strides[1])) / denominator)
        return result^
