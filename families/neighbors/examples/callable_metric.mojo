"""A stateful native distance for public brute-force neighbor estimators."""
import numojo as nm
from std.math import isfinite
from moxie.neighbors._contracts import DenseMetric
from moxie.neighbors._distance import at

@fieldwise_init
struct WeightedL1[dtype_: DType](DenseMetric):
    comptime dtype = Self.dtype_
    var weights: nm.NDArray[DType.float64]

    def validate_features(self, n_features: Int) raises:
        if self.weights.ndim != 1 or self.weights.shape[0] != n_features:
            raise Error("ValueError: one metric weight required per feature")
        for j in range(n_features):
            var weight = self.weights.unsafe_get(j * Int(self.weights.strides[0]))
            if not isfinite(weight) or weight <= 0:
                raise Error("ValueError: metric weights must be finite and positive")

    def distance(self, x: nm.NDArray[Self.dtype], i: Int,
                 y: nm.NDArray[Self.dtype], j: Int) raises -> Float64:
        var total = Float64(0)
        for col in range(x.shape[1]):
            total += self.weights.unsafe_get(col * Int(self.weights.strides[0])) * abs(at(x, i, col) - at(y, j, col))
        return total
