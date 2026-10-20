"""Dense RBF affinity, shared by both graph estimators."""
from std.math import isfinite, max
from numax.core.libm import exp
from numojo import NDArray, Shape, zeros
from moxie._dense import row_squared_norms
from ._contracts import DenseAffinity, validate_features

def finite_features[dtype: DType](X: NDArray[dtype]) raises:
    if X.ndim != 2:
        raise Error("ValueError: expected rank-2 features")
    validate_features(X, X.shape[0])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not isfinite(X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))):
                raise Error("ValueError: features must be finite")

struct RBFAffinity[dtype: DType = DType.float64](DenseAffinity, Copyable, Movable):
    comptime Input = NDArray[Self.dtype]
    var gamma: Float64
    def __init__(out self, gamma: Float64 = 20):
        self.gamma = gamma
    def affinity(self, train: Self.Input, query: Self.Input) raises -> NDArray[DType.float64]:
        finite_features(train)
        finite_features(query)
        if train.shape[1] != query.shape[1] or not isfinite(self.gamma) or self.gamma < 0:
            raise Error("ValueError: incompatible features or invalid gamma")
        var a = train.astype[DType.float64]()
        var b = query.astype[DType.float64]()
        var an = row_squared_norms(a)
        var bn = row_squared_norms(b)
        var result = a @ b.T()
        for i in range(train.shape[0]):
            for j in range(query.shape[0]):
                var distance = max(Float64(0), an.unsafe_get(i) + bn.unsafe_get(j) - 2 * result.unsafe_get(i * query.shape[0] + j))
                var value = exp(-self.gamma * distance)
                comptime if Self.dtype == DType.float32:
                    # sklearn rounds distances and RBF arithmetic to the input dtype.
                    value = Float64(Float32(exp(Float64(-Float32(self.gamma) * Float32(distance)))))
                result.unsafe_set(i * query.shape[0] + j, value)
        return result^
