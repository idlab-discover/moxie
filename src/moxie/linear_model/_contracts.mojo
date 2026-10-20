"""Dense linear-model plumbing; no estimator or solver implementation."""
from numojo import NDArray
from std.math import isfinite


def validate_xy[dtype: DType](X: NDArray[dtype], y: NDArray[dtype]) raises:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if X.ndim != 2:
        raise Error("ValueError: X must be a dense matrix")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise Error("ValueError: X must have samples and features")
    if y.ndim != 1 and y.ndim != 2:
        raise Error("ValueError: y must have rank one or two")
    if y.shape[0] != X.shape[0]:
        raise Error("ValueError: inconsistent sample counts")
    if y.ndim == 2:
        if y.shape[1] == 0:
            raise Error("ValueError: y must have targets")


def validate_weights[dtype: DType](weights: NDArray[dtype], n_samples: Int, nonnegative: Bool = True) raises:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if weights.ndim != 1 or weights.shape[0] != n_samples or n_samples <= 0:
        raise Error("ValueError: weights must match samples")
    var positive = False
    for i in range(n_samples):
        var value = weights.unsafe_get(i * Int(weights.strides[0]))
        if not isfinite(value) or (nonnegative and value < 0):
            raise Error("ValueError: invalid sample weight")
        positive = positive or value > 0
    if nonnegative and not positive:
        raise Error("ValueError: weights must include positive mass")


def snapshot[dtype: DType](array: NDArray[dtype]) -> NDArray[dtype]:
    """Independent learned storage; never retain caller mutation aliases."""
    return array.copy()


@fieldwise_init
struct LinearShape(Copyable, Movable):
    var n_features: Int
    var n_outputs: Int
    var vector_target: Bool

    def validate(self) raises:
        if self.n_features <= 0 or self.n_outputs <= 0:
            raise Error("ValueError: invalid coefficient dimensions")
        if self.vector_target and self.n_outputs != 1:
            raise Error("ValueError: vector target requires one output")


@fieldwise_init
struct PreparedData[dtype: DType](Movable):
    var X: NDArray[Self.dtype]
    var y: NDArray[Self.dtype]
    var X_offset: NDArray[Self.dtype]
    var y_offset: NDArray[Self.dtype]
    var X_scale: NDArray[Self.dtype]
    var sample_weight_sqrt: Optional[NDArray[Self.dtype]]
    var vector_target: Bool


@fieldwise_init
struct LinearParameters[dtype: DType](Movable):
    var coef: NDArray[Self.dtype]
    var intercept: NDArray[Self.dtype]

    def validate(self, shape: LinearShape) raises:
        shape.validate()
        if self.coef.ndim != 2 or self.intercept.ndim != 1:
            raise Error("ValueError: coefficient/intercept rank")
        if self.coef.shape[0] != shape.n_outputs or self.coef.shape[1] != shape.n_features or self.intercept.shape[0] != shape.n_outputs:
            raise Error("ValueError: coefficient/intercept shape")


trait DensePreprocessor:
    comptime dtype: DType
    def prepare(self, X: NDArray[Self.dtype], y: NDArray[Self.dtype], weights: NDArray[Self.dtype], fit_intercept: Bool, rescale_with_sw: Bool) raises -> PreparedData[Self.dtype]:
        ...


trait DenseLinearPrediction:
    comptime dtype: DType

    def decision(self, X: NDArray[Self.dtype], coef: NDArray[Self.dtype], intercept: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        ...


trait DenseSolver:
    """Engine-specific configuration/state/result, without imposing capabilities."""
    comptime dtype: DType
    comptime Config: Copyable
    comptime State: Movable
    comptime Result: Movable

    def solve(self, X: NDArray[Self.dtype], y: NDArray[Self.dtype], weights: NDArray[Self.dtype], config: Self.Config, mut state: Self.State) raises -> Self.Result:
        ...
