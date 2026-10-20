"""Private dense storage contracts; no neural-network computation."""
from numojo import NDArray
from std.math import isfinite


def validate_matrix[dtype: DType](X: NDArray[dtype]) raises:
    """Validate nonempty finite input without changing layout or ownership."""
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if X.ndim != 2:
        raise Error("ValueError: expected a dense rank-2 matrix")
    if X.shape[0] <= 0 or X.shape[1] <= 0:
        raise Error("ValueError: expected samples and features")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not isfinite(X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))):
                raise Error("ValueError: expected finite matrix values")


def snapshot_matrix[dtype: DType](X: NDArray[dtype]) raises -> NDArray[dtype]:
    validate_matrix(X)
    return X.copy()


struct LayerLayout(Copyable, Movable):
    """Owned topology including input and output widths; no tensor buffer."""
    var units: List[Int]

    def __init__(out self, units: List[Int]) raises:
        if len(units) < 2:
            raise Error("ValueError: topology needs input and output widths")
        for width in units:
            if width <= 0:
                raise Error("ValueError: layer widths must be positive")
        self.units = units.copy()


struct LayerTensors[dtype: DType](Copyable, Movable):
    """Independent arrays in sklearn coefs/intercepts order, also for gradients."""
    var coefs: List[NDArray[Self.dtype]]
    var intercepts: List[NDArray[Self.dtype]]

    def __init__(
        out self, layout: LayerLayout,
        coefs: List[NDArray[Self.dtype]], intercepts: List[NDArray[Self.dtype]],
    ) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        if len(coefs) != len(layout.units) - 1 or len(intercepts) != len(coefs):
            raise Error("ValueError: tensor count differs from topology")
        # Validate every shape before copying any tensor. Gradient values may
        # be nonfinite; the numerical engine owns post-update finite checks.
        for i in range(len(coefs)):
            if coefs[i].ndim != 2 or intercepts[i].ndim != 1:
                raise Error("ValueError: expected coefficient matrix and bias vector")
            if (coefs[i].shape[0] != layout.units[i]
                or coefs[i].shape[1] != layout.units[i + 1]
                or intercepts[i].shape[0] != layout.units[i + 1]):
                raise Error("ValueError: tensor dimensions differ from topology")
        self.coefs = List[NDArray[Self.dtype]]()
        self.intercepts = List[NDArray[Self.dtype]]()
        for i in range(len(coefs)):
            self.coefs.append(coefs[i].copy())
            self.intercepts.append(intercepts[i].copy())


def validate_incremental(solver: String, early_stopping: Bool) raises:
    if solver != "sgd" and solver != "adam":
        raise Error("ValueError: partial_fit requires sgd or adam")
    if early_stopping:
        raise Error("ValueError: partial_fit does not support early_stopping")
