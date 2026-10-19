"""Dense two-block contracts shared by the Moxie cross-decomposition estimators."""
# Validation semantics follow scikit-learn; see LICENSES/scikit-learn.txt.
from numojo import NDArray, Shape
from std.math import isfinite

comptime Matrix = NDArray[DType.float64]


@fieldwise_init
struct DenseBlocks(Copyable, Movable):
    var x: Matrix
    var y: Matrix


@fieldwise_init
struct OwnedTarget(Copyable, Movable):
    var values: Matrix
    var was_1d: Bool


@fieldwise_init
struct PairedScores(Copyable, Movable):
    var x_scores: Matrix
    var y_scores: Matrix


@fieldwise_init
struct ProjectionWeights(Copyable, Movable):
    var x_weights: Matrix
    var y_weights: Matrix


@fieldwise_init
struct Scaling(Copyable, Movable):
    var x_mean: Matrix
    var y_mean: Matrix
    var x_std: Matrix
    var y_std: Matrix


@fieldwise_init
struct WeightStep(Copyable, Movable):
    var x_weights: Matrix
    var y_weights: Matrix
    var n_iter: Int
    var converged: Bool
    var constant_target: Bool


@fieldwise_init
struct Loadings(Copyable, Movable):
    var x_loadings: Matrix
    var y_loadings: Matrix


struct KernelOptions(Copyable, Movable):
    """Private engine controls, not the public estimator constructor."""

    var mode: String
    var deflation_mode: String
    var algorithm: String
    var max_iter: Int
    var tol: Float64

    def __init__(
        out self,
        mode: String = "A",
        deflation_mode: String = "regression",
        algorithm: String = "nipals",
        max_iter: Int = 500,
        tol: Float64 = 1e-6,
    ):
        self.mode = mode
        self.deflation_mode = deflation_mode
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol

    def validate(self) raises:
        if self.mode != "A" and self.mode != "B":
            raise Error("ValueError: mode must be A or B")
        if (
            self.deflation_mode != "regression"
            and self.deflation_mode != "canonical"
        ):
            raise Error("ValueError: unknown deflation mode")
        if self.algorithm != "nipals" and self.algorithm != "svd":
            raise Error("ValueError: algorithm must be nipals or svd")
        if self.max_iter < 1 or not isfinite(self.tol) or self.tol < 0:
            raise Error("ValueError: invalid iteration controls")


def validate_matrix(x: Matrix, min_samples: Int = 1) raises:
    if x.ndim != 2:
        raise Error("ValueError: expected a rank-2 dense matrix")
    if x.shape[0] < min_samples or x.shape[1] < 1:
        raise Error("ValueError: insufficient samples or features")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not isfinite(
                x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]))
            ):
                raise Error("ValueError: input contains NaN or infinity")


def validate_blocks(
    x: Matrix, y: Matrix, n_components: Int, regression: Bool = False
) raises:
    """Internal normalized 2-D targets; public 1-D normalization is separate."""
    validate_matrix(x, 2)
    validate_matrix(y, 2)
    if x.shape[0] != y.shape[0]:
        raise Error("ValueError: X and y have different sample counts")
    var bound = min(x.shape[0], x.shape[1])
    if not regression:
        bound = min(bound, y.shape[1])
    if n_components < 1 or n_components > bound:
        raise Error("ValueError: n_components outside rank upper bound")


def owned_blocks(
    x: Matrix, y: Matrix, n_components: Int, regression: Bool = False
) raises -> DenseBlocks:
    """Independent workspaces for copy=True."""
    validate_blocks(x, y, n_components, regression)
    return DenseBlocks(x.copy(), y.copy())


trait TwoBlockEngine(Movable):
    """Implemented once by PLSEngine in _pls; no default numerical bodies."""

    def center_scale(
        self, mut x: Matrix, mut y: Matrix, scale: Bool
    ) raises -> Scaling:
        ...

    def leading_weights(
        self, x: Matrix, y: Matrix, options: KernelOptions
    ) raises -> WeightStep:
        ...

    def deflate(
        self,
        mut x: Matrix,
        mut y: Matrix,
        scores: PairedScores,
        canonical: Bool,
    ) raises -> Loadings:
        ...

    def projection_weights(
        self, x: Matrix, y: Matrix, n_components: Int
    ) raises -> ProjectionWeights:
        ...

    def rotations(self, weights: Matrix, loadings: Matrix) raises -> Matrix:
        ...


def owned_target(y: Matrix) raises -> OwnedTarget:
    """Normalize a vector target with an independent copy, preserving rank intent.
    """
    if y.ndim == 1:
        var values = y.reshape(Shape(y.shape[0], 1))
        validate_matrix(values)
        return OwnedTarget(values^, True)
    validate_matrix(y)
    return OwnedTarget(y.copy(), False)
