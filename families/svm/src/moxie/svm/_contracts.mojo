"""Dense family plumbing; solver implementations belong to engine children."""
import numojo as nm
from std.math import isfinite

comptime Dense = nm.NDArray[DType.float64]
comptime Indices = nm.NDArray[DType.int64]


def validate_matrix(x: Dense, columns: Int = -1) raises:
    if x.ndim != 2:
        raise Error("ValueError: expected dense rank-2 input")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise Error("ValueError: expected nonempty samples and features")
    if columns >= 0 and x.shape[1] != columns:
        raise Error("ValueError: incompatible feature or training-column count")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not isfinite(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]))):
                raise Error("ValueError: input must be finite")


def validate_vector(x: Dense, length: Int) raises:
    if x.ndim != 1 or x.shape[0] != length:
        raise Error("ValueError: expected matching rank-1 vector")
    for i in range(length):
        if not isfinite(x.unsafe_get(i * Int(x.strides[0]))):
            raise Error("ValueError: vector must be finite")


def pair_index(classes: Int, first: Int, second: Int) raises -> Int:
    """Lexicographic one-vs-one packing: (0,1), (0,2), ..., (k-2,k-1)."""
    if classes < 2 or first < 0 or second <= first or second >= classes:
        raise Error("ValueError: invalid class pair")
    return first * (2 * classes - first - 1) // 2 + second - first - 1


@fieldwise_init
struct KernelSpec(Copyable, Movable):
    """Resolved kernel configuration, not public estimator defaults."""
    var kind: String
    var gamma: Float64
    var degree: Int
    var coef0: Float64

    def validate(self) raises:
        if self.kind != "linear" and self.kind != "poly" and self.kind != "rbf" and self.kind != "sigmoid" and self.kind != "precomputed" and self.kind != "callable":
            raise Error("ValueError: unknown kernel")
        if not isfinite(self.gamma) or self.gamma < 0 or self.degree < 0 or not isfinite(self.coef0):
            raise Error("ValueError: invalid resolved kernel parameters")


struct TrainingSnapshot(Copyable, Movable):
    """An independent fitted snapshot using NuMojo ownership, including strides."""
    var x: Dense
    var n_features: Int
    var n_train: Int
    var precomputed: Bool

    def __init__(out self, x: Dense, precomputed: Bool = False) raises:
        validate_matrix(x)
        if precomputed and x.shape[0] != x.shape[1]:
            raise Error("ValueError: training Gram matrix must be square")
        self.x = x.copy()
        self.n_features = x.shape[1]
        self.n_train = x.shape[0]
        self.precomputed = precomputed

    def validate_query(self, x: Dense) raises:
        validate_matrix(x, self.n_train if self.precomputed else self.n_features)


@fieldwise_init
struct KernelModel(Movable):
    """Owned engine result; internal LIBSVM signs, no public binary sign flip."""
    var support: Indices
    var support_vectors: Dense
    var n_support: Indices
    var dual_coef: Dense
    var rho: Dense
    var iterations: Indices
    var fit_status: Int
    var prob_a: Dense
    var prob_b: Dense


@fieldwise_init
struct LinearModel(Movable):
    var coef: Dense
    var intercept: Dense
    var iterations: Indices
    var fit_status: Int


@fieldwise_init
struct KernelOptions(Copyable, Movable):
    var formulation: String
    var kernel: KernelSpec
    var c: Float64
    var nu: Float64
    var epsilon: Float64
    var tol: Float64
    var cache_mb: Float64
    var shrinking: Bool
    var probability: Bool
    var max_iter: Int
    var seed: Int


@fieldwise_init
struct LinearOptions(Copyable, Movable):
    var solver: Int
    var c: Float64
    var epsilon: Float64
    var tol: Float64
    var fit_intercept: Bool
    var intercept_scaling: Float64
    var max_iter: Int
    var seed: Int


trait KernelProvider(Movable):
    """A native callable supplies a full dense cross-kernel matrix."""
    def evaluate(self, x: Dense, train: Dense) raises -> Dense:
        ...


trait KernelEngine(Movable):
    """The LIBSVM owner alone supplies this numerical implementation."""
    def solve(self, x: Dense, y: Dense, sample_weight: Dense,
              class_weight: Dense, options: KernelOptions) raises -> KernelModel:
        ...

    def decision_values(self, x: Dense, model: KernelModel,
                        kernel: KernelSpec) raises -> Dense:
        ...

    def class_probabilities(self, decisions: Dense, model: KernelModel) raises -> Dense:
        ...


trait LinearEngine(Movable):
    """The LIBLINEAR owner alone supplies this numerical implementation."""
    def solve(self, x: Dense, y: Dense, sample_weight: Dense,
              class_weight: Dense, options: LinearOptions) raises -> LinearModel:
        ...

    def decision_values(self, x: Dense, model: LinearModel) raises -> Dense:
        ...


@fieldwise_init
struct ClassPacking(Movable):
    """Class-contiguous support blocks; labels themselves belong to wrappers."""
    var counts: Indices
    var offsets: Indices

    def validate(self, total_support: Int) raises:
        if total_support < 0:
            raise Error("ValueError: negative total support")
        if self.counts.ndim != 1 or self.offsets.ndim != 1:
            raise Error("ValueError: packing requires vectors")
        var classes = self.counts.shape[0]
        if classes < 1 or self.offsets.shape[0] != classes + 1:
            raise Error("ValueError: incompatible support offsets")
        var total = 0
        for i in range(classes):
            var count = Int(self.counts.unsafe_get(i * Int(self.counts.strides[0])))
            if count < 0 or count > total_support - total or Int(self.offsets.unsafe_get(i * Int(self.offsets.strides[0]))) != total:
                raise Error("ValueError: invalid support block")
            total += count
        if total != total_support or Int(self.offsets.unsafe_get(classes * Int(self.offsets.strides[0]))) != total:
            raise Error("ValueError: inconsistent total support")


@fieldwise_init
struct WorkingSetState(Movable):
    """LIBSVM-owned mutable state; swap every parallel vector together."""
    var alpha: Dense
    var gradient: Dense
    var gradient_bar: Dense
    var linear_term: Dense
    var diagonal: Dense
    var upper_bound: Dense
    var signs: Indices
    var alpha_status: Indices
    var active_set: Indices
    var active_size: Int
    var unshrunk: Bool


trait KernelColumnCache(Movable):
    """Full columns keyed by original rows; read-only leases survive eviction."""
    def column(mut self, index: Int) raises -> nm.NDArray[DType.float32]:
        ...
