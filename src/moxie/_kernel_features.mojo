"""Dense feature-map validation and ownership; no approximation engines."""
from numojo import NDArray
from std.math import isfinite
from moxie.utils.validation import FittedState


def validate_matrix[dtype: DType](x: NDArray[dtype], nonnegative: Bool = False) raises:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if x.ndim != 2:
        raise Error("ValueError: expected a dense rank-2 matrix")
    if x.shape[0] <= 0 or x.shape[1] <= 0:
        raise Error("ValueError: expected nonempty samples and features")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            var value = x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]))
            if not isfinite(value):
                raise Error("ValueError: input must be finite")
            if nonnegative and value < 0:
                raise Error("ValueError: input must be nonnegative")


def owned_snapshot[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    validate_matrix(x)
    return x.copy()


def component_width(n_components: Int) raises -> Int:
    if n_components < 1:
        raise Error("ValueError: n_components must be positive")
    return n_components


def additive_width(n_features: Int, sample_steps: Int) raises -> Int:
    if n_features < 1 or sample_steps < 1:
        raise Error("ValueError: features and sample_steps must be positive")
    # Guard native signed integer arithmetic before multiplication.
    var limit = Int.MAX
    if sample_steps > limit // 2 + 1:
        raise Error("ValueError: feature count overflow")
    var blocks = 2 * (sample_steps - 1) + 1
    if n_features > limit // blocks:
        raise Error("ValueError: feature count overflow")
    return n_features * blocks


def basis_width(n_samples: Int, n_components: Int) raises -> Int:
    var requested = component_width(n_components)
    if n_samples < 1:
        raise Error("ValueError: training samples must be positive")
    return min(n_samples, requested)


struct FeatureMapState(Copyable, Movable):
    """Optional composition for maps requiring fit; not for stateless additive maps."""
    var input_state: FittedState
    var n_features_out_: Int

    def __init__(out self):
        self.input_state = FittedState()
        self.n_features_out_ = 0

    def record_fit[dtype: DType](mut self, x: NDArray[dtype], width: Int) raises:
        validate_matrix(x)
        var checked_width = component_width(width)
        self.input_state.record_fit(x.shape[0], x.shape[1])
        self.n_features_out_ = checked_width

    def check_transform[dtype: DType](self, x: NDArray[dtype]) raises:
        if not self.input_state.is_fitted():
            raise Error("NotFittedError: call fit before transform")
        validate_matrix(x)
        self.input_state.check_features(x.shape[1])

    def reset(mut self):
        self.input_state.reset()
        self.n_features_out_ = 0
