"""Dense mixture plumbing. Contracts shared by the concrete Gaussian and Bayesian engines.

Behavioral reference: scikit-learn (BSD-3-Clause), see LICENSES/scikit-learn.txt.
"""
from numojo import NDArray, Shape
from std.math import isfinite
from moxie._random import RandomState


def covariance_shape(kind: String, components: Int, features: Int) raises -> Shape:
    if components < 1 or features < 1:
        raise Error("ValueError: positive component and feature counts required")
    if kind == "full":
        return Shape(components, features, features)
    if kind == "tied":
        return Shape(features, features)
    if kind == "diag":
        return Shape(components, features)
    if kind == "spherical":
        return Shape(components)
    raise Error("ValueError: covariance_type must be full, tied, diag or spherical")


def check_shape[dtype: DType](x: NDArray[dtype], expected: Shape) raises:
    if x.ndim != len(expected):
        raise Error("ValueError: array rank does not match mixture layout")
    for axis in range(x.ndim):
        if x.shape[axis] != expected[axis]:
            raise Error("ValueError: array shape does not match mixture layout")


def check_input[dtype: DType](x: NDArray[dtype], components: Int = 1,
                             fitting: Bool = True, features: Int = -1) raises:
    comptime assert dtype == DType.float32 or dtype == DType.float64
    if x.ndim != 2:
        raise Error("ValueError: expected dense rank-2 mixture input")
    if x.shape[1] < 1 or x.shape[0] < (2 if fitting else 1):
        raise Error("ValueError: insufficient samples or features")
    if components < 1 or (fitting and x.shape[0] < components):
        raise Error("ValueError: expected n_samples >= n_components >= 1")
    if features >= 0 and x.shape[1] != features:
        raise Error("ValueError: feature count differs from fitted data")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not isfinite(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1]))):
                raise Error("ValueError: mixture input must be finite")


def snapshot[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    """Independent fitted storage; a strided copy can retain backing-buffer size."""
    comptime assert dtype == DType.float32 or dtype == DType.float64
    return x.copy()


struct EMControl(Copyable, Movable):
    var n_components: Int
    var covariance_type: String
    var tol: Float64
    var reg_covar: Float64
    var max_iter: Int
    var n_init: Int
    var init_params: String
    var warm_start: Bool
    var verbose: Int
    var verbose_interval: Int

    def __init__(out self):
        self.n_components = 1
        self.covariance_type = "full"
        self.tol = 1e-3
        self.reg_covar = 1e-6
        self.max_iter = 100
        self.n_init = 1
        self.init_params = "kmeans"
        self.warm_start = False
        self.verbose = 0
        self.verbose_interval = 10

    def validate(self) raises:
        _ = covariance_shape(self.covariance_type, self.n_components, 1)
        if not isfinite(self.tol) or self.tol < 0 or not isfinite(self.reg_covar) or self.reg_covar < 0:
            raise Error("ValueError: finite nonnegative tol and reg_covar required")
        if self.max_iter < 0 or self.n_init < 1 or self.verbose < 0 or self.verbose_interval < 1:
            raise Error("ValueError: invalid iteration or verbosity control")
        if self.init_params != "kmeans" and self.init_params != "k-means++" and self.init_params != "random" and self.init_params != "random_from_data":
            raise Error("ValueError: unsupported init_params")

    def restart_count(self, fitted: Bool) raises -> Int:
        self.validate()
        return 1 if self.warm_start and fitted else self.n_init


trait DenseMixtureEngine(Movable):
    comptime dtype: DType


trait EMHooks(DenseMixtureEngine):
    """Typed counterpart of BaseMixture's abstract hooks; no default engine."""
    comptime State: Movable & Deinitable

    def initialize(mut self, x: NDArray[Self.dtype], resp: NDArray[Self.dtype]) raises:
        ...

    def estimate_log_prob(self, x: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        ...

    def estimate_log_weights(self) raises -> NDArray[Self.dtype]:
        ...

    def m_step(mut self, x: NDArray[Self.dtype], log_resp: NDArray[Self.dtype]) raises:
        ...

    def lower_bound(self, log_resp: NDArray[Self.dtype], mean_log_prob: Float64) raises -> Float64:
        ...

    def get_state(self) raises -> Self.State:
        ...

    def set_state(mut self, var state: Self.State) raises:
        ...


trait ResponsibilityEngine(DenseMixtureEngine):

    def normalize(self, weighted_log_prob: NDArray[Self.dtype]) raises -> Tuple[NDArray[Self.dtype], NDArray[Self.dtype]]:
        ...


trait PrecisionEngine(DenseMixtureEngine):

    def from_covariances(self, covariance: NDArray[Self.dtype], kind: String) raises -> NDArray[Self.dtype]:
        ...

    def from_precisions(self, precision: NDArray[Self.dtype], kind: String) raises -> NDArray[Self.dtype]:
        ...


trait InitializationEngine(DenseMixtureEngine):

    def responsibilities(self, x: NDArray[Self.dtype], components: Int,
                         method: String, mut rng: RandomState) raises -> NDArray[Self.dtype]:
        ...
