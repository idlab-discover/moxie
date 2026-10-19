"""Raising conformance fixture; no numerical engine is implemented."""
from numojo import NDArray, zeros, Shape
from moxie._random import RandomState
from moxie.mixture._contracts import EMHooks, ResponsibilityEngine, PrecisionEngine, InitializationEngine
from std.testing import assert_raises

struct RaisingEngine(EMHooks, ResponsibilityEngine, PrecisionEngine, InitializationEngine):
    comptime dtype = DType.float64
    comptime State = Int

    def __init__(out self):
        pass

    def initialize(mut self, x: NDArray[Self.dtype], resp: NDArray[Self.dtype]) raises:
        raise Error("NotImplementedError: contract probe only")

    def estimate_log_prob(self, x: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        raise Error("NotImplementedError: contract probe only")

    def estimate_log_weights(self) raises -> NDArray[Self.dtype]:
        raise Error("NotImplementedError: contract probe only")

    def m_step(mut self, x: NDArray[Self.dtype], log_resp: NDArray[Self.dtype]) raises:
        raise Error("NotImplementedError: contract probe only")

    def lower_bound(self, log_resp: NDArray[Self.dtype], mean_log_prob: Float64) raises -> Float64:
        raise Error("NotImplementedError: contract probe only")

    def get_state(self) raises -> Self.State:
        raise Error("NotImplementedError: contract probe only")

    def set_state(mut self, var state: Self.State) raises:
        raise Error("NotImplementedError: contract probe only")

    def normalize(self, weighted_log_prob: NDArray[Self.dtype]) raises -> Tuple[NDArray[Self.dtype], NDArray[Self.dtype]]:
        raise Error("NotImplementedError: contract probe only")

    def from_covariances(self, covariance: NDArray[Self.dtype], kind: String) raises -> NDArray[Self.dtype]:
        raise Error("NotImplementedError: contract probe only")

    def from_precisions(self, precision: NDArray[Self.dtype], kind: String) raises -> NDArray[Self.dtype]:
        raise Error("NotImplementedError: contract probe only")

    def responsibilities(self, x: NDArray[Self.dtype], components: Int,
                         method: String, mut rng: RandomState) raises -> NDArray[Self.dtype]:
        raise Error("NotImplementedError: contract probe only")

def dispatch[E: EMHooks & ResponsibilityEngine & PrecisionEngine & InitializationEngine](mut engine: E) raises:
    var x = zeros[E.dtype](Shape(2, 1))
    var rng = RandomState(42)
    with assert_raises():
        engine.initialize(x, x)
    with assert_raises():
        _ = engine.estimate_log_prob(x)
    with assert_raises():
        _ = engine.estimate_log_weights()
    with assert_raises():
        engine.m_step(x, x)
    with assert_raises():
        _ = engine.lower_bound(x, 0)
    with assert_raises():
        _ = engine.get_state()
    with assert_raises():
        _ = engine.normalize(x)
    with assert_raises():
        _ = engine.from_covariances(x, "diag")
    with assert_raises():
        _ = engine.from_precisions(x, "diag")
    with assert_raises():
        _ = engine.responsibilities(x, 1, "random", rng)

def main() raises:
    var probe = RaisingEngine()
    dispatch(probe)
    with assert_raises():
        probe.set_state(0)
    print("PASS typed engine hooks; all numerical methods raise")
