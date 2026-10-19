"""Ownership fixture, not a mixture estimator."""
from numojo import NDArray
from moxie.mixture._contracts import snapshot, check_input
struct ProbeA:
    @staticmethod
    def capture[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
        check_input(x)
        return snapshot(x)
