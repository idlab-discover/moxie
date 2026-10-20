"""Ownership probe, not a clustering estimator."""
import numojo as nm
from moxie.cluster._contracts import snapshot_centers
from moxie._dense import row_squared_norms
from moxie._random import RandomState

@fieldwise_init
struct ConsumerA:
    def snapshot(self, x: nm.NDArray[DType.float64]) raises -> nm.NDArray[DType.float64]:
        return snapshot_centers(x)
    def norms(self, x: nm.NDArray[DType.float64]) raises -> nm.NDArray[DType.float64]:
        return row_squared_norms(x)
    def draw(self, mut state: RandomState) -> UInt32:
        return state.next_uint32()
