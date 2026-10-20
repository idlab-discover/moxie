"""Ownership probe, not a clustering estimator."""
import numojo as nm
from moxie.cluster._contracts import snapshot_centers
from moxie._dense import row_squared_norms
from moxie._random import RandomState
from moxie.cluster import AffinityPropagation, KMeans, kmeans_plusplus

@fieldwise_init
struct ConsumerA:
    def fit(self, x: nm.NDArray[DType.float64]) raises -> AffinityPropagation[DType.float64]:
        var model = AffinityPropagation(random_state=5)
        _ = model.fit(x)
        return model^
    def snapshot(self, x: nm.NDArray[DType.float64]) raises -> nm.NDArray[DType.float64]:
        return snapshot_centers(x)
    def norms(self, x: nm.NDArray[DType.float64]) raises -> nm.NDArray[DType.float64]:
        return row_squared_norms(x)
    def draw(self, mut state: RandomState) -> UInt32:
        return state.next_uint32()
    def fit_kmeans(self, x: nm.NDArray[DType.float64]) raises -> KMeans[DType.float64]:
        var model = KMeans(2, random_state=5)
        _ = model.fit(x)
        return model^
