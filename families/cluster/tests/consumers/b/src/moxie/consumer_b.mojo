"""Ownership probe, not a clustering estimator."""
import numojo as nm
from moxie.cluster._contracts import snapshot_centers
from moxie._dense import row_squared_norms
from moxie._random import RandomState
from moxie.cluster import AffinityPropagation, KMeans, kmeans_plusplus
from moxie.cluster._contracts import Labels

@fieldwise_init
struct ConsumerB:
    def predict(self, model: AffinityPropagation[DType.float64], x: nm.NDArray[DType.float64]) raises -> Labels:
        return model.predict(x)
    def snapshot(self, x: nm.NDArray[DType.float64]) raises -> nm.NDArray[DType.float64]:
        return snapshot_centers(x)
    def norms(self, x: nm.NDArray[DType.float64]) raises -> nm.NDArray[DType.float64]:
        return row_squared_norms(x)
    def draw(self, mut state: RandomState) -> UInt32:
        return state.next_uint32()
    def predict_kmeans(self, model: KMeans[DType.float64], x: nm.NDArray[DType.float64]) raises -> Labels:
        return model.predict(x)
