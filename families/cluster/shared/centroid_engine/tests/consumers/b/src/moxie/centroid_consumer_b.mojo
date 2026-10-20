"""Independent update consumer fixture."""
import numojo as nm
from moxie.cluster._contracts import Labels, CenterUpdate
from moxie.cluster._centroid_engine import update

def update_samples(x: nm.NDArray[DType.float64], labels: Labels, weights: nm.NDArray[DType.float64], centers: nm.NDArray[DType.float64]) raises -> CenterUpdate[DType.float64]:
    return update(x, labels, weights, centers)
