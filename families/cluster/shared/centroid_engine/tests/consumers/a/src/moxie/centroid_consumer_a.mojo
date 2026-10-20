"""Independent assignment consumer fixture."""
import numojo as nm
from moxie.cluster._contracts import AssignmentResult
from moxie.cluster._centroid_engine import assign

def assign_samples(x: nm.NDArray[DType.float64], centers: nm.NDArray[DType.float64], weights: nm.NDArray[DType.float64]) raises -> AssignmentResult:
    return assign(x, centers, weights)
