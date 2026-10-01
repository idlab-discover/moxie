"""Share storage explicitly; keep fitted snapshots independent with copy()."""
import numojo as nm
from moxie._dense import row_squared_norms, sum_axis2d


def main() raises:
    var samples = nm.arange[DType.float64](6).reshape(nm.Shape(2, 3))
    var snapshot = samples.copy()
    var shared = samples.view()
    shared[nm.Item(0, 0)] = 10
    print("row squared norms:", row_squared_norms(samples))
    print("original column sums:", sum_axis2d(snapshot, axis=0))
