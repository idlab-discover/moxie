"""Create and validate owned centers without pretending to fit a model."""
import numojo as nm
from moxie.cluster._contracts import snapshot_centers, validate_labels
from moxie._dense import row_squared_norms


def main() raises:
    var centers = nm.zeros[DType.float64](nm.Shape(2, 2))
    centers[nm.Item(0, 0)] = 3
    centers[nm.Item(0, 1)] = 4
    var owned = snapshot_centers(centers)
    var squared_norms = row_squared_norms(owned)
    var labels = nm.zeros[DType.int64](nm.Shape(3))
    labels[nm.Item(2)] = 1
    validate_labels(labels, 3, 2)
    print("First center squared norm:", squared_norms.item(0))
    print("Three labels validated against two centers; no clustering was run.")
