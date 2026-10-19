"""Prepare independent dense blocks; this example does not fit an estimator."""
import numojo as nm
from moxie.cross_decomposition._contracts import owned_blocks


def main() raises:
    var x = nm.ones[DType.float64](nm.Shape(4, 3))
    var y = nm.ones[DType.float64](nm.Shape(4, 1))
    var blocks = owned_blocks(x, y, 2, regression=True)
    print("Validated owned workspaces:", blocks.x.shape[0], "rows")
    print("PLSEngine and all four estimators are owned by Moxie.")
