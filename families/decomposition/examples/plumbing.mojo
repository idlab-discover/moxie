"""Take an independent component snapshot; this does not fit an estimator."""
from numojo import arange, Shape, Item
from moxie.decomposition._contracts import owned_matrix, validate_projection

def main() raises:
    var x = arange[DType.float64](6).reshape(Shape(2, 3))
    var components = owned_matrix(x)
    validate_projection(x, components)
    x[Item(0, 0)] = 99
    print("Independent snapshot retains", components.item(0, 0))
