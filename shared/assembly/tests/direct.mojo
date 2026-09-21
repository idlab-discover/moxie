"""Direct source imports bypass the assembled moxie namespace."""
from moxie._dense import row_squared_norms, sum_axis2d
from numojo import Shape, ones
from std.testing import assert_equal, assert_raises


def probe[dtype: DType]() raises:
    var data = ones[dtype](Shape(2, 3))
    var shared = data.view()
    shared.unsafe_set(0, 2)
    var norms = row_squared_norms(data)
    var sums = sum_axis2d(data, 1)
    with assert_raises():
        _ = sum_axis2d(data, 9)
    assert_equal(Float64(norms.item(0)), 6.0)
    print(norms.item(0), sums.item(0))


def main() raises:
    probe[DType.float32]()
    probe[DType.float64]()
