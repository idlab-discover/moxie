import numojo as nm
from moxie._naive_bayes_dense import prepare_batch, int_classes
from std.testing import assert_equal

def main() raises:
    var X = nm.ones[DType.float64](nm.Shape(2, 1))
    var y: List[Int] = [1, 0]
    var current = List[Int]()
    var classes: List[Int] = [1, 0]
    var labeled = prepare_batch(X, y, current, classes.copy())
    assert_equal(int_classes(labeled.domain)[0], 0)
    assert_equal(int_classes(labeled.domain)[1], 1)
