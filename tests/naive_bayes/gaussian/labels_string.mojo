import numojo as nm
from moxie._labels import StringTarget
from moxie._naive_bayes_dense import prepare_batch
from std.testing import assert_equal

def main() raises:
    var X = nm.ones[DType.float64](nm.Shape(2, 1))
    var y = StringTarget(["b", "a"])
    var labeled = prepare_batch(X, y, classes=StringTarget(["b", "a"]))
    assert_equal(labeled.domain.classes()[0], "a")
    assert_equal(labeled.domain.classes()[1], "b")
    assert_equal(labeled.batch.targets.item(0), 1)
