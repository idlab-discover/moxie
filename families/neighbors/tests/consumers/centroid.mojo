"""The same public estimator works in both isolated assembled consumers."""
import numojo as nm
from moxie.neighbors import NearestCentroid
from std.testing import assert_equal


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(4, 1))
    var y = nm.zeros[DType.float64](nm.Shape(4))
    for i in range(4):
        X.unsafe_set(i, Float64(i))
        y.unsafe_set(i, Float64(i // 2))
    var model = NearestCentroid()
    model.fit(X, y)
    assert_equal(model.score(X, y), 1)
    assert_equal(model.predict_proba(X).shape[1], 2)
    print("PASS: assembled NearestCentroid consumer")
