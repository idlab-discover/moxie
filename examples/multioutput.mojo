"""Independent and chained regression with Moxie's production DummyRegressor."""
import numojo as nm
from moxie.dummy import DummyRegressor
from moxie.multioutput import MultiOutputRegressor, RegressorChain
from std.testing import assert_equal


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(3, 1))
    var y = nm.zeros[DType.float64](nm.Shape(3, 2))
    for i in range(3):
        y.unsafe_set(2 * i, Float64(i))
        y.unsafe_set(2 * i + 1, Float64(i + 2))
    var model = MultiOutputRegressor(DummyRegressor())
    var prediction = model.fit(X, y).predict(X)
    assert_equal(Float64(prediction.unsafe_get(0)), 1.0)
    assert_equal(Float64(prediction.unsafe_get(1)), 3.0)
    var weights = nm.ones[DType.float64](nm.Shape(3))
    _ = model.fit(X, y, sample_weight=weights)
    var chain = RegressorChain(DummyRegressor(), random_order=True, random_state=7)
    var chained = chain.fit(X, y).predict(X)
    assert_equal(Float64(chained.unsafe_get(0)), 1.0)
    assert_equal(Float64(chained.unsafe_get(1)), 3.0)
    _ = chain.fit(X, y, sample_weight=weights)
    print("Independent and chain predictions:", prediction, chained)
