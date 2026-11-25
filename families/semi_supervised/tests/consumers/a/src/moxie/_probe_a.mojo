"""Producer fixture, not an estimator implementation."""
from numojo import NDArray, Shape, zeros
from std.testing import assert_equal
from moxie._calibration_cv import CalibratedClassifierCV
from moxie.semi_supervised._calibrated_adapter import CalibratedClassifierAdapter
from moxie.semi_supervised._contracts import snapshot_features

def produce() raises -> NDArray[DType.float64]:
    var caller = zeros[DType.float64](Shape(2, 2))
    caller.unsafe_set(0, 3)
    var owned = snapshot_features(caller, 2)
    caller.unsafe_set(0, 99)
    return owned^


def check_default_calibration() raises:
    var features = zeros[DType.float64](Shape(12, 1))
    var labels = List[Int](capacity=12)
    for i in range(12):
        features.unsafe_set(i, Float64(i) - 5.5)
        labels.append(-2 if i < 6 else 9)
    var adapter = CalibratedClassifierAdapter(CalibratedClassifierCV())
    adapter.fit(features, labels)
    assert_equal(adapter.predict(features), labels)
