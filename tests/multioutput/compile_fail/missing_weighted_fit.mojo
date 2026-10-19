from test_estimators import Nearest
from moxie.multioutput import MultiOutputRegressor
import numojo as nm

def main() raises:
    var model = MultiOutputRegressor(Nearest[DType.float64]())
    var X = nm.ones[DType.float64](nm.Shape(2, 1))
    var weights = nm.ones[DType.float64](nm.Shape(2))
    _ = model.fit(X, X, sample_weight=weights)
