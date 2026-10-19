from test_estimators import WeightedMean
from moxie.multioutput import MultiOutputRegressor
import numojo as nm

def main() raises:
    var model = MultiOutputRegressor(WeightedMean())
    var X = nm.ones[DType.float64](nm.Shape(2, 1))
    _ = model.partial_fit(X, X)
