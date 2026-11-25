"""Independent wrappers expose only their own learned fields."""
from moxie.multioutput import MultiOutputClassifier, MultiOutputRegressor
from test_estimators import Nearest


def main() raises:
    var regressor = MultiOutputRegressor(Nearest[DType.float64]())
    var classifier = MultiOutputClassifier(Nearest[DType.int64]())
    _ = regressor.classes_
    _ = classifier.order_
