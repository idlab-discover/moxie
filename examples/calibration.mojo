"""Fit calibration on held-out folds and predict probabilities for new rows."""
import numojo as nm
from calibration_centroid import Centroid
from moxie.calibration import CalibratedClassifierCV, CalibrationDisplay


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(30, 2))
    var y = nm.zeros[DType.float64](nm.Shape(30))
    for i in range(30):
        X[nm.Item(i, 0)] = Float64(i % 7)
        X[nm.Item(i, 1)] = Float64(i % 5)
        y[nm.Item(i)] = 1.0 if i % 7 > 3 else 0.0
    var model = CalibratedClassifierCV(Centroid(), method="sigmoid", cv=3)
    _ = model.fit(X, y)
    var query = nm.zeros[DType.float64](nm.Shape(1, 2))
    query[nm.Item(0, 0)] = 5
    query[nm.Item(0, 1)] = 2
    var probabilities = model.predict_proba(query)
    print("P(class 0):", probabilities.item(0, 0))
    print("P(class 1):", probabilities.item(0, 1))

    var X_test = nm.zeros[DType.float64](nm.Shape(14, 2))
    var y_test = nm.zeros[DType.float64](nm.Shape(14))
    for i in range(14):
        X_test[nm.Item(i, 0)] = Float64(i % 7) + 0.1
        X_test[nm.Item(i, 1)] = Float64(i % 5) + 0.2
        y_test[nm.Item(i)] = 1 if i % 7 > 3 else 0
    var display = CalibrationDisplay.from_estimator(model, X_test, y_test)
    display.figure_.value().savefig("builds/reliability.svg")
    print("Saved held-out reliability diagram to builds/reliability.svg")
