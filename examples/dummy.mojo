"""Compare a target against its constant baseline, without using features."""
from moxie import DummyClassifier, DummyRegressor
from moxie.dummy import Label
from numojo import Shape, Item
import numojo as nm

def main() raises:
    var X = nm.zeros[DType.float64](Shape(4, 2))
    var target = nm.zeros[DType.float64](Shape(4))
    for i in range(4):
        target[Item(i)] = Float64(i + 1)
    var regressor = DummyRegressor(strategy="mean")
    regressor.fit(X, target)
    var result = regressor.predict(X, return_std=True)
    print("Mean prediction:", result.prediction.item(0))
    print("Predictive standard deviation:", result.std.value().item(0))
    var labels: List[Label] = [Label("cat"), Label("dog"), Label("cat"), Label("cat")]
    var classifier = DummyClassifier(strategy="prior")
    classifier.fit(X, labels)
    print("Most frequent label:", classifier.predict(X)[0].text)
    print("Prior of first sorted class:", classifier.predict_proba(X).item(0, 0))
    var X32 = X.astype[DType.float32]()
    var target32 = target.astype[DType.float32]()
    var regressor32 = DummyRegressor[dtype=DType.float32]()
    regressor32.fit(X32, target32)
    print("Float32 mean prediction:", regressor32.predict(X32).item(0))
