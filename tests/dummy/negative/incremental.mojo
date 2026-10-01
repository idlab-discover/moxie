from moxie.dummy import DummyClassifier
from numojo import Shape
import numojo as nm

def main() raises:
    var model = DummyClassifier()
    var X = nm.zeros[DType.float64](Shape(2, 1))
    model.partial_fit(X, X)
