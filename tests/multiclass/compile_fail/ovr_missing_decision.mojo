import numojo as nm
from numojo import Shape
from moxie.multiclass import OneVsRestClassifier
from ovr_fixtures import ProbaClassifier


# Must fail: decision_function exists only when the component provides it.
def main() raises:
    var model = OneVsRestClassifier(ProbaClassifier())
    _ = model.decision_function(nm.zeros[DType.float64](Shape(1, 2)))
