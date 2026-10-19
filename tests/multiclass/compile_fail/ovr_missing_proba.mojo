import numojo as nm
from numojo import Shape
from moxie.multiclass import OneVsRestClassifier
from ovr_fixtures import MarginClassifier


# Must fail: predict_proba exists only when the component provides it.
def main() raises:
    var model = OneVsRestClassifier(MarginClassifier())
    _ = model.predict_proba(nm.zeros[DType.float64](Shape(1, 2)))
