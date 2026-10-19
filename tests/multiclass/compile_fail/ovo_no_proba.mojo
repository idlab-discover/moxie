import numojo as nm
from numojo import Shape
from moxie.multiclass import OneVsOneClassifier
from ovr_fixtures import MarginProbaClassifier


# Must fail: like sklearn, OneVsOneClassifier never exposes predict_proba.
def main() raises:
    var model = OneVsOneClassifier(MarginProbaClassifier())
    _ = model.predict_proba(nm.zeros[DType.float64](Shape(1, 2)))
