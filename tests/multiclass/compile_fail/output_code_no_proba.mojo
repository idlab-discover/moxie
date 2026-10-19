from moxie.multiclass import OutputCodeClassifier
from ovr_fixtures import ProbaClassifier
from numojo import Shape
import numojo as nm


def main() raises:
    var m = OutputCodeClassifier(ProbaClassifier())
    _ = m.predict_proba(nm.zeros[DType.float64](Shape(1, 1)))
