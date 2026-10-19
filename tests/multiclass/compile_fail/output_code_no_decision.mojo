from moxie.multiclass import OutputCodeClassifier
from ovr_fixtures import MarginClassifier
from numojo import Shape
import numojo as nm


def main() raises:
    var m = OutputCodeClassifier(MarginClassifier())
    _ = m.decision_function(nm.zeros[DType.float64](Shape(1, 1)))
