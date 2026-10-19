from moxie.multiclass import OutputCodeClassifier
from ovr_incremental_fixtures import OnlineClassifier
from numojo import Shape
import numojo as nm


def main() raises:
    var model = OutputCodeClassifier(OnlineClassifier())
    _ = model.partial_fit(nm.zeros[DType.float64](Shape(2, 1)), nm.zeros[DType.int64](Shape(2)))
