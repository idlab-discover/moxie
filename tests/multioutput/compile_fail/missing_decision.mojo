from test_estimators import Nearest
from moxie.multioutput import ClassifierChain
import numojo as nm

def main() raises:
    var model = ClassifierChain(Nearest[DType.int64]())
    var X = nm.ones[DType.float64](nm.Shape(2, 1))
    _ = model.decision_function(X)
