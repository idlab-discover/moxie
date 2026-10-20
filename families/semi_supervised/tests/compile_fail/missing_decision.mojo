from numojo import zeros, Shape
from moxie.semi_supervised import SelfTrainingClassifier, GraphClassifierAdapter, LabelPropagation

def main() raises:
    var adapter = GraphClassifierAdapter(LabelPropagation())
    var model = SelfTrainingClassifier[type_of(adapter)](adapter.copy())
    var X = zeros[DType.float64](Shape(1, 1))
    _ = model.decision_function(X)
