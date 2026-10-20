from numojo import zeros, Shape
from moxie.semi_supervised import LabelPropagation, SelfTrainingClassifier, GaussianNBAdapter
from moxie._nb_gaussian import GaussianNB

def main() raises:
    var X = zeros[DType.float64](Shape(6, 1))
    X.unsafe_set(0, -1.2)
    X.unsafe_set(1, -0.8)
    X.unsafe_set(2, 0.8)
    X.unsafe_set(3, 1.2)
    X.unsafe_set(4, -0.9)
    X.unsafe_set(5, 0.9)
    var labels = List[Int]([0, 0, 1, 1, -1, -1])
    var graph = LabelPropagation(gamma=2)
    graph.fit(X, labels)
    print("Graph labels:", graph.transduction_)
    var base = GaussianNBAdapter(GaussianNB())
    var self_training = SelfTrainingClassifier[type_of(base)](base.copy())
    self_training.fit(X, labels)
    print("Self-training labels:", self_training.predict(X))
    print("Stopped:", self_training.termination_condition_)
    _ = self_training.predict_log_proba(X)
    print("Accuracy:", self_training.score(X, [0, 0, 1, 1, 0, 1]))
