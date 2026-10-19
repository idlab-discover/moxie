"""Family capability traits for sklearn.multiclass; see docs/multiclass/family.md."""
from moxie.base import ClassifierMixin, DecisionFunction, ProbabilityPredictor


trait PairwiseInput:
    """Family opt-in for sklearn's configuration-dependent input_tags.pairwise."""

    def pairwise_input(self) -> Bool:
        """Whether the current constructor configuration consumes pairwise matrices."""
        ...


def pairwise_input[E: AnyType](estimator: E) -> Bool:
    """Undeclared components have sklearn's default non-pairwise behavior."""
    comptime if conforms_to(E, PairwiseInput):
        return estimator.pairwise_input()
    return False


trait MulticlassClassifier(ClassifierMixin):
    pass


trait MulticlassDecision(DecisionFunction, MulticlassClassifier):
    pass


trait MulticlassProbability(MulticlassClassifier, ProbabilityPredictor):
    pass
