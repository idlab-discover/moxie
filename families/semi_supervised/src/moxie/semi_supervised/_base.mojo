"""Capability contract for dense semi-supervised classifiers. See docs/family.md."""
from moxie.base import ClassifierMixin, ProbabilityPredictor


trait SemiSupervisedClassifier(ClassifierMixin, ProbabilityPredictor):
    pass
