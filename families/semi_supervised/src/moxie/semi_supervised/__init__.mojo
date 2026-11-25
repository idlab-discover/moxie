"""Dense semi-supervised classifiers and native composition adapters."""
from ._base import SemiSupervisedClassifier
from ._label_propagation import LabelPropagation, LabelSpreading
from ._self_training import SelfTrainingClassifier, GraphClassifierAdapter
from ._gaussian_adapter import GaussianNBAdapter
from ._logistic_adapter import LogisticRegressionAdapter
from ._calibrated_adapter import CalibratedClassifierAdapter
from ._svc_adapter import SVCAdapter
