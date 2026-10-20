"""Dense support vector estimators and their capability contracts."""
from ._base import (
    SupportVectorClassifier,
    ProbabilisticSVC,
    SupportVectorRegressor,
    OneClassSupportVector,
)

from ._c_svm import SVC, SVR, SVCParams, SVRParams
from ._nu_svm import NuSVC, NuSVR, NuSVCParams, NuSVRParams
from ._one_class import OneClassSVM, OneClassSVMParams
from ._classification import Gamma, Probability

from ._linear import LinearSVC, LinearSVR, Dual, ClassWeight
from ._bounds import l1_min_c
