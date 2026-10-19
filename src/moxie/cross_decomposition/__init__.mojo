"""Dense partial least squares and canonical correlation estimators."""
from ._base import CrossDecompositionTransformer, PLSPredictor
from ._pls import PLSRegression, PLSCanonical, CCA, PLSEngine, PLSParams
from ._plssvd import PLSSVD, PLSSVDParams
