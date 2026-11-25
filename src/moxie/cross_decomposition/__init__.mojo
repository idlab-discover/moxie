"""Dense partial least squares and canonical correlation estimators."""
from ._base import CrossDecompositionTransformer, PairedTableScores, PLSPredictor
from ._pls import PLSRegression, PLSCanonical, CCA, PLSEngine, PLSMetadataRequest, PLSParams
from ._plssvd import PLSSVD, PLSSVDParams
from ._serialization import (
    CCASchema,
    PLSCanonicalSchema,
    PLSRegressionSchema,
    PLSSVDSchema,
)
