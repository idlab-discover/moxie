# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Linear and quadratic discriminant analysis (sklearn.discriminant_analysis).

Dense Float32/Float64 NuMojo input. See docs/discriminant_analysis/api.md for Mojo adaptations.
The family traits are re-exported unchanged from the family contracts.
"""
from moxie._discriminant_analysis_contracts import (
    DiscriminantClassifier,
    LinearDiscriminantTransformer,
)
from moxie._discriminant_analysis._lda import (
    LinearDiscriminantAnalysis,
    LinearDiscriminantAnalysisParams,
)
from moxie._discriminant_analysis._qda import (
    QuadraticDiscriminantAnalysis,
    QuadraticDiscriminantAnalysisParams,
)
from moxie._discriminant_analysis._covariance import (
    CovarianceEstimator,
    NoCovarianceEstimator,
    Shrinkage,
)
from moxie._discriminant_analysis._labels import ClassLabel
