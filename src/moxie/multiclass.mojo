"""Multiclass meta-estimators following sklearn.multiclass; see docs/multiclass/family.md."""
from moxie._multiclass_contracts import (
    PairwiseInput,
    pairwise_input,
    MulticlassClassifier,
    MulticlassDecision,
    MulticlassProbability,
)
from moxie._multiclass_ovo import OneVsOneClassifier, OneVsOneParams
from moxie._multiclass_ovr import (
    BinaryComponent,
    OneVsRestClassifier,
    OneVsRestParams,
)

from moxie._multiclass_output_code import OutputCodeClassifier, OutputCodeParams

from moxie._multiclass_labels import StringLabels
from moxie._labels import StringTarget
