"""Interfaces for the compose family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import (
    FitTransformer,
    RegressorMixin,
    TransformerMixin,
    UnsupervisedEstimator,
)


trait ColumnComposer(FitTransformer, TransformerMixin, UnsupervisedEstimator):
    pass


trait TransformedTargetRegressorAPI(RegressorMixin):
    pass
