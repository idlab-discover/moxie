"""Interfaces for the feature_selection family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import (
    InverseTransformer,
    SupervisedEstimator,
    TransformerMixin,
    UnsupervisedEstimator,
)


trait FeatureSelector(InverseTransformer, TransformerMixin):
    comptime Support: Movable

    def get_support(self, indices: Bool = False) raises -> Self.Support:
        ...


trait SupervisedSelector(FeatureSelector, SupervisedEstimator):
    pass


trait UnsupervisedSelector(FeatureSelector, UnsupervisedEstimator):
    pass
