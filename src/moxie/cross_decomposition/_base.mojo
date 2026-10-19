"""Capabilities shared by the dense cross-decomposition estimators."""
from moxie.base import RegressorMixin, SupervisedEstimator, TransformerMixin


trait CrossDecompositionTransformer(SupervisedEstimator, TransformerMixin):
    comptime PairedTransform: Movable

    def transform(
        self, X: Self.Input, y: Self.Target
    ) raises -> Self.PairedTransform:
        ...


trait PLSPredictor(RegressorMixin):
    pass
