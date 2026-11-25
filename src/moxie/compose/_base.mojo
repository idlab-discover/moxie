"""Interfaces for the compose family contracts; no algorithms are implemented. See docs/family.md."""
from moxie.base import (
    BaseEstimator,
    FitTransformer,
    RegressorMixin,
    TransformerMixin,
    UnsupervisedEstimator,
)


trait ColumnComposer(FitTransformer, TransformerMixin, UnsupervisedEstimator):
    pass


trait NamedColumnTransformers(BaseEstimator):
    """Typed access to the children recorded by a fitted column composition."""

    def named_transformers_[T: TransformerMixin & Copyable & Deinitable](
        self, name: String,
    ) raises -> T:
        ...


trait TransformedTargetRegressorAPI(RegressorMixin):
    pass
