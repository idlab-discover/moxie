"""Stable pipeline capability traits, re-exported by moxie.pipeline."""
from moxie.base import BaseEstimator, Predictor, TransformerMixin


trait PipelineEstimator(BaseEstimator):
    pass


trait PredictivePipeline(PipelineEstimator, Predictor):
    pass


trait TransformingPipeline(PipelineEstimator, TransformerMixin):
    pass


trait PipelineSteps(BaseEstimator):
    """Typed read-only access to the two stored steps of a sequential pipeline."""

    comptime Prefix: BaseEstimator
    comptime Final: BaseEstimator

    def _step_index(self, name: StringSpan) raises -> Int:
        ...

    def _prefix_step[origin: ImmOrigin](
        ref[origin] self,
    ) -> ref[origin] Self.Prefix:
        ...

    def _final_step[origin: ImmOrigin](
        ref[origin] self,
    ) -> ref[origin] Self.Final:
        ...
