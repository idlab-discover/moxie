"""Stable pipeline capability traits, re-exported by moxie.pipeline."""
from moxie.base import BaseEstimator, Predictor, TransformerMixin


trait PipelineEstimator(BaseEstimator):
    pass


trait PredictivePipeline(PipelineEstimator, Predictor):
    pass


trait TransformingPipeline(PipelineEstimator, TransformerMixin):
    pass


trait PipelineSteps(BaseEstimator, Sized):
    """Typed read-only access to the two stored steps of a sequential pipeline."""

    comptime Prefix: BaseEstimator
    comptime Final: BaseEstimator
    comptime StepAt[index: Int] = Self.Prefix if index == 0 or index == -2 else Self.Final

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

    def __len__(self) -> Int:
        return 2

    def step[origin: ImmOrigin, //, index: Int](
        ref[origin] self,
    ) -> ref[origin] Self.StepAt[index]:
        """Borrow the step at a compile-time position."""
        comptime if index == 0 or index == -2:
            return rebind[Pointer[Self.StepAt[index], origin=origin]](
                Pointer(to=self._prefix_step())
            )[]
        else:
            comptime assert index == 1 or index == -1, "pipeline step index out of range"
            return rebind[Pointer[Self.StepAt[index], origin=origin]](
                Pointer(to=self._final_step())
            )[]

    def named_step[origin: ImmOrigin, //, Step: BaseEstimator](
        ref[origin] self,
        name: StringSpan,
    ) raises -> ref[origin] Step:
        """Borrow the step with the given name and concrete type."""
        var index = self._step_index(name)
        comptime if type_of(self._prefix_step()) == Step:
            if index == 0:
                return rebind[Pointer[Step, origin=origin]](
                    Pointer(to=self._prefix_step())
                )[]
        comptime if type_of(self._final_step()) == Step:
            if index == 1:
                return rebind[Pointer[Step, origin=origin]](
                    Pointer(to=self._final_step())
                )[]
        raise Error(t"TypeError: pipeline step '{name}' has a different type")
