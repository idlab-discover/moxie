"""Stable pipeline capability traits, re-exported by moxie.pipeline."""
from moxie.base import (
    BaseEstimator, InputEstimator, Predictor, ProbabilityPredictor, TransformerMixin,
)
from moxie._feature_metadata import FeatureNamesOut


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

    def _step_name(self, index: Int) -> String:
        ...

    def _step_index(self, name: StringSpan) raises -> Int:
        # A duplicate introduced by a public field edit resolves to the last step.
        if name == self._step_name(1):
            return 1
        if name == self._step_name(0):
            return 0
        raise Error(t"KeyError: no pipeline step named '{name}'")

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

    def named_steps[origin: ImmOrigin](
        ref[origin] self,
    ) -> NamedPipelineSteps[Self, origin]:
        """Borrow a name lookup view over the stored steps."""
        return NamedPipelineSteps[Self, origin](Pointer(to=self))

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


@fieldwise_init
struct NamedPipelineSteps[Parent: PipelineSteps, parent_origin: ImmOrigin](Sized):
    """Read-only, typed name lookup tied to the owning pipeline."""

    var _parent: Pointer[Self.Parent, origin=Self.parent_origin]

    def __len__(self) -> Int:
        return 1 if self._parent[]._step_name(0) == self._parent[]._step_name(1) else 2

    def __contains__(self, name: StringSpan) -> Bool:
        return name == self._parent[]._step_name(0) or name == self._parent[]._step_name(1)

    def keys(self) -> List[String]:
        """Return the unique step names in insertion order."""
        var first = self._parent[]._step_name(0)
        var last = self._parent[]._step_name(1)
        if first == last:
            return [first^]
        return [first^, last^]

    def get[Step: BaseEstimator](self, name: StringSpan) raises -> ref[Self.parent_origin] Step:
        """Borrow the step of the specified concrete type."""
        return self._parent[].named_step[Step](name)


comptime _SliceBound[bound: Int] = (
    2 if bound >= 2 else (1 if bound == -1 else (0 if bound < 0 else bound))
)


trait SliceablePipeline(PipelineSteps):
    """A sequential pipeline whose steps can be borrowed as a range."""

    def slice[origin: ImmOrigin, //, start: Int = 0, stop: Int = 2, stride: Int = 1](
        ref[origin] self,
    ) -> PipelineSlice[Self, origin, _SliceBound[start], _SliceBound[stop]]:
        """Borrow a read-only half-open range of immediate pipeline steps."""
        comptime assert stride == 1, "pipeline slicing only supports a step of 1"
        return PipelineSlice[Self, origin, _SliceBound[start], _SliceBound[stop]](
            Pointer(to=self)
        )


struct PipelineSlice[
    Parent: PipelineSteps,
    parent_origin: ImmOrigin,
    start: Int,
    stop: Int,
](Sized):
    """An immutable view tied to the lifetime of its source pipeline."""

    comptime _size = Self.stop - Self.start if Self.stop > Self.start else 0
    comptime Selected = Self.Parent if Self._size == 2 else Self.Parent.StepAt[Self.start]
    comptime Input where conforms_to(Self.Selected, InputEstimator) = Self.Selected.Input
    comptime Transformed where conforms_to(Self.Selected, TransformerMixin) = Self.Selected.Transformed
    comptime Prediction where conforms_to(Self.Selected, Predictor) = Self.Selected.Prediction
    comptime Probabilities where conforms_to(Self.Selected, ProbabilityPredictor) = Self.Selected.Probabilities

    var _parent: Pointer[Self.Parent, origin=Self.parent_origin]

    def __init__(out self, parent: Pointer[Self.Parent, origin=Self.parent_origin]):
        comptime assert 0 <= Self.start and Self.start <= 2, "invalid pipeline slice start"
        comptime assert 0 <= Self.stop and Self.stop <= 2, "invalid pipeline slice stop"
        self._parent = parent

    def __len__(self) -> Int:
        return Self._size

    def step[index: Int](self) -> ref[Self.parent_origin] Self.Parent.StepAt[
        Self.start + (index + Self._size if index < 0 else index)
    ]:
        """Borrow a stored step at a position relative to this slice."""
        comptime position = index + Self._size if index < 0 else index
        comptime assert 0 <= position and position < Self._size, "pipeline slice step index out of range"
        return self._parent[].step[Self.start + position]()

    def named_step[Step: BaseEstimator](self, name: StringSpan) raises -> ref[Self.parent_origin] Step:
        """Borrow a stored step by its original name and type."""
        var index = self._parent[]._step_index(name)
        if index < Self.start or index >= Self.stop:
            raise Error(t"KeyError: no pipeline slice step named '{name}'")
        return self._parent[].named_step[Step](name)

    def estimator(self) -> ref[Self.parent_origin] Self.Selected where Self._size > 0:
        """Borrow the pipeline or single step that executes the selected range."""
        comptime if Self._size == 2:
            return rebind[Pointer[Self.Selected, origin=Self.parent_origin]](self._parent)[]
        else:
            return rebind[Pointer[Self.Selected, origin=Self.parent_origin]](
                Pointer(to=self.step[0]())
            )[]

    def __sklearn_is_fitted__(self) -> Bool:
        comptime if Self._size > 0:
            return self.estimator().__sklearn_is_fitted__()
        else:
            return True

    def transform(
        self, X: Self.Input,
    ) raises -> Self.Transformed where Self._size > 0 and conforms_to(Self.Selected, TransformerMixin):
        return self.estimator().transform(X)

    def predict(
        self, X: Self.Input,
    ) raises -> Self.Prediction where Self._size > 0 and conforms_to(Self.Selected, Predictor):
        return self.estimator().predict(X)

    def predict_proba(
        self, X: Self.Input,
    ) raises -> Self.Probabilities where Self._size > 0 and conforms_to(Self.Selected, ProbabilityPredictor):
        return self.estimator().predict_proba(X)


    def get_feature_names_out(
        self, input_features: Optional[List[String]] = None,
    ) raises -> List[String] where Self._size > 0 and conforms_to(Self.Selected, FeatureNamesOut):
        return self.estimator().get_feature_names_out(input_features)
