"""Freeze an owned estimator and share its fitted state across clones."""
from std.memory import ArcPointer
from moxie.base import (
    BaseEstimator,
    InputEstimator,
    SupervisedEstimator,
    UnsupervisedEstimator,
    Predictor,
    SupervisedScorer,
    ClassifierMixin,
    NumericClassifierClasses,
    RegressorMixin,
    TransformerMixin,
    InverseTransformer,
    ProbabilityPredictor,
    LogProbabilityPredictor,
    DecisionFunction,
    EstimatorTags,
    EstimatorTagsProvider,
    get_tags,
    SampleScorer,
    DensityMixin,
    OutlierMixin,
    WeightedSupervisedEstimator,
    IncrementalClassifier,
    IncrementalRegressor,
    IncrementalTransformer,
    ClusterMixin,
)
from moxie._feature_metadata import FeatureNamesOut, InputFeatureCount
from moxie._pipeline_contracts import PipelineSteps
from moxie.compose._base import NamedColumnTransformers
from moxie.metadata_routing import (
    FitScoreMetadataRequest,
    FitScoreMetadataRequestProvider,
)
from moxie.utils.validation import check_is_fitted
from ._base import FrozenEstimatorAPI


trait _FrozenIdentity(BaseEstimator):
    def _is_clusterer(self) -> Bool:
        ...


comptime _SliceBound[bound: Int] = (
    2 if bound >= 2 else (1 if bound == -1 else (0 if bound < 0 else bound))
)


struct FrozenPipelineSlice[
    Parent: PipelineSteps & Copyable & Deinitable,
    start: Int,
    stop: Int,
](Sized):
    """An immutable view of a contiguous range of frozen pipeline steps."""

    comptime _size = Self.stop - Self.start if Self.stop > Self.start else 0
    comptime Selected = Self.Parent if Self._size == 2 else Self.Parent.StepAt[Self.start]
    comptime Input where conforms_to(Self.Selected, InputEstimator) = Self.Selected.Input
    comptime Transformed where conforms_to(Self.Selected, TransformerMixin) = Self.Selected.Transformed
    comptime Prediction where conforms_to(Self.Selected, Predictor) = Self.Selected.Prediction
    comptime Probabilities where conforms_to(Self.Selected, ProbabilityPredictor) = Self.Selected.Probabilities

    var _parent: Self.Parent

    def __init__(out self, var parent: Self.Parent):
        comptime assert 0 <= Self.start and Self.start <= 2, "invalid pipeline slice start"
        comptime assert 0 <= Self.stop and Self.stop <= 2, "invalid pipeline slice stop"
        self._parent = parent^

    def __len__(self) -> Int:
        return Self._size

    def step[origin: ImmOrigin, //, index: Int](ref[origin] self) -> ref[self._parent] Self.Parent.StepAt[
        Self.start + (index + Self._size if index < 0 else index)
    ]:
        """Borrow the selected step at a position relative to this slice."""
        comptime position = index + Self._size if index < 0 else index
        comptime assert 0 <= position and position < Self._size, "pipeline slice step index out of range"
        return self._parent.step[Self.start + position]()

    def named_step[origin: ImmOrigin, //, Step: BaseEstimator](
        ref[origin] self, name: StringSpan,
    ) raises -> ref[self._parent] Step:
        """Borrow a step by its original name and concrete type."""
        var index = self._parent._step_index(name)
        if index < Self.start or index >= Self.stop:
            raise Error(t"KeyError: no pipeline slice step named '{name}'")
        return self._parent.named_step[Step](name)

    def estimator[origin: ImmOrigin](ref[origin] self) -> ref[origin] Self.Selected where Self._size > 0:
        """Borrow the estimator that executes this slice."""
        comptime if Self._size == 2:
            return rebind[ImmPointer[Self.Selected, origin]](Pointer(to=self._parent))[]
        else:
            return rebind[ImmPointer[Self.Selected, origin]](
                Pointer(to=self._parent.step[Self.start]())
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


@fieldwise_init
struct FrozenParams[E: BaseEstimator & Deinitable](Copyable, Movable):
    """Only the estimator handle; deep traversal never exposes inner params."""

    var _estimator: ArcPointer[Self.E]

    def estimator(self) -> ref[self._estimator] Self.E:
        """Borrow the wrapped estimator without allowing mutation."""
        return self._estimator[]


struct FrozenEstimator[E: BaseEstimator & Deinitable](
    ClassifierMixin where conforms_to(E, ClassifierMixin),
    Copyable,
    DecisionFunction where conforms_to(E, DecisionFunction),
    DensityMixin where conforms_to(E, DensityMixin),
    EstimatorTagsProvider where conforms_to(E, EstimatorTagsProvider),
    FeatureNamesOut where conforms_to(E, FeatureNamesOut),
    FitScoreMetadataRequestProvider where conforms_to(
        E, FitScoreMetadataRequestProvider
    ),
    FrozenEstimatorAPI,
    _FrozenIdentity,
    IncrementalClassifier where conforms_to(E, IncrementalClassifier),
    IncrementalRegressor where conforms_to(E, IncrementalRegressor),
    IncrementalTransformer where conforms_to(E, IncrementalTransformer),
    InputEstimator where conforms_to(E, InputEstimator),
    InputFeatureCount where conforms_to(E, InputFeatureCount),
    InverseTransformer where conforms_to(E, InverseTransformer),
    LogProbabilityPredictor where conforms_to(E, LogProbabilityPredictor),
    NamedColumnTransformers where conforms_to(E, NamedColumnTransformers),
    NumericClassifierClasses where conforms_to(E, NumericClassifierClasses),
    OutlierMixin where conforms_to(E, OutlierMixin),
    PipelineSteps where conforms_to(E, PipelineSteps),
    Predictor where conforms_to(E, Predictor),
    ProbabilityPredictor where conforms_to(E, ProbabilityPredictor),
    RegressorMixin where conforms_to(E, RegressorMixin),
    SampleScorer where conforms_to(E, SampleScorer),
    SupervisedEstimator where conforms_to(E, SupervisedEstimator),
    SupervisedScorer where conforms_to(E, SupervisedScorer),
    TransformerMixin where conforms_to(E, TransformerMixin),
    UnsupervisedEstimator where conforms_to(E, UnsupervisedEstimator),
    WeightedSupervisedEstimator where conforms_to(
        E, WeightedSupervisedEstimator
    ),
):
    """No-op fit, preserved fitted clone, and conditional inference forwarding.

    Transfer an estimator with `FrozenEstimator(model^)`. Clones and parameter
    records share ownership. Explicit partial_fit updates that shared state.
    Replacement is local to this wrapper.
    See docs/frozen/api.md for the native ownership and capability contract.
    """

    comptime Classes = Self.E.Classes if conforms_to(
        Self.E, IncrementalClassifier
    ) else NoneType
    comptime Params = FrozenParams[Self.E]
    comptime Prefix where conforms_to(Self.E, PipelineSteps) = Self.E.Prefix
    comptime Final where conforms_to(Self.E, PipelineSteps) = Self.E.Final
    comptime ClassLabelDType where conforms_to(
        Self.E, NumericClassifierClasses
    ) = Self.E.ClassLabelDType
    comptime Input where conforms_to(Self.E, InputEstimator) = Self.E.Input
    comptime Target = Self.E.Target if conforms_to(
        Self.E, SupervisedEstimator
    ) else (
        Self.E.Target if conforms_to(
            Self.E, UnsupervisedEstimator
        ) else NoneType
    )
    comptime Prediction where conforms_to(Self.E, Predictor) = Self.E.Prediction
    comptime SampleWeight = Self.E.SampleWeight if conforms_to(
        Self.E, SupervisedScorer
    ) else (
        Self.E.SampleWeight if conforms_to(
            Self.E, WeightedSupervisedEstimator
        ) else NoneType
    )
    comptime Transformed where conforms_to(
        Self.E, TransformerMixin
    ) = Self.E.Transformed
    comptime Probabilities where conforms_to(
        Self.E, ProbabilityPredictor
    ) = Self.E.Probabilities
    comptime Decision where conforms_to(
        Self.E, DecisionFunction
    ) = Self.E.Decision
    comptime SampleScores where conforms_to(
        Self.E, SampleScorer
    ) = Self.E.SampleScores

    var _estimator: ArcPointer[Self.E]

    def estimator(self) -> ref[self._estimator] Self.E:
        """Borrow the wrapped estimator without allowing mutation."""
        return self._estimator[]

    # The wrapper's borrowed ArcPointer keeps the pipeline and its steps alive.
    def _prefix_step[origin: ImmOrigin](
        ref[origin] self,
    ) -> ref[origin] Self.Prefix where conforms_to(Self.E, PipelineSteps):
        return Pointer(to=self._estimator[]._prefix_step()).unsafe_origin_cast[origin]()[]

    def _final_step[origin: ImmOrigin](
        ref[origin] self,
    ) -> ref[origin] Self.Final where conforms_to(Self.E, PipelineSteps):
        return Pointer(to=self._estimator[]._final_step()).unsafe_origin_cast[origin]()[]

    def _step_index(self, name: StringSpan) raises -> Int where conforms_to(
        Self.E, PipelineSteps
    ):
        return self._estimator[]._step_index(name)

    def slice[start: Int = 0, stop: Int = 2, stride: Int = 1](
        self,
    ) -> FrozenPipelineSlice[
        Self, _SliceBound[start], _SliceBound[stop]
    ] where conforms_to(Self.E, PipelineSteps):
        """Share a read-only half-open range with Python-style bound clipping."""
        comptime assert stride == 1, "pipeline slicing only supports a step of 1"
        return FrozenPipelineSlice[
            Self, _SliceBound[start], _SliceBound[stop]
        ](self.copy())

    def named_transformers_[T: TransformerMixin & Copyable & Deinitable](
        self, name: String,
    ) raises -> T where conforms_to(Self.E, NamedColumnTransformers):
        """Return a fitted column transformer child using Compose's copy semantics."""
        return self._estimator[].named_transformers_[T](name)

    def __init__(out self, var estimator: Self.E):
        self._estimator = ArcPointer(estimator^)

    def __init__(out self, params: Self.Params):
        self._estimator = params._estimator.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self._estimator.copy())

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self._estimator = params._estimator.copy()
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
        *,
        var estimator: Self.E,
    ) raises -> ref[self] Self:
        self._estimator = ArcPointer(estimator^)
        return self

    def set_params[
        origin: MutOrigin,
    ](
        ref[origin] self,
    ) raises -> ref[self] Self:
        return self

    def __sklearn_clone__(self) raises -> Self:
        return self.copy()

    def _is_clusterer(self) -> Bool:
        return is_clusterer(self._estimator[])

    def __sklearn_is_fitted__(self) -> Bool:
        return self._estimator[].__sklearn_is_fitted__()

    def __sklearn_tags__(self) -> EstimatorTags where conforms_to(
        Self.E, EstimatorTagsProvider
    ):
        """Preserve the child's tags and opt the wrapper out of estimator checks."""
        var tags = get_tags(self._estimator[])
        tags._skip_test = True
        return tags

    def get_metadata_routing(self) -> FitScoreMetadataRequest where conforms_to(
        Self.E, FitScoreMetadataRequestProvider
    ):
        """Return the child's fit and score sample-weight requests."""
        return self._estimator[].get_metadata_routing()

    def get_classes(self) raises -> Self.Target where conforms_to(
        Self.E, NumericClassifierClasses
    ):
        return rebind_var[Self.Target](self._estimator[].get_classes())

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self where conforms_to(Self.E, SupervisedEstimator):
        check_is_fitted(self)
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self where conforms_to(Self.E, UnsupervisedEstimator):
        check_is_fitted(self)
        return self

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        *,
        ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self where conforms_to(Self.E, WeightedSupervisedEstimator):
        check_is_fitted(self)
        return self

    def fit[
        Features: AnyType,
        Targets: AnyType,
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Features,
        ref[y_origin] y: Targets,
    ) raises -> ref[self] Self:
        """The reference ignores even inputs outside the estimator's input type.
        """
        check_is_fitted(self)
        return self

    def partial_fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self where conforms_to(Self.E, IncrementalRegressor):
        _ = self._estimator[].partial_fit(X, rebind[Self.E.Target](y))
        return self

    def partial_fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        classes: Optional[Self.Classes] = None,
    ) raises -> ref[self] Self where conforms_to(Self.E, IncrementalClassifier):
        _ = self._estimator[].partial_fit(
            X,
            rebind[Self.E.Target](y),
            rebind[Optional[Self.E.Classes]](classes),
        )
        return self

    def partial_fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self where conforms_to(Self.E, IncrementalTransformer):
        _ = self._estimator[].partial_fit(X, rebind[Optional[Self.E.Target]](y))
        return self

    def predict(
        self, X: Self.Input
    ) raises -> Self.Prediction where conforms_to(Self.E, Predictor):
        return self._estimator[].predict(X)

    def score(
        self,
        X: Self.Input,
        y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64 where conforms_to(Self.E, SupervisedScorer):
        # Under this constraint the conditional aliases name identical types.
        # rebind bridges the compiler's differently written type expressions.
        return self._estimator[].score(
            X,
            rebind[Self.E.Target](y),
            rebind[Optional[Self.E.SampleWeight]](sample_weight),
        )

    def score(
        self, X: Self.Input, y: Optional[Self.Target] = None
    ) raises -> Float64 where conforms_to(Self.E, DensityMixin):
        return self._estimator[].score(X, rebind[Optional[Self.E.Target]](y))

    def predict_proba(
        self, X: Self.Input
    ) raises -> Self.Probabilities where conforms_to(
        Self.E, ProbabilityPredictor
    ):
        return self._estimator[].predict_proba(X)

    def predict_log_proba(
        self, X: Self.Input
    ) raises -> Self.Probabilities where conforms_to(
        Self.E, LogProbabilityPredictor
    ):
        return self._estimator[].predict_log_proba(X)

    def decision_function(
        self, X: Self.Input
    ) raises -> Self.Decision where conforms_to(Self.E, DecisionFunction):
        return self._estimator[].decision_function(X)

    def transform(
        self, X: Self.Input
    ) raises -> Self.Transformed where conforms_to(Self.E, TransformerMixin):
        return self._estimator[].transform(X)

    def inverse_transform(
        self, X: Self.Transformed
    ) raises -> Self.Input where conforms_to(Self.E, InverseTransformer):
        return self._estimator[].inverse_transform(X)

    def score_samples(
        self, X: Self.Input
    ) raises -> Self.SampleScores where conforms_to(Self.E, SampleScorer):
        return self._estimator[].score_samples(X)

    def get_feature_names_out(
        self, input_features: Optional[List[String]] = None
    ) raises -> List[String] where conforms_to(Self.E, FeatureNamesOut):
        return self._estimator[].get_feature_names_out(input_features)

    def get_n_features_in(
        self,
    ) raises -> Int where conforms_to(Self.E, InputFeatureCount):
        return self._estimator[].get_n_features_in()


def is_clusterer[E: BaseEstimator](estimator: E) -> Bool:
    """Query cluster identity through frozen wrappers, without fit_predict."""
    comptime if conforms_to(E, _FrozenIdentity):
        return estimator._is_clusterer()
    else:
        return conforms_to(E, ClusterMixin)
