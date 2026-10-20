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
    RegressorMixin,
    TransformerMixin,
    InverseTransformer,
    ProbabilityPredictor,
    LogProbabilityPredictor,
    DecisionFunction,
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
from moxie.utils.validation import check_is_fitted
from ._base import FrozenEstimatorAPI


trait _FrozenIdentity(BaseEstimator):
    def _is_clusterer(self) -> Bool:
        ...


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
    FeatureNamesOut where conforms_to(E, FeatureNamesOut),
    FrozenEstimatorAPI,
    _FrozenIdentity,
    IncrementalClassifier where conforms_to(E, IncrementalClassifier),
    IncrementalRegressor where conforms_to(E, IncrementalRegressor),
    IncrementalTransformer where conforms_to(E, IncrementalTransformer),
    InputEstimator where conforms_to(E, InputEstimator),
    InputFeatureCount where conforms_to(E, InputFeatureCount),
    InverseTransformer where conforms_to(E, InverseTransformer),
    LogProbabilityPredictor where conforms_to(E, LogProbabilityPredictor),
    OutlierMixin where conforms_to(E, OutlierMixin),
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
    See docs/family.md for the native ownership and capability contract.
    """

    comptime Classes = Self.E.Classes if conforms_to(
        Self.E, IncrementalClassifier
    ) else NoneType
    comptime Params = FrozenParams[Self.E]
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
