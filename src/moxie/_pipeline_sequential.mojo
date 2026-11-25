# Adapted from scikit-learn pipeline.py; BSD-3-Clause, see LICENSES/scikit-learn.txt.
"""Typed supervised two-stage composition, using existing estimator capabilities."""
from std.iter import downcast
from moxie._feature_metadata import InputFeatureCount
from moxie.base import (
    BaseEstimator, SupervisedEstimator, UnsupervisedEstimator, Predictor,
    TransformerMixin, FitTransformer, InverseTransformer, SupervisedScorer, clone,
    ProbabilityPredictor, LogProbabilityPredictor, DecisionFunction, SampleScorer,
)
from moxie._pipeline import validate_names
from moxie._pipeline_contracts import PredictivePipeline, TransformingPipeline
from moxie._pipeline_union import UnionFeatureNames, UnionInputFeatures
from moxie.utils.validation import check_is_fitted


@fieldwise_init
struct PipelineParams[P: Copyable & Deinitable, E: Copyable & Deinitable](Copyable, Movable):
    var transformer_name: String
    var estimator_name: String
    var transformer: Self.P
    var estimator: Self.E


def _fit_transform[P: TransformerMixin, T: Copyable & Deinitable](
    mut step: P, X: P.Input, y: T,
) raises -> P.Transformed:
    """Prefer fit_transform; otherwise fit then transform, like sklearn.

    sklearn passes y to every step and unsupervised steps ignore it. An
    unsupervised step whose target type differs from the final estimator's
    (e.g. StandardScaler before SVC) therefore receives y=None.
    """
    comptime if conforms_to(P, FitTransformer):
        comptime if P.Target != T and conforms_to(P, UnsupervisedEstimator):
            var none = Optional[downcast[P.Target, Movable & Deinitable]](None)
            return step.fit_transform(X, rebind[Optional[P.Target]](none))
        else:
            comptime assert P.Target == T, "pipeline target types must match"
            var target = Optional[T](y.copy())
            return step.fit_transform(X, rebind[Optional[P.Target]](target))
    elif conforms_to(P, SupervisedEstimator):
        comptime assert P.Target == T, "pipeline target types must match"
        _ = step.fit(X, rebind[P.Target](y))
        return step.transform(X)
    elif conforms_to(P, UnsupervisedEstimator):
        comptime if P.Target != T:
            var none = Optional[downcast[P.Target, Movable & Deinitable]](None)
            _ = step.fit(X, rebind[Optional[P.Target]](none))
        else:
            var target = Optional[T](y.copy())
            _ = step.fit(X, rebind[Optional[P.Target]](target))
        return step.transform(X)
    else:
        comptime assert False, "pipeline transformer must support fit or fit_transform"
    raise Error("TypeError: pipeline target must be copyable and destructible")


def _validate_pipeline_names(first: String, last: String) raises:
    validate_names([first, last], ["steps", "memory", "verbose", "transform_input"])


def _pipeline_feature_names[P: TransformerMixin, E: BaseEstimator](
    first: P, last: E, first_name: String, last_name: String,
    input_features: Optional[List[String]],
) raises -> List[String]:
    comptime if conforms_to(P, UnionFeatureNames):
        var names = first.get_feature_names_out(input_features)
        comptime if conforms_to(E, UnionFeatureNames):
            return last.get_feature_names_out(Optional[List[String]](names^))
        else:
            raise Error("AttributeError: Estimator " + last_name + " does not provide get_feature_names_out")
    else:
        raise Error("AttributeError: Estimator " + first_name + " does not provide get_feature_names_out")


struct Pipeline[
    P: TransformerMixin & Deinitable,
    E: SupervisedEstimator & Predictor & Deinitable,
](SupervisedEstimator, PredictivePipeline, UnionFeatureNames, UnionInputFeatures,
  SupervisedScorer where conforms_to(E, SupervisedScorer),
  ProbabilityPredictor where conforms_to(E, ProbabilityPredictor),
  LogProbabilityPredictor where conforms_to(E, LogProbabilityPredictor),
  DecisionFunction where conforms_to(E, DecisionFunction),
  SampleScorer where conforms_to(E, SampleScorer),
):
    """A named transformer followed by a supervised predictor.

    Components are move-owned and used directly (memory=None semantics).
    Feature names are delegated through both steps when supported.
    Scoring is conditional on the final estimator's SupervisedScorer capability.
    Probability and decision methods follow the final estimator's capabilities.
    No classification or transform capability is implied.
    """
    comptime Input = Self.P.Input
    comptime Target = Self.E.Target
    comptime Prediction = Self.E.Prediction
    comptime SampleWeight = Self.E.SampleWeight if conforms_to(Self.E, SupervisedScorer) else Int
    comptime Probabilities = Self.E.Probabilities if conforms_to(Self.E, ProbabilityPredictor) else Int
    comptime Decision = Self.E.Decision if conforms_to(Self.E, DecisionFunction) else Int
    comptime SampleScores = Self.E.SampleScores if conforms_to(Self.E, SampleScorer) else Int
    comptime Params = PipelineParams[Self.P.Params, Self.E.Params]
    var transformer_name: String
    var estimator_name: String
    var transformer: Self.P
    var estimator: Self.E

    def __init__(
        out self, var transformer: Self.P, var estimator: Self.E,
        *, transformer_name: String = "transformer", estimator_name: String = "estimator",
    ) raises:
        comptime assert Self.P.Transformed == Self.E.Input, "pipeline intermediate types must match"
        self.transformer_name = transformer_name
        self.estimator_name = estimator_name
        self.transformer = transformer^
        self.estimator = estimator^
        self._validate_names()

    def _validate_names(self) raises:
        _validate_pipeline_names(self.transformer_name, self.estimator_name)

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.transformer_name, self.estimator_name,
                           self.transformer.get_params(deep), self.estimator.get_params(deep))

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        _validate_pipeline_names(params.transformer_name, params.estimator_name)
        self.transformer_name = params.transformer_name
        self.estimator_name = params.estimator_name
        _ = self.transformer.set_params(params.transformer)
        _ = self.estimator.set_params(params.estimator)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(clone(self.transformer), clone(self.estimator),
                    transformer_name=self.transformer_name, estimator_name=self.estimator_name)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.estimator.__sklearn_is_fitted__()

    def n_features_in_(self) raises -> Int:
        """Delegate to the first step without duplicating its learned state."""
        comptime if conforms_to(Self.P, UnionInputFeatures):
            return self.transformer.n_features_in_()
        elif conforms_to(Self.P, InputFeatureCount):
            return self.transformer.get_n_features_in()
        else:
            raise Error("AttributeError: first pipeline step " + self.transformer_name +
                        " does not expose n_features_in_")

    def get_feature_names_out(
        self, input_features: Optional[List[String]] = None,
    ) raises -> List[String]:
        """Pass names through every step; children own validation and fit checks."""
        return _pipeline_feature_names(
            self.transformer, self.estimator, self.transformer_name, self.estimator_name, input_features,
        )

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        self._validate_names()
        comptime if conforms_to(Self.Target, Copyable) and conforms_to(Self.Target, Deinitable) and conforms_to(Self.P.Transformed, Deinitable):
            var transformed = _fit_transform(self.transformer, X, y)
            _ = self.estimator.fit(rebind[Self.E.Input](transformed), y)
        else:
            comptime assert False, "pipeline needs copyable targets and destructible intermediate values"
        return self

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        check_is_fitted(self)
        comptime if conforms_to(Self.P.Transformed, Deinitable):
            var transformed = self.transformer.transform(X)
            return self.estimator.predict(rebind[Self.E.Input](transformed))
        else:
            comptime assert False, "pipeline needs destructible intermediate values"

    def score(
        self, X: Self.Input, y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64 where conforms_to(Self.E, SupervisedScorer):
        check_is_fitted(self)
        comptime if conforms_to(Self.P.Transformed, Deinitable):
            var transformed = self.transformer.transform(X)
            return self.estimator.score(rebind[Self.E.Input](transformed), y,
                                        rebind[Optional[Self.E.SampleWeight]](sample_weight))
        else:
            comptime assert False, "pipeline needs destructible intermediate values"

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities where conforms_to(Self.E, ProbabilityPredictor):
        check_is_fitted(self)
        comptime if conforms_to(Self.P.Transformed, Deinitable):
            var transformed = self.transformer.transform(X)
            return rebind_var[Self.Probabilities](self.estimator.predict_proba(rebind[Self.E.Input](transformed)))
        else:
            comptime assert False, "pipeline needs destructible intermediate values"

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities where conforms_to(Self.E, LogProbabilityPredictor):
        check_is_fitted(self)
        comptime if conforms_to(Self.P.Transformed, Deinitable):
            var transformed = self.transformer.transform(X)
            return rebind_var[Self.Probabilities](self.estimator.predict_log_proba(rebind[Self.E.Input](transformed)))
        else:
            comptime assert False, "pipeline needs destructible intermediate values"

    def decision_function(self, X: Self.Input) raises -> Self.Decision where conforms_to(Self.E, DecisionFunction):
        check_is_fitted(self)
        comptime if conforms_to(Self.P.Transformed, Deinitable):
            var transformed = self.transformer.transform(X)
            return rebind_var[Self.Decision](self.estimator.decision_function(rebind[Self.E.Input](transformed)))
        else:
            comptime assert False, "pipeline needs destructible intermediate values"

    def score_samples(self, X: Self.Input) raises -> Self.SampleScores where conforms_to(Self.E, SampleScorer):
        check_is_fitted(self)
        comptime if conforms_to(Self.P.Transformed, Deinitable):
            var transformed = self.transformer.transform(X)
            return rebind_var[Self.SampleScores](self.estimator.score_samples(rebind[Self.E.Input](transformed)))
        else:
            comptime assert False, "pipeline needs destructible intermediate values"


def _fit_optional_step[P: TransformerMixin & UnsupervisedEstimator, T: Copyable & Deinitable](
    mut step: P, X: P.Input, y: Optional[T],
) raises -> P.Transformed:
    comptime assert P.Target == T, "transformer pipeline target types must match"
    comptime if conforms_to(P, FitTransformer):
        return step.fit_transform(X, rebind[Optional[P.Target]](y))
    else:
        _ = step.fit(X, rebind[Optional[P.Target]](y))
        return step.transform(X)


struct TransformerPipeline[
    P: TransformerMixin & UnsupervisedEstimator & Copyable & Deinitable,
    E: TransformerMixin & UnsupervisedEstimator & Copyable & Deinitable,
](
    TransformingPipeline,
    TransformerMixin,
    UnsupervisedEstimator,
    FitTransformer,
    Copyable,
    UnionFeatureNames,
    InverseTransformer where conforms_to(P, InverseTransformer) and conforms_to(E, InverseTransformer),
):
    """Two named transformer stages, nestable as ordinary transformers."""

    comptime Input = Self.P.Input
    comptime Target = Self.E.Target
    comptime Transformed = Self.E.Transformed
    comptime Params = PipelineParams[Self.P.Params, Self.E.Params]
    var transformer_name: String
    var estimator_name: String
    var transformer: Self.P
    var estimator: Self.E

    def __init__(
        out self, var transformer: Self.P, var estimator: Self.E,
        *, transformer_name: String = "transformer", estimator_name: String = "estimator",
    ) raises:
        comptime assert Self.P.Transformed == Self.E.Input, "pipeline intermediate types must match"
        comptime assert Self.P.Target == Self.E.Target, "transformer pipeline target types must match"
        self.transformer_name = transformer_name
        self.estimator_name = estimator_name
        self.transformer = transformer^
        self.estimator = estimator^
        self._validate_names()

    def _validate_names(self) raises:
        _validate_pipeline_names(self.transformer_name, self.estimator_name)

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.transformer_name, self.estimator_name,
                           self.transformer.get_params(deep), self.estimator.get_params(deep))

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        _validate_pipeline_names(params.transformer_name, params.estimator_name)
        self.transformer_name = params.transformer_name
        self.estimator_name = params.estimator_name
        _ = self.transformer.set_params(params.transformer)
        _ = self.estimator.set_params(params.estimator)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(clone(self.transformer), clone(self.estimator),
                    transformer_name=self.transformer_name, estimator_name=self.estimator_name)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.estimator.__sklearn_is_fitted__()

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        self._validate_names()
        comptime if conforms_to(Self.Target, Copyable & Deinitable) and conforms_to(Self.P.Transformed, Deinitable):
            var transformed = _fit_optional_step(self.transformer, X, y)
            _ = self.estimator.fit(rebind[Self.E.Input](transformed), y)
        else:
            comptime assert False, "pipeline needs copyable targets and destructible intermediate values"
        return self

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self._validate_names()
        comptime if conforms_to(Self.Target, Copyable & Deinitable) and conforms_to(Self.P.Transformed, Deinitable):
            var transformed = _fit_optional_step(self.transformer, X, y)
            return _fit_optional_step(self.estimator, rebind[Self.E.Input](transformed), y)
        else:
            comptime assert False, "pipeline needs copyable targets and destructible intermediate values"

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        check_is_fitted(self)
        comptime if conforms_to(Self.P.Transformed, Deinitable):
            var transformed = self.transformer.transform(X)
            return self.estimator.transform(rebind[Self.E.Input](transformed))
        else:
            comptime assert False, "pipeline needs destructible intermediate values"

    def inverse_transform(
        self, X: Self.Transformed,
    ) raises -> Self.Input where conforms_to(Self.P, InverseTransformer) and conforms_to(Self.E, InverseTransformer):
        """Apply the children’s inverse transforms in reverse order."""
        check_is_fitted(self)
        comptime if conforms_to(Self.E.Input, Deinitable):
            var intermediate = self.estimator.inverse_transform(X)
            return self.transformer.inverse_transform(rebind[Self.P.Transformed](intermediate))
        else:
            comptime assert False, "pipeline needs destructible intermediate values"

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        return _pipeline_feature_names(
            self.transformer, self.estimator, self.transformer_name, self.estimator_name, input_features,
        )
