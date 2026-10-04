# Adapted from scikit-learn pipeline.py; BSD-3-Clause, see LICENSES/scikit-learn.txt.
"""Typed supervised two-stage composition, using existing estimator capabilities."""
from moxie._feature_metadata import InputFeatureCount
from moxie.base import (
    BaseEstimator, SupervisedEstimator, UnsupervisedEstimator, Predictor,
    TransformerMixin, FitTransformer, SupervisedScorer, clone,
    ProbabilityPredictor, LogProbabilityPredictor, DecisionFunction, SampleScorer,
)
from moxie._pipeline import validate_names
from moxie._pipeline_contracts import PredictivePipeline
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
    """Prefer fit_transform; otherwise fit then transform, like sklearn."""
    comptime if conforms_to(P, FitTransformer):
        comptime assert P.Target == T, "pipeline target types must match"
        var target = Optional[T](y.copy())
        return step.fit_transform(X, rebind[Optional[P.Target]](target))
    elif conforms_to(P, SupervisedEstimator):
        comptime assert P.Target == T, "pipeline target types must match"
        _ = step.fit(X, rebind[P.Target](y))
        return step.transform(X)
    elif conforms_to(P, UnsupervisedEstimator):
        comptime assert P.Target == T, "pipeline target types must match"
        var target = Optional[T](y.copy())
        _ = step.fit(X, rebind[Optional[P.Target]](target))
        return step.transform(X)
    else:
        comptime assert False, "pipeline transformer must support fit or fit_transform"
    raise Error("TypeError: pipeline target must be copyable and destructible")


struct Pipeline[
    P: TransformerMixin & Deinitable,
    E: SupervisedEstimator & Predictor & Deinitable,
](SupervisedEstimator, PredictivePipeline, UnionFeatureNames, UnionInputFeatures,
  SupervisedScorer where conforms_to(E, SupervisedScorer),
  ProbabilityPredictor where conforms_to(E, ProbabilityPredictor),
  LogProbabilityPredictor where conforms_to(E, LogProbabilityPredictor),
  DecisionFunction where conforms_to(E, DecisionFunction),
  SampleScorer where conforms_to(E, SampleScorer),
) where conforms_to(P.Params, Deinitable) and conforms_to(E.Params, Deinitable):
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
        validate_names([self.transformer_name, self.estimator_name],
                       ["steps", "memory", "verbose", "transform_input"])

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
        validate_names([params.transformer_name, params.estimator_name],
                       ["steps", "memory", "verbose", "transform_input"])
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
        comptime if conforms_to(Self.P, UnionFeatureNames):
            var names = self.transformer.get_feature_names_out(input_features)
            comptime if conforms_to(Self.E, UnionFeatureNames):
                return self.estimator.get_feature_names_out(Optional[List[String]](names^))
            else:
                raise Error("AttributeError: Estimator " + self.estimator_name +
                            " does not provide get_feature_names_out")
        else:
            raise Error("AttributeError: Estimator " + self.transformer_name +
                        " does not provide get_feature_names_out")

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
