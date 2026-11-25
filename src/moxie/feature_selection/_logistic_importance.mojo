"""Coefficient adapter for the native dense LogisticRegression."""

from std.math import abs
from moxie.base import (
    ClassifierMixin,
    DecisionFunction,
    LogProbabilityPredictor,
    WeightedSupervisedEstimator,
)
from moxie.linear_model import LogisticRegression
from ._model_importance import SelectionClassLabels


struct LogisticRegressionImportance[
    dt: DType = DType.float64,
    label_dtype: DType = DType.int64,
](
    SelectionClassLabels,
    ClassifierMixin,
    DecisionFunction,
    LogProbabilityPredictor,
    WeightedSupervisedEstimator,
):
    """Expose logistic coefficients and classifier capabilities to selectors."""

    comptime dtype = Self.dt
    comptime Model = LogisticRegression[Self.dtype, Self.label_dtype]
    comptime Params = Self.Model.Params
    comptime Input = Self.Model.Input
    comptime Target = Self.Model.Target
    comptime Prediction = Self.Model.Prediction
    comptime Classes = Self.Target
    comptime Decision = Self.Model.Decision
    comptime Probabilities = Self.Model.Probabilities
    comptime SampleWeight = Self.Model.SampleWeight
    var model: Self.Model

    def __init__(out self, var model: LogisticRegression[Self.dt, Self.label_dtype]):
        self.model = model^

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.model.get_params(deep)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        _ = self.model.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.model.__sklearn_clone__())

    def __sklearn_is_fitted__(self) -> Bool:
        return self.model.__sklearn_is_fitted__()

    def fit_selection(
        mut self, X: Self.Input, y: Optional[Self.Target], sample_weight: Optional[Self.SampleWeight],
    ) raises:
        if not y:
            raise Error("ValueError: LogisticRegression requires y")
        ref target = y.value()
        if sample_weight:
            _ = self.fit(X, target, sample_weight=sample_weight.value())
        else:
            _ = self.fit(X, target)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        _ = self.model.fit(X, y)
        return self

    def fit[
        origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        *,
        ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        _ = self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        return self.model.predict(X)

    def decision_function(self, X: Self.Input) raises -> Self.Decision:
        return self.model.decision_function(X)

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return self.model.predict_proba(X)

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return self.model.predict_log_proba(X)

    def score(
        self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        return self.model.score(X, y, sample_weight)

    def classes_(self) raises -> Self.Classes:
        _ = self.n_features()
        return self.model.classes_.copy()

    def n_features(self) raises -> Int:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: underlying LogisticRegression is not fitted")
        return self.model.n_features_in_

    def feature_importances(self) raises -> Self.Input:
        _ = self.n_features()
        return self.model.coef_.copy()

    def importances(self, getter: String) raises -> Self.Input:
        if getter != "auto" and getter != "coef_":
            raise Error("ValueError: LogisticRegression exposes coef_; use a typed adapter for other getters")
        return self.feature_importances()

    def uses_l1_threshold(self) -> Bool:
        # sklearn checks the configured penalty and ratio when choosing a threshold.
        return (
            self.model.penalty.or_else(String()) == "l1"
            or abs(self.model.l1_ratio.or_else(0) - 1) <= 1.001e-5
        )
