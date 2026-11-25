"""Coefficient adapter for the native dense linear-kernel SVC."""

import numojo as nm
from moxie.base import (
    ClassifierMixin,
    DecisionFunction,
    LogProbabilityPredictor,
    WeightedSupervisedEstimator,
)
from moxie.svm import SVC
from ._model_importance import SelectionClassLabels


struct SVCImportance[
    Label: Copyable & Comparable & Deinitable = Float64,
    dt: DType = DType.float64,
](
    SelectionClassLabels,
    ClassifierMixin,
    DecisionFunction,
    LogProbabilityPredictor,
    WeightedSupervisedEstimator,
):
    """Expose the linear-kernel SVC's one-vs-one coefficients to selectors."""

    comptime dtype = Self.dt
    comptime Model = SVC[Self.Label]
    comptime Params = Self.Model.Params
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.Model.Target
    comptime Prediction = Self.Model.Prediction
    comptime Classes = Self.Target
    comptime Decision = Self.Model.Decision
    comptime Probabilities = Self.Model.Probabilities
    comptime SampleWeight = Self.Input
    var model: Self.Model

    def __init__(out self, var model: SVC[Self.Label]):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.model = model^

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.model.get_params(deep)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        _ = self.model.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.model.__sklearn_clone__())

    def __sklearn_is_fitted__(self) -> Bool:
        return self.model.__sklearn_is_fitted__()

    @staticmethod
    def _promote(values: Self.Input) raises -> Self.Model.Input:
        # NuMojo astype reads linear offsets, so materialize strided data first.
        if values.is_c_contiguous():
            return values.astype[DType.float64]()
        return values.contiguous().astype[DType.float64]()

    @staticmethod
    def _model_weights(weights: Self.SampleWeight) raises -> Self.Model.SampleWeight:
        comptime if Self.dtype == DType.float64:
            return rebind[Self.Model.SampleWeight](weights).contiguous()
        else:
            return Self._promote(weights)

    @staticmethod
    def _optional_weights(
        weights: Optional[Self.SampleWeight],
    ) raises -> Optional[Self.Model.SampleWeight]:
        if weights:
            return Self._model_weights(weights.value())
        return None

    def _fit(
        mut self, X: Self.Input, y: Self.Target, weights: Optional[Self.Model.SampleWeight]
    ) raises:
        if self.model.kernel != "linear":
            raise Error("ValueError: SVCImportance requires kernel=\"linear\"")
        comptime if Self.dtype == DType.float64:
            _ = self.model.fit(rebind[Self.Model.Input](X), y, weights)
        else:
            _ = self.model.fit(Self._promote(X), y, weights)

    def fit_selection(
        mut self, X: Self.Input, y: Optional[Self.Target], sample_weight: Optional[Self.SampleWeight]
    ) raises:
        if not y:
            raise Error("ValueError: SVC requires y")
        self._fit(X, y.value(), Self._optional_weights(sample_weight))

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target
    ) raises -> ref[self] Self:
        self._fit(X, y, None)
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
        self._fit(X, y, Self._model_weights(sample_weight))
        return self

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        comptime if Self.dtype == DType.float64:
            return self.model.predict(rebind[Self.Model.Input](X))
        else:
            return self.model.predict(Self._promote(X))

    def decision_function(self, X: Self.Input) raises -> Self.Decision:
        comptime if Self.dtype == DType.float64:
            return self.model.decision_function(rebind[Self.Model.Input](X))
        else:
            return self.model.decision_function(Self._promote(X))

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        comptime if Self.dtype == DType.float64:
            return self.model.predict_proba(rebind[Self.Model.Input](X))
        else:
            return self.model.predict_proba(Self._promote(X))

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        comptime if Self.dtype == DType.float64:
            return self.model.predict_log_proba(rebind[Self.Model.Input](X))
        else:
            return self.model.predict_log_proba(Self._promote(X))

    def score(
        self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None
    ) raises -> Float64:
        var weights = Self._optional_weights(sample_weight)
        comptime if Self.dtype == DType.float64:
            return self.model.score(rebind[Self.Model.Input](X), y, weights)
        else:
            return self.model.score(Self._promote(X), y, weights)

    def classes_(self) raises -> Self.Classes:
        _ = self.n_features()
        return self.model.classes_.copy()

    def n_features(self) raises -> Int:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: underlying SVC is not fitted")
        return self.model.n_features_in_

    def feature_importances(self) raises -> Self.Input:
        var coefficients = self.model.coef_()
        comptime if Self.dtype == DType.float64:
            return rebind_var[Self.Input](coefficients^)
        else:
            return coefficients.astype[Self.dtype]()

    def importances(self, getter: String) raises -> Self.Input:
        if getter != "auto" and getter != "coef_":
            raise Error("ValueError: SVC exposes coef_; use a typed adapter for other getters")
        return self.feature_importances()

    def uses_l1_threshold(self) -> Bool:
        return False
