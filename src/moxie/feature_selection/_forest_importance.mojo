"""Feature selection over native dense RandomForest and ExtraTrees estimators."""

import numojo as nm
from moxie.base import (
    ClassifierMixin,
    LogProbabilityPredictor,
    Predictor,
    SupervisedScorer,
    RegressorMixin,
    WeightedSupervisedEstimator,
)
from moxie.ensemble._forest import Forest
from ._model_importance import SelectionClassLabels, SelectionEstimator


struct ForestImportance[
    classifier: Bool,
    extra: Bool = False,
    label_dtype: DType = DType.float64,
    string_labels: Bool = False,
    dt: DType = DType.float32,
](
    SelectionEstimator,
    SelectionClassLabels where classifier,
    ClassifierMixin where classifier,
    LogProbabilityPredictor where classifier,
    RegressorMixin where not classifier,
    Predictor,
    SupervisedScorer,
    WeightedSupervisedEstimator,
):
    """Adapt a single-output classifier or a regressor to dense selectors."""

    comptime dtype = Self.dt
    comptime Model = Forest[
        Self.classifier, Self.extra, False, Self.label_dtype, Self.string_labels, Self.dt
    ]
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.Model.Target
    comptime Prediction = Self.Model.Prediction
    comptime SampleWeight = Self.Input
    comptime Probabilities = Self.Model.Probabilities
    comptime Classes = Self.Model.Classes
    comptime Params = Self.Model.Params

    var model: Self.Model

    def __init__(
        out self,
        var model: Forest[
            Self.classifier, Self.extra, False, Self.label_dtype, Self.string_labels, Self.dt
        ],
    ):
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

    @staticmethod
    def _weights(weights: Self.SampleWeight) raises -> Self.Model.SampleWeight:
        comptime if Self.dtype == DType.float64:
            return rebind[Self.Model.SampleWeight](weights).copy()
        else:
            if weights.is_c_contiguous():
                return weights.astype[DType.float64]()
            return weights.contiguous().astype[DType.float64]()

    def fit_selection(
        mut self, X: Self.Input, y: Optional[Self.Target],
        sample_weight: Optional[Self.SampleWeight],
    ) raises:
        if not y:
            raise Error("ValueError: forest selection requires y")
        if sample_weight:
            _ = self.fit(X, y.value(), sample_weight=sample_weight.value())
        else:
            _ = self.fit(X, y.value())

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        _ = self.model.fit(X, y)
        return self

    def fit[
        origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin,
        sample_weight_origin: ImmOrigin,
    ](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target, *,
        ref[sample_weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        _ = self.model.fit(X, y, Optional[Self.Model.SampleWeight](Self._weights(sample_weight)))
        return self

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        return self.model.predict(X)

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities where Self.classifier:
        return self.model.predict_proba(X)

    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities where Self.classifier:
        return self.model.predict_log_proba(X)

    def score(
        self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        var weights: Optional[Self.Model.SampleWeight] = None
        if sample_weight:
            weights = Self._weights(sample_weight.value())
        return self.model.score(X, y, weights)

    def classes_(self) raises -> Self.Classes where Self.classifier:
        _ = self.n_features()
        return self.model.classes_.copy()

    def n_features(self) raises -> Int:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: underlying forest is not fitted")
        return self.model.n_features_in_

    def feature_importances(self) raises -> Self.Input:
        _ = self.n_features()
        comptime if Self.dtype == DType.float64:
            return rebind[Self.Input](self.model.feature_importances_).copy()
        else:
            return self.model.feature_importances_.astype[Self.dtype]()

    def uses_l1_threshold(self) -> Bool:
        return False

    def allows_nan(self) -> Bool:
        return True
