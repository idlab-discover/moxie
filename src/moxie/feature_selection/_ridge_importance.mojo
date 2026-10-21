"""Feature-selection adapter for the real dense Ridge estimator."""
import numojo as nm
from moxie.linear_model import Ridge
from ._model_importance import SelectionEstimator
from moxie.base import RegressorMixin

struct RidgeImportance[dt: DType = DType.float64](SelectionEstimator, RegressorMixin):
    """Ridge coefficients as importances; sklearn's default threshold is the mean."""
    comptime dtype = Self.dt
    comptime Target = nm.NDArray[Self.dtype]
    comptime Input = Self.Target
    comptime Prediction = Self.Target
    comptime SampleWeight = Self.Target
    comptime Params = Ridge[Self.dtype].Params
    var model: Ridge[Self.dtype]

    def __init__(out self, var model: Ridge[Self.dtype]):
        self.model = model^

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.model.get_params(deep)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        _ = self.model.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.model.__sklearn_clone__())

    def __sklearn_is_fitted__(self) -> Bool:
        return self.model.__sklearn_is_fitted__()

    def fit_selection(mut self, X: nm.NDArray[Self.dtype], y: Optional[Self.Target], sample_weight: Optional[nm.NDArray[Self.dtype]]) raises:
        if not y:
            raise Error("ValueError: Ridge requires y")
        if sample_weight:
            _ = self.model.fit(X, y.value(), sample_weight=sample_weight.value())
        else:
            _ = self.model.fit(X, y.value())

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        _ = self.model.fit(X, y)
        return self

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        return self.model.predict(X)

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[Self.SampleWeight] = None) raises -> Float64:
        return self.model.score(X, y, sample_weight)

    def n_features(self) raises -> Int:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: underlying Ridge is not fitted")
        return self.model.n_features_in_

    def feature_importances(self) raises -> nm.NDArray[Self.dtype]:
        _ = self.n_features()
        return self.model.coef_.copy()

    def importances(self, getter: String) raises -> nm.NDArray[Self.dtype]:
        if getter != "auto" and getter != "coef_":
            raise Error("ValueError: Ridge exposes coef_; use a typed adapter for other getters")
        return self.feature_importances()

    def uses_l1_threshold(self) -> Bool:
        return False
