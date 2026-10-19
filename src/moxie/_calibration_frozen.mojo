"""Bridge Moxie's frozen ownership wrapper to the calibration response protocol."""
from .frozen import FrozenEstimator, FrozenParams
from ._calibration_cv import CalibrationEstimator
from ._calibration_algorithms import Array
from .utils.validation import check_is_fitted


struct FrozenCalibrationEstimator[E: CalibrationEstimator](CalibrationEstimator):
    """Preserve a FrozenEstimator's shared fitted handle across calibration folds."""
    comptime Params = FrozenParams[Self.E]
    var estimator: FrozenEstimator[Self.E]

    def __init__(out self, var estimator: FrozenEstimator[Self.E]):
        self.estimator = estimator^

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.estimator.get_params(deep)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        _ = self.estimator.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.estimator.__sklearn_clone__())

    def __sklearn_is_fitted__(self) -> Bool:
        return self.estimator.__sklearn_is_fitted__()

    def is_frozen(self) -> Bool:
        return True

    def fit(mut self, X: Array, y: Array, sample_weight: Optional[Array] = None) raises:
        check_is_fitted(self.estimator)

    def class_labels(self) raises -> Array:
        return self.estimator.estimator().class_labels()

    def response(self, X: Array) raises -> Array:
        return self.estimator.estimator().response(X)

    def uses_probabilities(self) -> Bool:
        return self.estimator.estimator().uses_probabilities()
