# Adapted from scikit-learn sklearn/covariance/_elliptic_envelope.py;
# authors: scikit-learn developers. SPDX-License-Identifier: BSD-3-Clause
"""Outlier detection with a robust Gaussian envelope (sklearn `EllipticEnvelope`).

Fits the same MinCovDet state (`RobustFit`) and adds `offset_`, the
`100 * contamination` percentile of the negative robust distances.
"""
import numojo as nm
from moxie.covariance._param_update import NullableUpdate
from numojo import NDArray, Shape
from std.math import floor, isfinite
from moxie._random import RandomState, entropy_state
from moxie.covariance._base import RobustCovarianceDetector
from moxie.covariance._robust_covariance import RobustFit, argsort
from moxie.covariance._empirical_covariance import shape_text


def percentile_linear(values: NDArray[DType.float64], q: Float64) raises -> Float64:
    """np.percentile(values, q) with the default 'linear' method and NumPy's lerp."""
    var n = values.size
    var order = argsort(values)
    var position = Float64(n - 1) * q / 100.0
    var below = Int(floor(position))
    var above = min(below + 1, n - 1)
    var t = position - Float64(below)
    var a = values.unsafe_get(order[below])
    var b = values.unsafe_get(order[above])
    var difference = b - a
    if t >= 0.5:
        return b - difference * (1.0 - t)
    return a + difference * t


@fieldwise_init
struct EllipticEnvelopeParams(Copyable, Movable, Writable):
    var store_precision: Bool
    var assume_centered: Bool
    var support_fraction: Optional[Float64]
    var contamination: Float64
    var random_state: Optional[Int]


struct EllipticEnvelope[dtype: DType = DType.float64](RobustCovarianceDetector, Copyable):
    """Robust-covariance outlier detector (sklearn `EllipticEnvelope`).

    Defaults: store_precision=True, assume_centered=False, support_fraction=None,
    contamination=0.1 (in (0, 0.5]), random_state=None. `predict` returns +1
    for inliers and -1 for outliers (Int64); `score` is (weighted) accuracy.
    MinCovDet attributes are available as accessor methods plus `offset_()`.
    """

    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[DType.int64]
    comptime Prediction = NDArray[DType.int64]
    comptime Decision = NDArray[Self.dtype]
    comptime SampleScores = NDArray[Self.dtype]
    comptime Params = EllipticEnvelopeParams
    comptime NAME = "EllipticEnvelope"

    var store_precision: Bool
    var assume_centered: Bool
    var support_fraction: Optional[Float64]
    var contamination: Float64
    var random_state: Optional[Int]
    var _fit: RobustFit[Self.dtype]
    var _offset: Float64

    def __init__(out self, *, store_precision: Bool = True, assume_centered: Bool = False,
                 support_fraction: Optional[Float64] = None, contamination: Float64 = 0.1, random_state: Optional[Int] = None):
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.contamination = contamination
        self.random_state = random_state
        self._fit = RobustFit[Self.dtype]()
        self._offset = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.store_precision, self.assume_centered, self.support_fraction, self.contamination, self.random_state)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.store_precision = params.store_precision
        self.assume_centered = params.assume_centered
        self.support_fraction = params.support_fraction
        self.contamination = params.contamination
        self.random_state = params.random_state
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *,
        store_precision: Optional[Bool] = None,
        assume_centered: Optional[Bool] = None,
        support_fraction: NullableUpdate[Float64] = NullableUpdate[Float64](),
        contamination: Optional[Float64] = None,
        random_state: NullableUpdate[Int] = NullableUpdate[Int](),
    ) raises -> ref[self] Self:
        """Update supplied constructor parameters without changing fitted state."""
        if store_precision:
            self.store_precision = store_precision.value().copy()
        if assume_centered:
            self.assume_centered = assume_centered.value().copy()
        if support_fraction.supplied:
            self.support_fraction = support_fraction.value.copy()
        if contamination:
            self.contamination = contamination.value().copy()
        if random_state.supplied:
            self.random_state = random_state.value.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(store_precision=self.store_precision, assume_centered=self.assume_centered,
                    support_fraction=self.support_fraction, contamination=self.contamination, random_state=self.random_state)

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fit.model.is_fitted()

    def _check_contamination(self) raises:
        if not (self.contamination > 0.0 and self.contamination <= 0.5):
            raise Error(String("InvalidParameterError: The 'contamination' parameter of EllipticEnvelope must be a float in the range (0.0, 0.5]. Got ",
                               self.contamination, " instead."))

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        var state = RandomState(self.random_state.value()) if self.random_state else entropy_state()
        self.fit(X, state)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, mut random_state: RandomState) raises -> ref[self] Self:
        """Fit drawing from a caller-owned RandomState stream."""
        self._check_contamination()
        self._fit.fit(X, self.store_precision, self.assume_centered, self.support_fraction, random_state, Self.NAME)
        var dist = self._fit.state(Self.NAME).dist_.copy()
        var negative = nm.zeros[DType.float64](Shape(dist.size))
        for k in range(dist.size):
            negative.unsafe_set(k, -Float64(dist.unsafe_get(k)))
        self._offset = percentile_linear(negative, 100.0 * self.contamination)
        return self

    def fit_predict(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Prediction:
        """Fit on X and return +1 inlier / -1 outlier labels; y is ignored."""
        self.fit(X)
        return self.predict(X)

    def fit_predict(mut self, X: Self.Input, mut random_state: RandomState) raises -> Self.Prediction:
        """Fit and label X using a caller-owned random stream."""
        self.fit(X, random_state)
        return self.predict(X)

    def score_samples(self, X: Self.Input) raises -> Self.SampleScores:
        """Negative squared Mahalanobis distances to the robust location."""
        var d = self._fit.model.mahalanobis(X, Self.NAME)
        var result = nm.zeros[Self.dtype](Shape(d.size))
        for k in range(d.size):
            result.unsafe_set(k, -d.unsafe_get(k))
        return result^

    def decision_function(self, X: Self.Input) raises -> Self.Decision:
        """score_samples(X) - offset_; negative values are outliers."""
        var d = self._fit.model.mahalanobis(X, Self.NAME)
        var result = nm.zeros[Self.dtype](Shape(d.size))
        for k in range(d.size):
            result.unsafe_set(k, (-Float64(d.unsafe_get(k)) - self._offset).cast[Self.dtype]())
        return result^

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        var values = self.decision_function(X)
        var labels = nm.zeros[DType.int64](Shape(values.size))
        for k in range(values.size):
            labels.unsafe_set(k, Int64(1) if values.unsafe_get(k) >= 0 else Int64(-1))
        return labels^

    def score(self, X: Self.Input, y: Self.Target, sample_weight: Optional[NDArray[DType.float64]] = None) raises -> Float64:
        """(Weighted) accuracy of predict(X) against labels y (+1 / -1)."""
        var labels = self.predict(X)
        if y.ndim != 1 or y.size != labels.size:
            raise Error(String("ValueError: EllipticEnvelope.score: y shape ", shape_text(y), " does not match ", labels.size, " samples"))
        if sample_weight:
            if sample_weight.value().ndim != 1 or sample_weight.value().size != labels.size:
                raise Error(String("ValueError: EllipticEnvelope.score: sample_weight shape ",
                                   shape_text(sample_weight.value()), " must be a 1D array with ", labels.size, " samples"))
        var total: Float64 = 0
        var correct: Float64 = 0
        var any_nonzero = False
        for k in range(labels.size):
            var weight: Float64 = 1.0
            if sample_weight:
                weight = sample_weight.value().unsafe_get(k * Int(sample_weight.value().strides[0]))
                if not isfinite(weight):
                    raise Error("ValueError: EllipticEnvelope.score: sample_weight contains NaN or infinity")
            any_nonzero = any_nonzero or weight != 0.0
            total += weight
            if labels.unsafe_get(k) == y.unsafe_get(k * Int(y.strides[0])):
                correct += weight
        if not any_nonzero:
            raise Error("ValueError: Sample weights must contain at least one non-zero number.")
        if total == 0.0:
            raise Error("ZeroDivisionError: Weights sum to zero, can't be normalized")
        return correct / total

    def offset_(self) raises -> Float64:
        self._fit.check_fitted(Self.NAME)
        return self._offset

    def mahalanobis(self, X: Self.Input) raises -> NDArray[Self.dtype]:
        return self._fit.model.mahalanobis(X, Self.NAME)

    def correct_covariance(mut self, data: Self.Input) raises -> NDArray[Self.dtype]:
        """Rescale dist_ and return corrected raw covariance without installing it."""
        return self._fit.correct_covariance(data, Self.NAME)

    def reweight_covariance(mut self, data: Self.Input) raises -> Tuple[NDArray[Self.dtype], NDArray[Self.dtype], NDArray[DType.bool]]:
        """Reweight training rows; return (location, unscaled covariance, support)."""
        return self._fit.reweight_covariance(data, self.store_precision, self.assume_centered, Self.NAME)

    def error_norm(self, comp_cov: Self.Input, norm: String = "frobenius", scaling: Bool = True, squared: Bool = True) raises -> Float64:
        """Covariance error with optional feature scaling and squared norm."""
        return self._fit.model.error_norm(comp_cov, Self.NAME, norm, scaling, squared)

    def get_precision(self) raises -> NDArray[Self.dtype]:
        return self._fit.model.get_precision(Self.NAME)

    def location_(self) raises -> NDArray[Self.dtype]:
        return self._fit.model.location(Self.NAME)

    def covariance_(self) raises -> NDArray[Self.dtype]:
        return self._fit.model.covariance(Self.NAME)

    def precision_(self) raises -> Optional[NDArray[Self.dtype]]:
        return self._fit.model.precision(Self.NAME)

    def n_features_in_(self) raises -> Int:
        return self._fit.model.n_features_in(Self.NAME)

    def raw_location_(self) raises -> NDArray[Self.dtype]:
        return self._fit.state(Self.NAME).raw_location_.copy()

    def raw_covariance_(self) raises -> NDArray[Self.dtype]:
        return self._fit.state(Self.NAME).raw_covariance_.copy()

    def raw_support_(self) raises -> NDArray[DType.bool]:
        return self._fit.state(Self.NAME).raw_support_.copy()

    def support_(self) raises -> NDArray[DType.bool]:
        return self._fit.state(Self.NAME).support_.copy()

    def dist_(self) raises -> NDArray[Self.dtype]:
        return self._fit.state(Self.NAME).dist_.copy()
