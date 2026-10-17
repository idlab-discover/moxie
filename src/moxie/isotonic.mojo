"""Pure Mojo weighted isotonic regression and linear interpolation.

Float32/Float64 list interface; see docs/isotonic/api.md for the supported contract.
Behavior follows sklearn.isotonic (BSD-3-Clause; see docs/isotonic/THIRD_PARTY_NOTICES).
"""
from std.math import isfinite, isnan
from std.utils import Variant
from moxie.base import (
    FitTransformer,
    RegressorMixin,
    TransformerMixin,
    WeightedSupervisedEstimator,
)
from moxie.utils.validation import check_is_fitted


trait IsotonicRegressorAPI(FitTransformer, RegressorMixin, TransformerMixin):
    pass


comptime Increasing = Variant[Bool, String]


@fieldwise_init
struct IsotonicRegressionParams(Copyable, Movable):
    var y_min: Optional[Float64]
    var y_max: Optional[Float64]
    var increasing: Increasing
    var out_of_bounds: String


def _validate[dtype: DType = DType.float64](values: List[Scalar[dtype]], name: String) raises:
    if len(values) == 0:
        raise Error("ValueError: " + name + " must contain at least one sample")
    for value in values:
        if not isfinite(value):
            raise Error(
                "ValueError: " + name + " must contain only finite values"
            )


def _weights[dtype: DType = DType.float64](n: Int, weights: Optional[List[Scalar[dtype]]]) raises -> List[Scalar[dtype]]:
    if weights:
        if len(weights.value()) != n:
            raise Error("ValueError: inconsistent number of sample weights")
        _validate(weights.value(), "sample_weight")
        return weights.value().copy()
    var result = List[Scalar[dtype]]()
    for _ in range(n):
        result.append(1.0)
    return result^


def _order[dtype: DType = DType.float64](X: List[Scalar[dtype]], y: List[Scalar[dtype]]) -> List[Int]:
    """Stable bottom-up merge sort of indices by (X, y), in O(n log n)."""
    var order = List[Int]()
    for i in range(len(X)):
        order.append(i)
    var scratch = order.copy()
    var width = 1
    while width < len(X):
        var start = 0
        while start < len(X):
            var middle = min(start + width, len(X))
            var end = min(start + 2 * width, len(X))
            var left = start
            var right = middle
            for out_index in range(start, end):
                var take_left = right == end
                if left < middle and right < end:
                    var a = order[left]
                    var b = order[right]
                    take_left = X[a] < X[b] or (X[a] == X[b] and y[a] <= y[b])
                if left < middle and take_left:
                    scratch[out_index] = order[left]
                    left += 1
                else:
                    scratch[out_index] = order[right]
                    right += 1
            start += 2 * width
        swap(order, scratch)
        width *= 2
    return order^


def _ranks[dtype: DType = DType.float64](values: List[Scalar[dtype]]) -> List[Float64]:
    var order = _order(values, values)
    var ranks = List[Float64]()
    for _ in range(len(values)):
        ranks.append(0.0)
    var start = 0
    while start < len(values):
        var end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        var rank = Float64(start + end - 1) / 2.0
        for i in range(start, end):
            ranks[order[i]] = rank
        start = end
    return ranks^


def check_increasing[dtype: DType = DType.float64](x: List[Scalar[dtype]], y: List[Scalar[dtype]]) raises -> Bool:
    """Return the sign of Spearman's rank correlation, with average tied ranks.

    Constant inputs (undefined correlation) return False. Python warning
    emission is not implemented; see the explicit differences in docs/isotonic/api.md.
    """
    _validate(x, "x")
    _validate(y, "y")
    if len(x) != len(y):
        raise Error("ValueError: inconsistent numbers of samples")
    var rx = _ranks(x)
    var ry = _ranks(y)
    var center = Float64(len(x) - 1) / 2.0
    var covariance: Float64 = 0
    var variance_x: Float64 = 0
    var variance_y: Float64 = 0
    for i in range(len(x)):
        var dx = rx[i] - center
        var dy = ry[i] - center
        covariance += dx * dy
        variance_x += dx * dx
        variance_y += dy * dy
    return variance_x > 0 and variance_y > 0 and covariance >= 0


def isotonic_regression(
    y: List[Float64],
    *,
    sample_weight: Optional[List[Float64]] = None,
    y_min: Optional[Float64] = None,
    y_max: Optional[Float64] = None,
    increasing: Bool = True,
) raises -> List[Float64]:
    """Weighted least-squares isotonic fit of already ordered observations.

    Weights must be strictly positive. Inputs are never modified. PAVA uses
    O(n) time and storage. Bounds clip the solution after pooling.
    """
    _validate(y, "y")
    if (y_min and isnan(y_min.value())) or (y_max and isnan(y_max.value())):
        raise Error("ValueError: bounds must not be NaN")
    var weights = _weights[DType.float64](len(y), sample_weight)
    for weight in weights:
        if weight <= 0:
            raise Error(
                "ValueError: isotonic_regression weights must be positive"
            )
    var means = List[Float64]()
    var masses = List[Float64]()
    var starts = List[Int]()
    for i in range(len(y)):
        var j = i if increasing else len(y) - 1 - i
        means.append(y[j])
        masses.append(weights[j])
        starts.append(i)
        while len(means) > 1:
            var last = len(means) - 1
            if means[last - 1] < means[last]:
                break
            var mass = masses[last - 1] + masses[last]
            means[last - 1] = (
                means[last - 1] * masses[last - 1] + means[last] * masses[last]
            ) / mass
            masses[last - 1] = mass
            _ = means.pop()
            _ = masses.pop()
            _ = starts.pop()
    var result = y.copy()
    for block in range(len(means)):
        var end = starts[block + 1] if block + 1 < len(means) else len(y)
        var value = means[block]
        if y_min:
            value = max(value, y_min.value())
        if y_max:
            value = min(value, y_max.value())
        for i in range(starts[block], end):
            var j = i if increasing else len(y) - 1 - i
            result[j] = value
    return result^


def _column[dtype: DType = DType.float64](X: List[List[Scalar[dtype]]]) raises -> List[Scalar[dtype]]:
    var result = List[Scalar[dtype]]()
    for row in X:
        if len(row) != 1:
            raise Error(
                "ValueError: X should be a 1d array or 2d array with 1 feature"
            )
        result.append(row[0])
    return result^


def isotonic_regression[dtype: DType](
    y: List[Scalar[dtype]],
    *,
    sample_weight: Optional[List[Scalar[dtype]]] = None,
    y_min: Optional[Float64] = None,
    y_max: Optional[Float64] = None,
    increasing: Bool = True,
) raises -> List[Scalar[dtype]]:
    """Scalar[dtype] output with Float64 PAVA, matching the SciPy-backed reference."""
    var wide_y = List[Float64]()
    for value in y:
        wide_y.append(Float64(value))
    var wide_w: Optional[List[Float64]] = None
    if sample_weight:
        var weights = List[Float64]()
        for value in sample_weight.value():
            weights.append(Float64(value))
        wide_w = weights^
    var fitted = isotonic_regression(wide_y, sample_weight=wide_w, increasing=increasing)
    var result = List[Scalar[dtype]]()
    if (y_min and isnan(y_min.value())) or (y_max and isnan(y_max.value())):
        raise Error("ValueError: bounds must not be NaN")
    for value in fitted:
        var rounded = Scalar[dtype](value)
        if y_min:
            rounded = max(rounded, Scalar[dtype](y_min.value()))
        if y_max:
            rounded = min(rounded, Scalar[dtype](y_max.value()))
        result.append(rounded)
    return result^


struct IsotonicRegression[dtype: DType = DType.float64](
    Copyable, IsotonicRegressorAPI, WeightedSupervisedEstimator
):
    """Monotonic regression with weighted predictor ties and linear prediction.

    Constructor configuration is separate from learned thresholds. fit commits
    learned state only after validation and fitting both succeed.
    """

    comptime Params = IsotonicRegressionParams
    comptime Input = List[Scalar[Self.dtype]]
    comptime Target = List[Scalar[Self.dtype]]
    comptime Prediction = List[Scalar[Self.dtype]]
    comptime Transformed = List[Scalar[Self.dtype]]
    comptime SampleWeight = List[Scalar[Self.dtype]]

    var y_min: Optional[Float64]
    var y_max: Optional[Float64]
    var increasing: Increasing
    var out_of_bounds: String
    var X_min_: Scalar[Self.dtype]
    var X_max_: Scalar[Self.dtype]
    var X_thresholds_: List[Scalar[Self.dtype]]
    var y_thresholds_: List[Scalar[Self.dtype]]
    var increasing_: Bool
    var _fitted: Bool
    var _bounds_error: Bool

    def __init__(
        out self,
        *,
        y_min: Optional[Float64] = None,
        y_max: Optional[Float64] = None,
        var increasing: Increasing = True,
        var out_of_bounds: String = "nan",
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing^
        self.out_of_bounds = out_of_bounds^
        self.X_min_ = 0
        self.X_max_ = 0
        self.X_thresholds_ = List[Scalar[Self.dtype]]()
        self.y_thresholds_ = List[Scalar[Self.dtype]]()
        self.increasing_ = True
        self._fitted = False
        self._bounds_error = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(
            self.y_min,
            self.y_max,
            self.increasing.copy(),
            self.out_of_bounds.copy(),
        )

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.y_min = params.y_min
        self.y_max = params.y_max
        self.increasing = params.increasing.copy()
        self.out_of_bounds = params.out_of_bounds.copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self,
        *,
        y_min: Optional[Float64] = None,
        y_max: Optional[Float64] = None,
        increasing: Optional[Increasing] = None,
        out_of_bounds: Optional[String] = None,
    ) raises -> ref[self] Self:
        """Update supplied parameters; use the Params record to clear a bound.
        """
        if y_min:
            self.y_min = y_min
        if y_max:
            self.y_max = y_max
        if increasing:
            self.increasing = increasing.value().copy()
        if out_of_bounds:
            self.out_of_bounds = out_of_bounds.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(
            y_min=self.y_min,
            y_max=self.y_max,
            increasing=self.increasing.copy(),
            out_of_bounds=self.out_of_bounds.copy(),
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        self._fit(X, y, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin, weight_origin: ImmOrigin](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        *,
        ref[weight_origin] sample_weight: Self.SampleWeight,
    ) raises -> ref[self] Self:
        self._fit(X, y, sample_weight.copy())
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self,
        ref[x_origin] X: List[List[Scalar[Self.dtype]]],
        ref[y_origin] y: Self.Target,
        *,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> ref[self] Self:
        self._fit(_column(X), y, sample_weight)
        return self

    def _fit(
        mut self,
        X: Self.Input,
        y: Self.Target,
        sample_weight: Optional[Self.SampleWeight],
    ) raises:
        _validate(X, "X")
        _validate(y, "y")
        if len(X) != len(y):
            raise Error("ValueError: inconsistent numbers of samples")
        if (
            self.out_of_bounds != "nan"
            and self.out_of_bounds != "clip"
            and self.out_of_bounds != "raise"
        ):
            raise Error("ValueError: out_of_bounds must be nan, clip, or raise")
        var direction: Bool
        if self.increasing.isa[Bool]():
            direction = self.increasing[Bool]
        else:
            if self.increasing[String] != "auto":
                raise Error("ValueError: increasing must be a Bool or auto")
            # sklearn infers direction before filtering nonpositive weights.
            direction = check_increasing(X, y)
        var weights = _weights[Self.dtype](len(X), sample_weight)
        var order = _order(X, y)
        var unique_x = List[Scalar[Self.dtype]]()
        var unique_y = List[Scalar[Self.dtype]]()
        var unique_w = List[Scalar[Self.dtype]]()
        var sum_y: Scalar[Self.dtype] = 0
        var sum_w: Scalar[Self.dtype] = 0
        var current_x: Scalar[Self.dtype] = 0
        for i in order:
            if weights[i] <= 0:
                continue
            if sum_w == 0:
                current_x = X[i]
            elif X[i] - current_x >= Scalar[Self.dtype](1e-6 if Self.dtype == DType.float32 else 1e-15):
                # Match numpy.finfo(dtype).resolution in sklearn._make_unique.
                unique_x.append(current_x)
                unique_y.append(sum_y / sum_w)
                unique_w.append(sum_w)
                current_x = X[i]
                sum_y = 0
                sum_w = 0
            sum_y += y[i] * weights[i]
            sum_w += weights[i]
        if sum_w == 0:
            raise Error(
                "ValueError: at least one sample must have positive weight"
            )
        unique_x.append(current_x)
        unique_y.append(sum_y / sum_w)
        unique_w.append(sum_w)
        var fitted = isotonic_regression(
            unique_y,
            sample_weight=unique_w^,
            increasing=direction,
            y_min=self.y_min,
            y_max=self.y_max,
        )
        var thresholds_x = List[Scalar[Self.dtype]]()
        var thresholds_y = List[Scalar[Self.dtype]]()
        for i in range(len(unique_x)):
            if i > 0 and i + 1 < len(unique_x):
                if fitted[i - 1] == fitted[i] and fitted[i] == fitted[i + 1]:
                    continue
            thresholds_x.append(unique_x[i])
            thresholds_y.append(fitted[i])
        self.X_min_ = unique_x[0]
        self.X_max_ = unique_x[len(unique_x) - 1]
        self.X_thresholds_ = thresholds_x^
        self.y_thresholds_ = thresholds_y^
        self.increasing_ = direction
        self._bounds_error = self.out_of_bounds == "raise"
        self._fitted = True

    def _interpolate(self, x: Scalar[Self.dtype]) raises -> Scalar[Self.dtype]:
        var n = len(self.X_thresholds_)
        # sklearn uses a constant callable for a single threshold, even for OOB.
        if n == 1:
            return self.y_thresholds_[0]
        if x < self.X_min_ or x > self.X_max_:
            if self._bounds_error:
                raise Error(
                    "ValueError: input is outside the interpolation range"
                )
            return Scalar[Self.dtype](Float64("nan"))
        var left = 0
        var right = n - 1
        while left + 1 < right:
            var middle = left + (right - left) // 2
            if self.X_thresholds_[middle] <= x:
                left = middle
            else:
                right = middle
        if x == self.X_thresholds_[left]:
            return self.y_thresholds_[left]
        if x == self.X_thresholds_[right]:
            return self.y_thresholds_[right]
        var slope = (self.y_thresholds_[right] - self.y_thresholds_[left]) / (
            self.X_thresholds_[right] - self.X_thresholds_[left]
        )
        return self.y_thresholds_[left] + slope * (x - self.X_thresholds_[left])

    def f_(self, T: Self.Input) raises -> Self.Prediction:
        """Evaluate the fitted interpolator directly (without input clipping).
        """
        check_is_fitted(self)
        _validate(T, "T")
        var result = List[Scalar[Self.dtype]]()
        for x in T:
            result.append(self._interpolate(x))
        return result^

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        check_is_fitted(self)
        _validate(X, "X")
        var result = List[Scalar[Self.dtype]]()
        for value in X:
            var x = value
            if self.out_of_bounds == "clip":
                x = min(max(x, self.X_min_), self.X_max_)
            result.append(self._interpolate(x))
        return result^

    def predict(self, X: List[List[Scalar[Self.dtype]]]) raises -> Self.Prediction:
        return self.predict(_column(X))

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        return self.predict(X)

    def transform(self, X: List[List[Scalar[Self.dtype]]]) raises -> Self.Transformed:
        return self.predict(X)

    def fit_transform(
        mut self,
        X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> Self.Transformed:
        if not y:
            raise Error("ValueError: IsotonicRegression requires y")
        _ = self.fit(X, y.value())
        return self.transform(X)

    def fit_transform(
        mut self,
        X: Self.Input,
        y: Self.Target,
        *,
        sample_weight: Self.SampleWeight,
    ) raises -> Self.Transformed:
        _ = self.fit(X, y, sample_weight=sample_weight)
        return self.transform(X)

    def fit_transform(
        mut self,
        X: List[List[Scalar[Self.dtype]]],
        y: Self.Target,
        *,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Self.Transformed:
        self._fit(_column(X), y, sample_weight)
        return self.transform(X)

    def score(
        self,
        X: Self.Input,
        y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        var predicted = self.predict(X)
        _validate(y, "y")
        _validate(predicted, "predictions")
        if len(y) != len(predicted):
            raise Error("ValueError: inconsistent numbers of samples")
        var weights = _weights[Self.dtype](len(y), sample_weight)
        if len(y) < 2:
            return Float64("nan")
        var total_weight: Float64 = 0
        var total: Float64 = 0
        for i in range(len(y)):
            total_weight += Float64(weights[i])
            total += Float64(weights[i]) * Float64(y[i])
        if total_weight == 0:
            raise Error("ValueError: score weights must not sum to zero")
        var mean = total / total_weight
        var numerator: Float64 = 0
        var denominator: Float64 = 0
        for i in range(len(y)):
            numerator += Float64(weights[i]) * Float64(y[i] - predicted[i]) ** 2
            denominator += Float64(weights[i]) * (Float64(y[i]) - mean) ** 2
        if numerator == 0:
            return 1.0
        if denominator == 0:
            return 0.0
        return 1.0 - numerator / denominator

    def score(
        self,
        X: List[List[Scalar[Self.dtype]]],
        y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        return self.score(_column(X), y, sample_weight)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        check_is_fitted(self)
        return [String("isotonicregression0")]
