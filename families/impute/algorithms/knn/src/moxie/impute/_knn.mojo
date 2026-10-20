"""Dense nearest-neighbor imputation using the shared neighbors query engine."""
import numojo as nm
from std.math import isnan, isfinite
from moxie._feature_metadata import FeatureNamesOut
from moxie.impute._base import Imputer
from moxie.impute._indicator import MissingIndicator
from moxie.impute._plumbing import MissingValue, FeatureLayout, missing_mask, select_features, append_indicator
from moxie.neighbors._contracts import DenseMetric
from moxie.neighbors._distance import Metric
from moxie.neighbors._query_engine import BruteForce, NeighborWeights, BuiltinWeights


def _weighted_average[dtype: DType](values: nm.NDArray[dtype], weights: nm.NDArray[dtype]) raises -> SIMD[dtype, 1]:
    var denominator = nm.sum(weights)
    var numerator = nm.sum(values * weights)
    # np.ma.average retains the numerator when zero division is masked.
    return numerator if denominator == 0 else numerator / denominator


@fieldwise_init
struct KNNParams[dtype: DType, M: DenseMetric & Deinitable, W: NeighborWeights & Deinitable](Copyable, Movable):
    var missing_values: MissingValue[Self.dtype]
    var n_neighbors: Int
    var weights: String
    var metric: String
    var copy_input: Bool
    var add_indicator: Bool
    var keep_empty_features: Bool
    var metric_callback: Optional[Self.M]
    var weights_callback: Optional[Self.W]


struct KNNImputer[dtype: DType = DType.float64, M: DenseMetric & Deinitable = Metric[dtype], W: NeighborWeights & Deinitable = BuiltinWeights](Imputer, FeatureNamesOut, Copyable, Movable):
    comptime Params = KNNParams[Self.dtype, Self.M, Self.W]
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = nm.NDArray[Self.dtype]
    comptime Transformed = nm.NDArray[Self.dtype]
    var _params: Self.Params
    var _layout: FeatureLayout
    var _fit_X: Optional[Self.Input]
    var _mask_fit_X: Optional[nm.NDArray[DType.bool]]
    var _metric: Optional[Self.M]
    var indicator_: Optional[MissingIndicator[Self.dtype]]
    var n_features_in_: Int
    var feature_names_in_: Optional[List[String]]

    def __init__(out self, *, missing_values: MissingValue[Self.dtype] = MissingValue[Self.dtype](), n_neighbors: Int = 5, weights: String = "uniform", metric: String = "nan_euclidean", copy: Bool = True, add_indicator: Bool = False, keep_empty_features: Bool = False, metric_callback: Optional[Self.M] = None, weights_callback: Optional[Self.W] = None):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        comptime assert Self.M.dtype == Self.dtype
        self._params = Self.Params(missing_values.copy(), n_neighbors, weights, metric, copy, add_indicator, keep_empty_features, metric_callback.copy(), weights_callback.copy())
        self._layout = FeatureLayout()
        self._fit_X = None
        self._mask_fit_X = None
        self._metric = None
        self.indicator_ = None
        self.n_features_in_ = 0
        self.feature_names_in_ = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._params.copy()

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self = Self(missing_values=params.missing_values, n_neighbors=params.n_neighbors, weights=params.weights, metric=params.metric, copy=params.copy_input, add_indicator=params.add_indicator, keep_empty_features=params.keep_empty_features, metric_callback=params.metric_callback, weights_callback=params.weights_callback)
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, missing_values: Optional[MissingValue[Self.dtype]] = None, n_neighbors: Optional[Int] = None, weights: Optional[String] = None, metric: Optional[String] = None, copy: Optional[Bool] = None, add_indicator: Optional[Bool] = None, keep_empty_features: Optional[Bool] = None, metric_callback: Optional[Self.M] = None, weights_callback: Optional[Self.W] = None) raises -> ref[self] Self:
        var params = self.get_params()
        if missing_values: params.missing_values = missing_values.value().copy()
        if n_neighbors: params.n_neighbors = n_neighbors.value()
        if weights: params.weights = weights.value()
        if metric: params.metric = metric.value()
        if copy: params.copy_input = copy.value()
        if add_indicator: params.add_indicator = add_indicator.value()
        if keep_empty_features: params.keep_empty_features = keep_empty_features.value()
        if metric_callback: params.metric_callback = metric_callback.copy()
        if weights_callback: params.weights_callback = weights_callback.copy()
        return self.set_params(params)

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        _ = result.set_params(self._params)
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._layout.state.is_fitted()

    def _configured_metric(self) raises -> Self.M:
        if self._params.n_neighbors <= 0:
            raise Error("ValueError: n_neighbors must be positive")
        if self._params.weights != "uniform" and self._params.weights != "distance" and self._params.weights != "callable":
            raise Error("ValueError: weights must be uniform, distance or callable")
        if self._params.weights == "callable" and not self._params.weights_callback:
            raise Error("ValueError: callable weights require weights_callback")
        if self._params.metric == "callable":
            if not self._params.metric_callback:
                raise Error("ValueError: callable metric requires metric_callback")
            return self._params.metric_callback.value().copy()
        if self._params.metric != "nan_euclidean":
            raise Error("ValueError: metric must be nan_euclidean or callable")
        comptime if Self.M == Metric[Self.dtype]:
            return rebind[Self.M](Metric[Self.dtype]("nan_euclidean")).copy()
        else:
            raise Error("ValueError: custom metric type requires metric='callable'")

    def _normalized(self, X: Self.Input, mask: nm.NDArray[DType.bool]) raises -> Self.Input:
        # Compact domain copy; finite sentinels become NaN for shared metrics.
        var result = nm.zeros[Self.dtype](nm.Shape(X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                var value = X.unsafe_get(i * Int(X.strides[0]) + j * Int(X.strides[1]))
                if mask.unsafe_get(i * X.shape[1] + j):
                    value = Float64("nan").cast[Self.dtype]()
                result.unsafe_set(i * X.shape[1] + j, value)
        return result^

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        self._fit(X, y)
        return self

    def _fit(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises:
        var metric = self._configured_metric()
        var mask = missing_mask(X, self._params.missing_values)
        metric.validate_features(X.shape[1])
        var layout = FeatureLayout()
        layout.fit(mask, keep_empty_features=self._params.keep_empty_features, error_on_new=False)
        var snapshot = self._normalized(X, mask)
        var indicator: Optional[MissingIndicator[Self.dtype]] = None
        if self._params.add_indicator:
            var fitted = MissingIndicator[Self.dtype](missing_values=self._params.missing_values, error_on_new=False)
            fitted.fit_mask(mask)
            indicator = fitted^
        self._fit_X = snapshot^
        self._mask_fit_X = mask^
        self._metric = metric^
        self._layout = layout^
        self.indicator_ = indicator^
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = None

    def fit_named(mut self, X: Self.Input, feature_names: List[String], y: Optional[Self.Target] = None) raises:
        if X.ndim != 2 or len(feature_names) != X.shape[1]:
            raise Error("ValueError: feature_names length mismatch")
        _ = self.fit(X, y)
        self.feature_names_in_ = feature_names.copy()

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        self._layout.state.check_features(X.shape[1] if X.ndim == 2 else 0)
        var mask = missing_mask(X, self._params.missing_values)
        var query = self._normalized(X, mask)
        var output = select_features(X, self._layout.retained_features)
        ref training = self._fit_X.value()
        ref training_mask = self._mask_fit_X.value()
        var p = self.n_features_in_
        for out_col in range(len(self._layout.retained_features)):
            var col = self._layout.retained_features[out_col]
            var donors = List[Int]()
            for row in range(training.shape[0]):
                if not training_mask.unsafe_get(row * p + col): donors.append(row)
            if len(donors) == 0:
                for row in range(X.shape[0]): output.unsafe_set(row * output.shape[1] + out_col, 0)
                continue
            var receivers = List[Int]()
            for row in range(X.shape[0]):
                if mask.unsafe_get(row * p + col): receivers.append(row)
            if len(receivers) == 0: continue
            var donor_X = nm.zeros[Self.dtype](nm.Shape(len(donors), p))
            var values = nm.zeros[Self.dtype](nm.Shape(len(donors)))
            for i in range(len(donors)):
                values.unsafe_set(i, training.unsafe_get(donors[i] * p + col))
                for j in range(p): donor_X.unsafe_set(i * p + j, training.unsafe_get(donors[i] * p + j))
            var fallback = nm.sum(values) / SIMD[Self.dtype, 1](len(donors))
            var engine = BruteForce(self._metric.value(), allow_nan=True)
            engine.fit(rebind[nm.NDArray[Self.M.dtype]](donor_X))
            var k = min(self._params.n_neighbors, len(donors))
            # Bound neighbor output and receiver copies independently of query count.
            for start in range(0, len(receivers), 128):
                var count = min(128, len(receivers) - start)
                var batch = nm.zeros[Self.dtype](nm.Shape(count, p))
                for i in range(count):
                    for j in range(p): batch.unsafe_set(i * p + j, query.unsafe_get(receivers[start + i] * p + j))
                var nearest = engine.kneighbors(rebind[nm.NDArray[Self.M.dtype]](batch).view(), k)
                for i in range(count):
                    var distances = nm.zeros[DType.float64](nm.Shape(k))
                    var donor_values = nm.zeros[Self.dtype](nm.Shape(k))
                    var all_nan = True
                    for j in range(k):
                        var distance = Float64(nearest.distances.value().unsafe_get(i * k + j))
                        distances.unsafe_set(j, distance)
                        all_nan = all_nan and isnan(distance)
                        donor_values.unsafe_set(j, values.unsafe_get(nearest.indices[i * k + j]))
                    var result = fallback
                    if not all_nan:
                        var weights: nm.NDArray[DType.float64]
                        if self._params.weights == "callable":
                            weights = self._params.weights_callback.value().apply(distances, [0, k])
                        else:
                            # Shared distance weights reject NaNs. Infinity has
                            # exactly the required zero-weight behavior here.
                            var safe = distances.copy()
                            for j in range(k):
                                var distance = safe.unsafe_get(j)
                                if isnan(distance):
                                    safe.unsafe_set(j, Float64("inf"))
                                elif self._params.weights == "distance" and distance > 0:
                                    # sklearn detects reciprocal overflow in the
                                    # distance dtype, then weights those donors as
                                    # exact matches. The shared helper uses Float64.
                                    if not isfinite((Float64(1) / distance).cast[Self.dtype]()):
                                        safe.unsafe_set(j, Float64(0))
                            weights = BuiltinWeights(self._params.weights).apply(safe, [0, k])
                        if weights.ndim != 1 or weights.shape[0] != k:
                            raise Error("ValueError: weights callback must return one weight per neighbor")
                        var native_weights = nm.zeros[Self.dtype](nm.Shape(k))
                        for j in range(k):
                            var weight = weights.unsafe_get(j * Int(weights.strides[0]))
                            if isnan(weight) or (self._params.weights != "callable" and isnan(distances.unsafe_get(j))): weight = 0
                            if not isfinite(weight):
                                raise Error("ValueError: infinite callable weights are unsupported")
                            native_weights.unsafe_set(j, weight.cast[Self.dtype]())
                        if self._params.weights == "callable":
                            # The native callback returns Float64, so NumPy's
                            # weighted-average promotion also computes in Float64.
                            var wide_values = nm.zeros[DType.float64](nm.Shape(k))
                            var wide_weights = nm.zeros[DType.float64](nm.Shape(k))
                            for j in range(k):
                                wide_values.unsafe_set(j, Float64(donor_values.unsafe_get(j)))
                                var weight = weights.unsafe_get(j * Int(weights.strides[0]))
                                wide_weights.unsafe_set(j, Float64(0) if isnan(weight) else weight)
                            result = _weighted_average(wide_values, wide_weights).cast[Self.dtype]()
                        else:
                            result = _weighted_average(donor_values, native_weights)
                    output.unsafe_set(receivers[start + i] * output.shape[1] + out_col, result)
        if self._params.add_indicator: return append_indicator(output, mask, self._layout)
        return output^

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        self._layout.state.check_features(self.n_features_in_)
        var names = List[String]()
        if input_features:
            if len(input_features.value()) != self.n_features_in_:
                raise Error("ValueError: input_features length mismatch")
            names = input_features.value().copy()
            if self.feature_names_in_:
                if names != self.feature_names_in_.value(): raise Error("ValueError: input_features differ from fitted names")
        elif self.feature_names_in_:
            names = self.feature_names_in_.value().copy()
        else:
            for j in range(self.n_features_in_): names.append("x" + String(j))
        var result = List[String]()
        # The pinned reference selects _valid_mask even with keep_empty_features.
        for j in range(self.n_features_in_):
            var empty = False
            for col in self._layout.empty_features:
                if col == j: empty = True
            if not empty: result.append(names[j])
        if self._params.add_indicator:
            for j in self._layout.indicator_features: result.append("missingindicator_" + names[j])
        return result^
