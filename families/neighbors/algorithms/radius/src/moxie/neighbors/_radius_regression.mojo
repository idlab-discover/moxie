"""Dense RadiusNeighborsRegressor; see docs/estimators.md for native type adaptations."""
import numojo as nm
from moxie.base import BaseEstimator
from moxie.neighbors._base import RadiusNeighborsRegressorAPI
from moxie.neighbors._distance import Metric
from moxie.neighbors._contracts import DenseMetric, TopKResult, RadiusResult
from moxie.neighbors._unsupervised import NearestNeighbors, NeighborParams
from moxie.neighbors._query_engine import NeighborWeights, BuiltinWeights
from moxie.neighbors._supervised import target_snapshot, target_at, class_arrays, weighted_neighbors, regression, probabilities, classified, prediction_score

@fieldwise_init
struct RadiusNeighborsRegressorParams[dtype: DType, M: DenseMetric & Deinitable = Metric[dtype], W: NeighborWeights & Deinitable = BuiltinWeights](Copyable, Movable):
    var neighbors: NeighborParams[Self.dtype, Self.M]
    var weighting: Optional[Self.W]
    var weights: String
    var outlier_label: Optional[nm.NDArray[Self.dtype]]
    var outlier_most_frequent: Bool

struct RadiusNeighborsRegressor[dtype: DType = DType.float64, M: DenseMetric & Deinitable = Metric[dtype], W: NeighborWeights & Deinitable = BuiltinWeights](RadiusNeighborsRegressorAPI):
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = nm.NDArray[Self.dtype]
    comptime Prediction = Self.Target
    comptime SampleWeight = nm.NDArray[Self.dtype]
    comptime NeighborResult = RadiusResult[Self.dtype]
    comptime Params = RadiusNeighborsRegressorParams[Self.dtype, Self.M, Self.W]
    var _query: NearestNeighbors[Self.dtype, Self.M]
    var _y: Optional[nm.NDArray[Self.dtype]]
    var weighting: Optional[Self.W]
    var weights: String
    var n_features_in_: Int
    var n_samples_fit_: Int
    var outputs_2d_: Bool
    var classes_: List[nm.NDArray[Self.dtype]]
    var outlier_label_: Optional[nm.NDArray[Self.dtype]]
    var _outlier_label: Optional[nm.NDArray[Self.dtype]]
    var _outlier_most_frequent: Bool
    var effective_metric_: String
    var effective_metric_params_: Self.M
    var _fit_method: String

    def __init__(out self, radius: Float64 = 1.0, *, weights: String = "uniform", weighting: Optional[Self.W] = None,
                 algorithm: String = "auto", leaf_size: Int = 30, p: Float64 = 2,
                 metric: String = "minkowski", metric_params: Optional[Self.M] = None,
                 n_jobs: Optional[Int] = None) raises:
        self._query = NearestNeighbors[Self.dtype, Self.M](radius=radius, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params.copy(), n_jobs=n_jobs)
        self.weighting = weighting.copy()
        self.weights = weights
        self._y = None
        self.n_features_in_ = 0
        self.n_samples_fit_ = 0
        self.outputs_2d_ = False
        self.classes_ = List[nm.NDArray[Self.dtype]]()
        self.outlier_label_ = None
        self._outlier_label = None
        self._outlier_most_frequent = False
        self.effective_metric_ = self._query.effective_metric_
        self.effective_metric_params_ = self._query.effective_metric_params_.copy()
        self._fit_method = "brute"

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self._query.get_params(), self.weighting.copy(), self.weights, self._outlier_label.copy(), self._outlier_most_frequent)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self._query.set_params(params.neighbors)
        self.weighting = params.weighting.copy()
        self.weights = params.weights
        self._outlier_label = params.outlier_label.copy()
        self._outlier_most_frequent = params.outlier_most_frequent
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self(metric_params=self._query.get_params().metric_params.copy())
        result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._y)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: nm.NDArray[Self.dtype], ref[y_origin] y: nm.NDArray[Self.dtype]) raises -> ref[self] Self:
        if not self.weighting and self.weights != "uniform" and self.weights != "distance": raise Error("ValueError: invalid weights")
        if X.ndim != 2: raise Error("ValueError: X must be a matrix")
        var target = target_snapshot(y, X.shape[0], False)
        var query = self._query.__sklearn_clone__()
        query.fit(X)
        var classes = List[nm.NDArray[Self.dtype]]()
        var outliers = Optional[nm.NDArray[Self.dtype]](None)
        self._query = query^
        self._y = target^
        self.classes_ = classes^
        self.outlier_label_ = outliers^
        self.outputs_2d_ = y.ndim == 2
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]
        self._fit_method = self._query._fit_method
        self.effective_metric_ = self._query.effective_metric_
        self.effective_metric_params_ = self._query.effective_metric_params_.copy()
        return self

    def kneighbors(self, X: Optional[nm.NDArray[Self.dtype]] = None,
                   n_neighbors: Optional[Int] = None, return_distance: Bool = True) raises -> TopKResult[Self.dtype]:
        return self._query.kneighbors(X, n_neighbors, return_distance)

    def radius_neighbors(self, X: Optional[nm.NDArray[Self.dtype]] = None,
                         radius: Optional[Float64] = None, return_distance: Bool = True,
                         sort_results: Bool = False) raises -> RadiusResult[Self.dtype]:
        return self._query.radius_neighbors(X, radius, return_distance, sort_results)

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        return self.predict(Optional[Self.Input](X.copy()))

    def predict(self, X: Optional[nm.NDArray[Self.dtype]] = None) raises -> nm.NDArray[Self.dtype]:
        if self.weighting: return self.predict_with_weights(X,self.weighting.value())
        return self.predict_with_weights(X, BuiltinWeights(self.weights))

    def predict_with_weights[WeightFn: NeighborWeights](self, X: Optional[nm.NDArray[Self.dtype]], weighting: WeightFn) raises -> nm.NDArray[Self.dtype]:
        if not self._y: raise Error("NotFittedError: estimator has not been fitted")
        var rows = weighted_neighbors(self._query, X, True, weighting)
        return regression(self._y.value(), rows)

    def score(self, X: nm.NDArray[Self.dtype], y: nm.NDArray[Self.dtype], sample_weight: Optional[nm.NDArray[Self.dtype]] = None) raises -> Float64:
        var pred = self.predict(X.copy())
        return prediction_score(pred, y, False, sample_weight)

    def score(self, X: Optional[nm.NDArray[Self.dtype]], y: nm.NDArray[Self.dtype], sample_weight: Optional[nm.NDArray[Self.dtype]] = None) raises -> Float64:
        var pred = self.predict(X)
        return prediction_score(pred, y, False, sample_weight)
