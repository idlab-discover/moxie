"""Dense RadiusNeighborsClassifier; see docs/estimators.md for native type adaptations."""
import numojo as nm
from moxie.base import BaseEstimator
from moxie.neighbors._base import RadiusNeighborsClassifierAPI
from moxie.neighbors._classification_targets import ClassificationTargets, LabelOps
from moxie.neighbors._distance import Metric
from moxie.neighbors._contracts import DenseMetric, TopKResult, RadiusResult
from moxie.neighbors._unsupervised import NearestNeighbors, NeighborParams
from moxie.neighbors._query_engine import NeighborWeights, BuiltinWeights
from moxie.neighbors._supervised import target_snapshot, target_at, class_arrays, weighted_neighbors, regression, probabilities, classified, prediction_score

@fieldwise_init
struct RadiusNeighborsClassifierParams[dtype: DType, M: DenseMetric & Deinitable = Metric[dtype], Y: Copyable & Deinitable = nm.NDArray[dtype], W: NeighborWeights & Deinitable = BuiltinWeights](Copyable, Movable):
    var neighbors: NeighborParams[Self.dtype, Self.M]
    var weighting: Optional[Self.W]
    var weights: String
    var outlier_label: Optional[Self.Y]
    var outlier_most_frequent: Bool

struct RadiusNeighborsClassifier[dtype: DType = DType.float64, M: DenseMetric & Deinitable = Metric[dtype], Y: Copyable & Deinitable = nm.NDArray[dtype], W: NeighborWeights & Deinitable = BuiltinWeights](RadiusNeighborsClassifierAPI):
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.Y
    comptime Prediction = Self.Target
    comptime SampleWeight = nm.NDArray[Self.dtype]
    comptime NeighborResult = RadiusResult[Self.dtype]
    comptime Probabilities = List[nm.NDArray[DType.float64]]
    comptime Params = RadiusNeighborsClassifierParams[Self.dtype, Self.M, Self.Y, Self.W]
    var _query: NearestNeighbors[Self.dtype, Self.M]
    var _y: Optional[ClassificationTargets[Self.Y]]
    var weighting: Optional[Self.W]
    var weights: String
    var n_features_in_: Int
    var n_samples_fit_: Int
    var outputs_2d_: Bool
    var classes_: List[Self.Y]
    var outlier_label_: Optional[Self.Y]
    var _outlier_label: Optional[Self.Y]
    var _outlier_most_frequent: Bool
    var effective_metric_: String
    var effective_metric_params_: Self.M
    var _fit_method: String

    def __init__(out self, radius: Float64 = 1.0, *, weights: String = "uniform", weighting: Optional[Self.W] = None,
                 algorithm: String = "auto", leaf_size: Int = 30, p: Float64 = 2,
                 metric: String = "minkowski", metric_params: Optional[Self.M] = None,
                 n_jobs: Optional[Int] = None, outlier_label: Optional[Self.Y] = None, outlier_most_frequent: Bool = False) raises:
        self._query = NearestNeighbors[Self.dtype, Self.M](radius=radius, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params.copy(), n_jobs=n_jobs)
        self.weighting = weighting.copy()
        self.weights = weights
        self._y = None
        self.n_features_in_ = 0
        self.n_samples_fit_ = 0
        self.outputs_2d_ = False
        self.classes_ = List[Self.Y]()
        self.outlier_label_ = None
        self._outlier_label = outlier_label.copy()
        self._outlier_most_frequent = outlier_most_frequent
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

    def set_params[origin: MutOrigin](ref[origin] self, *, radius: Optional[Float64] = None,
                                      weights: Optional[String] = None, weighting: Optional[Self.W] = None,
                                      algorithm: Optional[String] = None, leaf_size: Optional[Int] = None,
                                      p: Optional[Float64] = None, metric: Optional[String] = None,
                                      metric_params: Optional[Self.M] = None, n_jobs: Optional[Int] = None,
                                      outlier_label: Optional[Self.Y] = None, outlier_most_frequent: Optional[Bool] = None) raises -> ref[self] Self:
        """Update only the supplied parameters (use the Params record to reset Optionals to None)."""
        var neighbors = self._query.get_params()
        neighbors.update(radius=radius, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric,
                         metric_params=metric_params.copy(), n_jobs=n_jobs)
        _ = self._query.set_params(neighbors)
        if weights: self.weights = weights.value()
        if weighting: self.weighting = weighting.copy()
        if outlier_label:
            self._outlier_label = outlier_label.copy()
            self._outlier_most_frequent = False
        if outlier_most_frequent: self._outlier_most_frequent = outlier_most_frequent.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self(metric_params=self._query.get_params().metric_params.copy())
        result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._y)

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: nm.NDArray[Self.dtype], ref[y_origin] y: Self.Y) raises -> ref[self] Self:
        if not self.weighting and self.weights != "uniform" and self.weights != "distance": raise Error("ValueError: invalid weights")
        if X.ndim != 2: raise Error("ValueError: X must be a matrix")
        var target = ClassificationTargets(y,X.shape[0])
        var query = self._query.__sklearn_clone__()
        query.fit(X)
        var classes = target.classes.copy()
        var outliers = Optional[Self.Y](None)
        if self._outlier_most_frequent:
            outliers = target.most_frequent()
        elif self._outlier_label:
            _ = target.outlier_codes(self._outlier_label)
            outliers = self._outlier_label.copy()
        self.outputs_2d_ = target.outputs_2d
        self._query = query^
        self._y = target^
        self.classes_ = classes^
        self.outlier_label_ = outliers^
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

    def predict(self, X: Optional[nm.NDArray[Self.dtype]] = None) raises -> Self.Y:
        if self.weighting: return self.predict_with_weights(X,self.weighting.value())
        return self.predict_with_weights(X, BuiltinWeights(self.weights))

    def predict_with_weights[WeightFn: NeighborWeights](self, X: Optional[nm.NDArray[Self.dtype]], weighting: WeightFn) raises -> Self.Y:
        if not self._y: raise Error("NotFittedError: estimator has not been fitted")
        var probs = self.predict_proba_with_weights(X, weighting)
        return self._y.value().predict(probs,self.outlier_label_)

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        return self.predict_proba(Optional[Self.Input](X.copy()))

    def predict_proba(self, X: Optional[nm.NDArray[Self.dtype]] = None) raises -> List[nm.NDArray[DType.float64]]:
        if self.weighting: return self.predict_proba_with_weights(X,self.weighting.value())
        return self.predict_proba_with_weights(X, BuiltinWeights(self.weights))

    def predict_proba_with_weights[WeightFn: NeighborWeights](self, X: Optional[nm.NDArray[Self.dtype]], weighting: WeightFn) raises -> List[nm.NDArray[DType.float64]]:
        if not self._y: raise Error("NotFittedError: estimator has not been fitted")
        var rows = weighted_neighbors(self._query, X, True, weighting)
        return self._y.value().proba(rows,self.outlier_label_)

    def score(self, X: nm.NDArray[Self.dtype], y: Self.Y, sample_weight: Optional[nm.NDArray[Self.dtype]] = None) raises -> Float64:
        var pred = self.predict(X.copy())
        return ClassificationTargets[Self.Y].score(pred,y,sample_weight)

    def score(self, X: Optional[nm.NDArray[Self.dtype]], y: Self.Y, sample_weight: Optional[nm.NDArray[Self.dtype]] = None) raises -> Float64:
        var pred = self.predict(X)
        return ClassificationTargets[Self.Y].score(pred,y,sample_weight)
