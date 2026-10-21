"""Dense Local Outlier Factor using exact streaming neighbor queries."""
import numojo as nm
from std.math import max, isfinite, floor
from moxie.base import BaseEstimator
from moxie.neighbors._base import NoveltyDetector
from moxie.neighbors._unsupervised import NearestNeighbors, NeighborParams
from moxie.neighbors._distance import Metric
from moxie.neighbors._contracts import DenseMetric, TopKResult

@fieldwise_init
struct LOFParams[dtype: DType, M: DenseMetric & Deinitable = Metric[dtype]](Copyable, Movable):
    var neighbors: NeighborParams[Self.dtype, Self.M]
    var contamination: Optional[Float64]
    var novelty: Bool

struct LocalOutlierFactor[dtype: DType = DType.float64, M: DenseMetric & Deinitable = Metric[dtype]](NoveltyDetector):
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime Prediction = Self.Input
    comptime Decision = Self.Input
    comptime SampleScores = Self.Input
    comptime Params = LOFParams[Self.dtype, Self.M]
    var _query: NearestNeighbors[Self.dtype, Self.M]
    var contamination: Optional[Float64]
    var novelty: Bool
    var n_neighbors_: Int
    var n_features_in_: Int
    var n_samples_fit_: Int
    var negative_outlier_factor_: nm.NDArray[Self.dtype]
    var offset_: Float64
    var _lrd: nm.NDArray[Self.dtype]
    var _k_distance: nm.NDArray[Self.dtype]
    var effective_metric_: String
    var effective_metric_params_: Self.M

    def __init__(out self, n_neighbors: Int = 20, *, algorithm: String = "auto",
                 leaf_size: Int = 30, metric: String = "minkowski", p: Float64 = 2,
                 metric_params: Optional[Self.M] = None,
                 contamination: Optional[Float64] = None, novelty: Bool = False,
                 n_jobs: Optional[Int] = None) raises:
        self._query = NearestNeighbors[Self.dtype, Self.M](n_neighbors,algorithm=algorithm,leaf_size=leaf_size,metric=metric,p=p,metric_params=metric_params.copy(),n_jobs=n_jobs)
        self.contamination = contamination
        self.novelty = novelty
        self.n_neighbors_ = 0
        self.n_features_in_ = 0
        self.n_samples_fit_ = 0
        self.negative_outlier_factor_ = nm.zeros[Self.dtype](nm.Shape(0))
        self._lrd = nm.zeros[Self.dtype](nm.Shape(0))
        self._k_distance = nm.zeros[Self.dtype](nm.Shape(0))
        self.offset_ = -1.5
        self.effective_metric_ = self._query.effective_metric_
        self.effective_metric_params_ = self._query.effective_metric_params_.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self._query.get_params(), self.contamination, self.novelty)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self._query.set_params(params.neighbors)
        self.contamination = params.contamination
        self.novelty = params.novelty
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, n_neighbors: Optional[Int] = None,
                                      algorithm: Optional[String] = None, leaf_size: Optional[Int] = None,
                                      metric: Optional[String] = None, p: Optional[Float64] = None,
                                      metric_params: Optional[Self.M] = None, contamination: Optional[Float64] = None,
                                      novelty: Optional[Bool] = None, n_jobs: Optional[Int] = None) raises -> ref[self] Self:
        """Update only the supplied parameters; reset contamination to "auto" (None) through the Params record."""
        var neighbors = self._query.get_params()
        neighbors.update(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric,
                         metric_params=metric_params.copy(), n_jobs=n_jobs)
        _ = self._query.set_params(neighbors)
        if contamination: self.contamination = contamination
        if novelty: self.novelty = novelty.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self(metric_params=self._query.get_params().metric_params.copy())
        result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_samples_fit_ > 0

    @staticmethod
    def _density(result: TopKResult[Self.dtype], k_distance: nm.NDArray[Self.dtype]) raises -> nm.NDArray[Self.dtype]:
        var output = nm.zeros[Self.dtype](nm.Shape(result.n_queries))
        for i in range(result.n_queries):
            var total = Scalar[Self.dtype](0)
            for j in range(result.k):
                var pos = i*result.k+j
                total += max(result.distances.value().unsafe_get(pos),k_distance.unsafe_get(result.indices[pos]))
            output.unsafe_set(i,1/(total/Scalar[Self.dtype](result.k)+Scalar[Self.dtype](1e-10)))
        return output^

    @staticmethod
    def _scores(result: TopKResult[Self.dtype], lrd: nm.NDArray[Self.dtype], train_lrd: nm.NDArray[Self.dtype]) raises -> nm.NDArray[Self.dtype]:
        var output = nm.zeros[Self.dtype](nm.Shape(result.n_queries))
        for i in range(result.n_queries):
            var total = Scalar[Self.dtype](0)
            for j in range(result.k): total += train_lrd.unsafe_get(result.indices[i*result.k+j])/lrd.unsafe_get(i)
            output.unsafe_set(i,-total/Scalar[Self.dtype](result.k))
        return output^

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: nm.NDArray[Self.dtype], y: Optional[nm.NDArray[Self.dtype]] = None) raises -> ref[self] Self:
        if self.contamination:
            var c = self.contamination.value()
            if not isfinite(c) or c <= 0 or c > 0.5: raise Error("ValueError: contamination must be in (0, 0.5]")
        var query = self._query.__sklearn_clone__()
        query.fit(X)
        var k = min(query.get_params().n_neighbors,X.shape[0]-1)
        var neighbors = query.kneighbors(None,k)
        var kth = nm.zeros[Self.dtype](nm.Shape(X.shape[0]))
        for i in range(X.shape[0]): kth.unsafe_set(i,neighbors.distances.value().unsafe_get(i*k+k-1))
        var lrd = Self._density(neighbors,kth)
        var scores = Self._scores(neighbors,lrd,lrd)
        var offset = Float64(-1.5)
        if self.contamination:
            # Linear percentile over index metadata, preserving the scores array.
            var order = List[Int]()
            for i in range(X.shape[0]):
                order.append(i)
                var pos = i
                while pos > 0 and scores.unsafe_get(order[pos]) < scores.unsafe_get(order[pos-1]):
                    var tmp = order[pos]
                    order[pos] = order[pos-1]
                    order[pos-1] = tmp
                    pos -= 1
            var rank = self.contamination.value()*Float64(X.shape[0]-1)
            var lo = Int(floor(rank))
            var hi = min(lo+1,X.shape[0]-1)
            var a = Float64(scores.unsafe_get(order[lo]))
            var b = Float64(scores.unsafe_get(order[hi]))
            offset = a+(b-a)*(rank-Float64(lo))
        self._query = query^
        self.n_neighbors_ = k
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]
        self._k_distance = kth^
        self._lrd = lrd^
        self.negative_outlier_factor_ = scores^
        self.offset_ = offset
        self.effective_metric_ = self._query.effective_metric_
        self.effective_metric_params_ = self._query.effective_metric_params_.copy()
        return self

    def kneighbors(self, X: Optional[nm.NDArray[Self.dtype]] = None,
                   n_neighbors: Optional[Int] = None, return_distance: Bool = True) raises -> TopKResult[Self.dtype]:
        """KNeighborsMixin.kneighbors on the fitted data; the default k is the n_neighbors parameter."""
        if not self.__sklearn_is_fitted__(): raise Error("NotFittedError: LOF has not been fitted")
        return self._query.kneighbors(X, n_neighbors, return_distance)

    def score_samples(self, X: nm.NDArray[Self.dtype]) raises -> nm.NDArray[Self.dtype]:
        if not self.novelty: raise Error("AttributeError: score_samples requires novelty=True")
        if not self.__sklearn_is_fitted__(): raise Error("NotFittedError: LOF has not been fitted")
        var result = self._query.kneighbors(X.copy(),self.n_neighbors_)
        var density = Self._density(result,self._k_distance)
        return Self._scores(result,density,self._lrd)

    def decision_function(self, X: nm.NDArray[Self.dtype]) raises -> nm.NDArray[Self.dtype]:
        if not self.novelty: raise Error("AttributeError: decision_function requires novelty=True")
        var scores = self.score_samples(X)
        for i in range(scores.shape[0]): scores.unsafe_set(i,(Float64(scores.unsafe_get(i))-self.offset_).cast[Self.dtype]())
        return scores^

    def predict(self, X: nm.NDArray[Self.dtype]) raises -> nm.NDArray[Self.dtype]:
        if not self.novelty: raise Error("AttributeError: predict requires novelty=True")
        var scores = self.decision_function(X)
        for i in range(scores.shape[0]): scores.unsafe_set(i,Scalar[Self.dtype](-1) if scores.unsafe_get(i)<0 else Scalar[Self.dtype](1))
        return scores^

    def fit_predict(mut self, X: nm.NDArray[Self.dtype], y: Optional[nm.NDArray[Self.dtype]] = None) raises -> nm.NDArray[Self.dtype]:
        if self.novelty: raise Error("AttributeError: fit_predict requires novelty=False")
        self.fit(X,y)
        var labels = nm.ones[Self.dtype](nm.Shape(self.n_samples_fit_))
        for i in range(self.n_samples_fit_):
            if Float64(self.negative_outlier_factor_.unsafe_get(i)) < self.offset_: labels.unsafe_set(i,-1)
        return labels^
