"""Dense public queries backed by brute force or the family spatial trees."""
import numojo as nm
from std.math import isnan
from moxie.base import BaseEstimator, UnsupervisedEstimator
from moxie.neighbors._base import KNeighborsQuery
from moxie.neighbors._distance import Metric
from moxie.neighbors._query_engine import BruteForce
from moxie.neighbors._kd_tree import KDTree, kd_eligible
from moxie.neighbors._ball_tree import BallTree, ball_eligible
from moxie.neighbors._binary_tree import TreeStats
from moxie.neighbors._contracts import DenseMetric, validate_k, validate_matrix
from moxie.neighbors._contracts import TopKResult, RadiusResult, validate_radius_options

@fieldwise_init
struct NeighborParams[dtype: DType, M: DenseMetric & Deinitable = Metric[dtype]](Copyable, Movable):
    var n_neighbors: Int
    var radius: Float64
    var algorithm: String
    var leaf_size: Int
    var p: Float64
    var metric: String
    var metric_params: Optional[Self.M]
    var n_jobs: Optional[Int]

    def update(mut self, n_neighbors: Optional[Int] = None, radius: Optional[Float64] = None,
               algorithm: Optional[String] = None, leaf_size: Optional[Int] = None,
               p: Optional[Float64] = None, metric: Optional[String] = None,
               metric_params: Optional[Self.M] = None, n_jobs: Optional[Int] = None):
        """Replace only the supplied fields; keyword set_params overloads share this."""
        if n_neighbors: self.n_neighbors = n_neighbors.value()
        if radius: self.radius = radius.value()
        if algorithm: self.algorithm = algorithm.value()
        if leaf_size: self.leaf_size = leaf_size.value()
        if p: self.p = p.value()
        if metric: self.metric = metric.value()
        if metric_params: self.metric_params = metric_params.copy()
        if n_jobs: self.n_jobs = n_jobs

    def resolved_metric(self) raises -> Self.M:
        if self.metric_params:
            return self.metric_params.value().copy()
        comptime if Self.M == Metric[Self.dtype]:
            comptime assert Self.M == Metric[Self.dtype]
            var metric = Metric[Self.dtype]("minkowski" if self.metric == "p" else self.metric, self.p)
            return rebind[Self.M](metric).copy()
        else:
            raise Error("ValueError: native callable metric requires metric_params")

    def metric_name(self, metric: Self.M) -> String:
        comptime if Self.M == Metric[Self.dtype]:
            comptime assert Self.M == Metric[Self.dtype]
            return rebind[Metric[Self.dtype]](metric).name
        else:
            return "callable"

    def make_engine(self, metric: Self.M) raises -> BruteForce[Self.M]:
        comptime if Self.M == Metric[Self.dtype]:
            comptime assert Self.M == Metric[Self.dtype]
            var name = self.metric_name(metric)
            return BruteForce(metric, precomputed=name == "precomputed", allow_nan=name == "nan_euclidean" and isnan(rebind[Metric[Self.dtype]](metric).missing_values))
        else:
            return BruteForce(metric)

struct NearestNeighbors[dtype: DType = DType.float64, M: DenseMetric & Deinitable = Metric[dtype]](UnsupervisedEstimator, KNeighborsQuery):
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = nm.NDArray[Self.dtype]
    comptime NeighborResult = TopKResult[Self.dtype]
    comptime Params = NeighborParams[Self.dtype, Self.M]
    var _params: Self.Params
    var _engine: BruteForce[Self.M]
    var _tree: Optional[KDTree[Self.dtype]]
    var _ball: Optional[BallTree[Self.dtype, Self.M]]
    var n_features_in_: Int
    var n_samples_fit_: Int
    var effective_metric_: String
    var effective_metric_params_: Self.M
    var _fit_method: String

    def __init__(out self, n_neighbors: Int = 5, radius: Float64 = 1.0, *,
                 algorithm: String = "auto", leaf_size: Int = 30,
                 metric: String = "minkowski", p: Float64 = 2,
                 metric_params: Optional[Self.M] = None,
                 n_jobs: Optional[Int] = None) raises:
        self._params = Self.Params(n_neighbors, radius, algorithm, leaf_size, p, metric, metric_params.copy(), n_jobs)
        comptime assert Self.M.dtype == Self.dtype, "metric dtype must match estimator dtype"
        var m = self._params.resolved_metric()
        self._engine = self._params.make_engine(m)
        self._tree = None
        self._ball = None
        self.n_features_in_ = 0
        self.n_samples_fit_ = 0
        self.effective_metric_ = self._params.metric_name(m)
        self.effective_metric_params_ = m.copy()
        self._fit_method = "brute"

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        # sklearn parameter changes do not discard the fitted snapshot.
        self._params = params.copy()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, n_neighbors: Optional[Int] = None,
                                      radius: Optional[Float64] = None, algorithm: Optional[String] = None,
                                      leaf_size: Optional[Int] = None, metric: Optional[String] = None,
                                      p: Optional[Float64] = None, metric_params: Optional[Self.M] = None,
                                      n_jobs: Optional[Int] = None) raises -> ref[self] Self:
        """Update only the supplied parameters (use the Params record to reset Optionals to None)."""
        self._params.update(n_neighbors, radius, algorithm, leaf_size, p, metric, metric_params.copy(), n_jobs)
        return self

    def __sklearn_clone__(self) raises -> Self:
        ref p = self._params
        return Self(p.n_neighbors, p.radius, algorithm=p.algorithm, leaf_size=p.leaf_size,
                    metric=p.metric, p=p.p, metric_params=p.metric_params.copy(), n_jobs=p.n_jobs)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_samples_fit_ > 0

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        comptime assert Self.M.dtype == Self.dtype
        ref p = self._params
        if p.n_neighbors <= 0: raise Error("ValueError: n_neighbors must be positive")
        validate_radius_options(p.radius, True, False)
        if p.leaf_size <= 0: raise Error("ValueError: leaf_size must be positive")
        if p.algorithm != "auto" and p.algorithm != "brute" and p.algorithm != "kd_tree" and p.algorithm != "ball_tree":
            raise Error("ValueError: invalid algorithm")
        if p.n_jobs and p.n_jobs.value() == 0: raise Error("ValueError: n_jobs cannot be zero")
        var m = p.resolved_metric()
        var engine = p.make_engine(m)
        var tree = Optional[KDTree[Self.dtype]](None)
        var ball = Optional[BallTree[Self.dtype, Self.M]](None)
        var method = String("brute")
        comptime if Self.M == Metric[Self.dtype]:
            comptime assert Self.M == Metric[Self.dtype]
            ref builtin = rebind[Metric[Self.dtype]](m)
            validate_matrix(X,precomputed=builtin.name=="precomputed",training=True,allow_nan=builtin.name=="nan_euclidean" and isnan(builtin.missing_values))
            if p.algorithm == "kd_tree" or (p.algorithm == "auto" and kd_eligible(builtin) and X.shape[1] <= 15 and p.n_neighbors < X.shape[0]//2):
                tree = KDTree[Self.dtype](X,p.leaf_size,metric_params=builtin.copy())
                method = "kd_tree"
            elif p.algorithm == "ball_tree" or (p.algorithm == "auto" and ball_eligible(builtin) and X.shape[1] <= 15 and p.n_neighbors < X.shape[0]//2):
                ball = BallTree[Self.dtype, Self.M](X,p.leaf_size,metric_params=m.copy())
                method = "ball_tree"
            else: engine.fit(rebind[nm.NDArray[Self.M.dtype]](X))
        else:
            validate_matrix(X)
            if p.algorithm == "kd_tree":
                raise Error("ValueError: KDTree does not support callable metrics")
            if p.algorithm == "ball_tree" or (p.algorithm == "auto" and X.shape[1] <= 15 and p.n_neighbors < X.shape[0]//2):
                ball = BallTree[Self.dtype, Self.M](X,p.leaf_size,metric_params=m.copy())
                method = "ball_tree"
            else:
                engine.fit(rebind[nm.NDArray[Self.M.dtype]](X))
        self._tree = tree^
        self._ball = ball^
        self._engine = engine^
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]
        self.effective_metric_ = self._params.metric_name(m)
        self.effective_metric_params_ = m.copy()
        self._fit_method = method
        return self

    def kneighbors(self, X: Optional[nm.NDArray[Self.dtype]] = None,
                   n_neighbors: Optional[Int] = None, return_distance: Bool = True) raises -> TopKResult[Self.dtype]:
        comptime assert Self.M.dtype == Self.dtype
        var k = self._params.n_neighbors
        if n_neighbors: k = n_neighbors.value()
        if self._tree:
            var stats = TreeStats()
            if X: return self._tree.value()._query(X.value(),k,return_distance,False,False,stats)
            return self._tree.value()._query(self._tree.value().data,k,return_distance,False,True,stats)
        if self._ball:
            var stats = TreeStats()
            if X: return self._ball.value()._query(X.value(),k,return_distance,False,False,stats)
            return self._ball.value()._query(self._ball.value().data,k,return_distance,False,True,stats)
        var result = self._engine.kneighbors(rebind[Optional[nm.NDArray[Self.M.dtype]]](X), k, return_distance)
        return rebind[TopKResult[Self.dtype]](result).copy()

    def radius_neighbors(self, X: Optional[nm.NDArray[Self.dtype]] = None,
                         radius: Optional[Float64] = None, return_distance: Bool = True,
                         sort_results: Bool = False) raises -> RadiusResult[Self.dtype]:
        comptime assert Self.M.dtype == Self.dtype
        var r = self._params.radius
        if radius: r = radius.value()
        if self._tree:
            validate_radius_options(r,return_distance,sort_results)
            var radii = nm.full[DType.float64](nm.Shape(1),r)
            var stats = TreeStats()
            if X:
                var result = self._tree.value()._radius(X.value(),radii,return_distance,False,sort_results,False,stats)
                return result.neighbors.take()
            var result = self._tree.value()._radius(self._tree.value().data,radii,return_distance,False,sort_results,True,stats)
            return result.neighbors.take()
        if self._ball:
            validate_radius_options(r,return_distance,sort_results)
            var radii = nm.full[DType.float64](nm.Shape(1),r)
            var stats = TreeStats()
            if X:
                var result = self._ball.value()._radius(X.value(),radii,return_distance,False,sort_results,False,stats)
                return result.neighbors.take()
            var result = self._ball.value()._radius(self._ball.value().data,radii,return_distance,False,sort_results,True,stats)
            return result.neighbors.take()
        var result = self._engine.radius_neighbors(rebind[Optional[nm.NDArray[Self.M.dtype]]](X), r, return_distance, sort_results)
        return rebind[RadiusResult[Self.dtype]](result).copy()
