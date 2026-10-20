"""Exact dense kernel density with the family spatial trees."""
import numojo as nm
from std.utils import Variant
from std.math import log, isfinite, pow
from moxie.base import BaseEstimator
from moxie.neighbors._base import NeighborDensity
from moxie._random import RandomState, check_random_state
from moxie.neighbors._contracts import fit_snapshot, validate_matrix
from moxie.neighbors._distance import at
from moxie.neighbors._kde_sampling import sample_tophat
from moxie.neighbors._kd_tree import KDTree, kd_eligible
from moxie.neighbors._ball_tree import BallTree
from moxie.neighbors._distance import Metric
from moxie.neighbors._binary_tree import TreeStats

@fieldwise_init
struct KDEParams[dtype: DType](Copyable, Movable):
    var bandwidth: Float64
    var algorithm: String
    var kernel: String
    var metric: String
    var atol: Float64
    var rtol: Float64
    var breadth_first: Bool
    var leaf_size: Int
    var bandwidth_rule: String
    var metric_params: Optional[Metric[Self.dtype]]

struct KernelDensity[dtype: DType = DType.float64](NeighborDensity):
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime SampleScores = nm.NDArray[DType.float64]
    comptime Params = KDEParams[Self.dtype]
    var _params: Self.Params
    var tree_: Optional[Variant[KDTree[Self.dtype], BallTree[Self.dtype]]]
    var _X: Optional[nm.NDArray[Self.dtype]]
    var _weights: nm.NDArray[DType.float64]
    var _sum_weights: Float64
    var n_features_in_: Int
    var bandwidth_: Float64
    var _fit_method: String

    def __init__(out self, *, bandwidth: Float64 = 1, algorithm: String = "auto",
                 kernel: String = "gaussian", metric: String = "euclidean",
                 atol: Float64 = 0, rtol: Float64 = 0, breadth_first: Bool = True,
                 leaf_size: Int = 40, bandwidth_rule: String = "",
                 metric_params: Optional[Metric[Self.dtype]] = None) raises:
        self._params = Self.Params(bandwidth,algorithm,kernel,metric,atol,rtol,breadth_first,leaf_size,bandwidth_rule,metric_params.copy())
        self.tree_ = None
        self._X = None
        self._weights = nm.zeros[DType.float64](nm.Shape(0))
        self._sum_weights = 0
        self.n_features_in_ = 0
        self.bandwidth_ = bandwidth
        self._fit_method = "brute"

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self._params = params.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self._X)

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        self.fit(X,y,sample_weight=None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: nm.NDArray[Self.dtype], y: Optional[nm.NDArray[Self.dtype]] = None,
            *, sample_weight: Optional[nm.NDArray[Self.dtype]]) raises -> ref[self] Self:
        ref p = self._params
        if p.algorithm != "auto" and p.algorithm != "kd_tree" and p.algorithm != "ball_tree": raise Error("ValueError: invalid KDE algorithm")
        if p.leaf_size <= 0 or not isfinite(p.atol) or not isfinite(p.rtol) or p.atol < 0 or p.rtol < 0: raise Error("ValueError: invalid KDE tolerance or leaf size")
        if p.kernel != "gaussian" and p.kernel != "tophat" and p.kernel != "epanechnikov" and p.kernel != "exponential" and p.kernel != "linear" and p.kernel != "cosine": raise Error("ValueError: unknown kernel")
        var snapshot = fit_snapshot(X)
        var bandwidth = p.bandwidth
        if p.bandwidth_rule == "scott": bandwidth = pow(Float64(X.shape[0]),-Float64(1)/Float64(X.shape[1]+4))
        elif p.bandwidth_rule == "silverman": bandwidth = pow(Float64(X.shape[0])*Float64(X.shape[1]+2)/4,-Float64(1)/Float64(X.shape[1]+4))
        elif p.bandwidth_rule != "": raise Error("ValueError: bandwidth_rule must be scott or silverman")
        if not isfinite(bandwidth) or bandwidth <= 0: raise Error("ValueError: bandwidth must be positive and finite")
        var weights = nm.ones[DType.float64](nm.Shape(X.shape[0]))
        if sample_weight:
            ref w = sample_weight.value()
            if w.ndim != 1 or w.shape[0] != X.shape[0]: raise Error("ValueError: sample_weight shape mismatch")
            for i in range(X.shape[0]): weights.unsafe_set(i,Float64(w.unsafe_get(i * Int(w.strides[0]))))
        var total = Float64(0)
        for i in range(X.shape[0]):
            var w = weights.unsafe_get(i)
            if not isfinite(w) or w < 0: raise Error("ValueError: weights must be nonnegative and finite")
            total += w
        if total == 0: raise Error("ValueError: weights must have positive total")
        var metric = Metric[Self.dtype]("minkowski" if p.metric=="p" else p.metric)
        if p.metric_params: metric = p.metric_params.value().copy()
        var tree: Variant[KDTree[Self.dtype], BallTree[Self.dtype]]
        var method = String("kd_tree")
        if p.algorithm == "ball_tree" or (p.algorithm == "auto" and not kd_eligible(metric)):
            tree = BallTree[Self.dtype](X,p.leaf_size,p.metric,sample_weight.copy(),metric_params=metric.copy())
            method = "ball_tree"
        else: tree = KDTree[Self.dtype](X,p.leaf_size,p.metric,sample_weight.copy(),metric_params=metric.copy())
        self.tree_ = tree^
        self._fit_method = method
        self._X = snapshot^
        self._weights = weights^
        self._sum_weights = total
        self.n_features_in_ = X.shape[1]
        self.bandwidth_ = bandwidth
        return self

    def score_samples(self, X: nm.NDArray[Self.dtype]) raises -> nm.NDArray[DType.float64]:
        if not self._X: raise Error("NotFittedError: KDE has not been fitted")
        validate_matrix(X,self.n_features_in_)
        var stats = TreeStats()
        var output: nm.NDArray[DType.float64]
        if self._fit_method == "ball_tree":
            output = self.tree_.value()[BallTree[Self.dtype]]._density(X,self.bandwidth_,self._params.kernel,self._params.atol*self._sum_weights,self._params.rtol,self._params.breadth_first,True,stats)
        else:
            output = self.tree_.value()[KDTree[Self.dtype]]._density(X,self.bandwidth_,self._params.kernel,self._params.atol*self._sum_weights,self._params.rtol,self._params.breadth_first,True,stats)
        for i in range(X.shape[0]): output.unsafe_set(i,output.unsafe_get(i)-log(self._sum_weights))
        return output^

    def score(self, X: nm.NDArray[Self.dtype], y: Optional[nm.NDArray[Self.dtype]] = None) raises -> Float64:
        var scores = self.score_samples(X)
        var total = Float64(0)
        for i in range(scores.shape[0]): total += scores.unsafe_get(i)
        return total

    def sample(self, n_samples: Int = 1, random_state: Optional[Int] = None) raises -> nm.NDArray[DType.float64]:
        with check_random_state(random_state) as rng:
            return self.sample_with_rng(n_samples,rng[])

    def sample_with_rng(self, n_samples: Int, mut rng: RandomState) raises -> nm.NDArray[DType.float64]:
        if not self._X: raise Error("NotFittedError: KDE has not been fitted")
        if self._params.kernel != "gaussian" and self._params.kernel != "tophat": raise Error("NotImplementedError: sampling supports Gaussian and tophat kernels only")
        if n_samples <= 0: raise Error("ValueError: n_samples must be positive")
        var indices = List[Int]()
        for _ in range(n_samples):
            var threshold = rng.uniform()*self._sum_weights
            var cumulative = self._weights.unsafe_get(0)
            var chosen = 0
            while cumulative < threshold and chosen+1 < self._X.value().shape[0]:
                chosen += 1
                cumulative += self._weights.unsafe_get(chosen)
            indices.append(chosen)
        if self._params.kernel == "tophat":
            return sample_tophat(self._X.value(), indices, self.bandwidth_, rng)
        var result = nm.zeros[DType.float64](nm.Shape(n_samples,self.n_features_in_))
        for i in range(n_samples):
            for j in range(self.n_features_in_): result.unsafe_set(i*self.n_features_in_+j,rng.normal(at(self._X.value(),indices[i],j),self.bandwidth_))
        return result^
