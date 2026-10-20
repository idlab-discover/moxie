"""Streaming brute-force queries on the family's owned dense result records."""
import numojo as nm
from std.math import isnan
from moxie.neighbors._contracts import DenseMetric, DenseQueryBackend, TopKResult, RadiusResult, fit_snapshot, validate_matrix, validate_k, validate_radius_options
from moxie.neighbors._distance import at, SelfQueryMetric


def before(a: Float64, ai: Int, b: Float64, bi: Int) -> Bool:
    """Deterministic total ordering: defined distances, then NaNs, then index."""
    if isnan(a): return isnan(b) and ai < bi
    if isnan(b): return True
    return a < b or (a == b and ai < bi)


struct BruteForce[M: DenseMetric & Deinitable](DenseQueryBackend):
    comptime dtype = Self.M.dtype
    var _metric: Self.M
    var _fit_X: Optional[nm.NDArray[Self.dtype]]
    var n_samples_fit_: Int
    var n_features_in_: Int
    var _precomputed: Bool
    var _allow_nan: Bool
    var _chunk_size: Int

    def __init__(out self, metric: Self.M, chunk_size: Int = 256,
                 precomputed: Bool = False, allow_nan: Bool = False) raises:
        if chunk_size <= 0: raise Error("ValueError: chunk_size must be positive")
        self._metric = metric.copy()
        self._fit_X = None
        self.n_samples_fit_ = 0
        self.n_features_in_ = 0
        self._precomputed = precomputed
        self._allow_nan = allow_nan
        self._chunk_size = chunk_size

    def clone(self) raises -> Self:
        return Self(self._metric, self._chunk_size, self._precomputed, self._allow_nan)

    def fit(mut self, x: nm.NDArray[Self.dtype]) raises:
        var snapshot = fit_snapshot(x, self._precomputed, self._allow_nan)
        if not self._precomputed: self._metric.validate_features(x.shape[1])
        self.n_samples_fit_ = x.shape[0]
        self.n_features_in_ = x.shape[1]
        self._fit_X = snapshot^

    def _validate_query(self, x: Optional[nm.NDArray[Self.dtype]]) raises:
        if not self._fit_X: raise Error("NotFittedError: no indexed samples")
        if x:
            validate_matrix(x.value(), self.n_features_in_, self._precomputed, allow_nan=self._allow_nan)
            comptime if conforms_to(Self.M, SelfQueryMetric):
                comptime assert conforms_to(Self.M, SelfQueryMetric)
                if not self._precomputed: self._metric.validate_cross_query()

    def _query_metric(self, query_is_train: Bool) raises -> Self.M:
        """Per-call metric; self queries may derive parameters from fitted data."""
        comptime if conforms_to(Self.M, SelfQueryMetric):
            comptime assert conforms_to(Self.M, SelfQueryMetric)
            if query_is_train and not self._precomputed:
                return self._metric.self_query_metric(self._fit_X.value())
        return self._metric.copy()

    def _allows_undefined(self) -> Bool:
        comptime if conforms_to(Self.M, SelfQueryMetric):
            comptime assert conforms_to(Self.M, SelfQueryMetric)
            return self._metric.allows_undefined()
        return self._allow_nan

    def _distance(self, metric: Self.M, q: nm.NDArray[Self.dtype], i: Int, j: Int) raises -> Float64:
        var d: Float64
        if self._precomputed: d = at(q,i,j)
        else: d = metric.distance(q,i,self._fit_X.value(),j)
        if d < 0 or (isnan(d) and not self._allows_undefined()):
            raise Error("ValueError: metric returned a negative or undefined distance")
        return d

    def kneighbors(self, x: Optional[nm.NDArray[Self.dtype]], k: Int,
                   return_distance: Bool = True) raises -> TopKResult[Self.dtype]:
        validate_k(k,self.n_samples_fit_,not x)
        self._validate_query(x)
        if x: return self._kneighbors(x.value(),k,return_distance,False)
        return self._kneighbors(self._fit_X.value(),k,return_distance,True)

    def _kneighbors(self, q: nm.NDArray[Self.dtype], k: Int, return_distance: Bool,
                    query_is_train: Bool) raises -> TopKResult[Self.dtype]:
        var metric = self._query_metric(query_is_train)
        var nq = q.shape[0]
        var indices = List[Int]()
        var distances = Optional[nm.NDArray[Self.dtype]](None)
        var output = nm.zeros[Self.dtype](nm.Shape(nq if return_distance else 0,k))
        # One row of insertion-selection scratch: O(k), independent of n_fit.
        var best = nm.zeros[DType.float64](nm.Shape(k))
        var ids = List[Int]()
        for _ in range(k): ids.append(0)
        for i in range(nq):
            var count = 0
            for start in range(0,self.n_samples_fit_,self._chunk_size):
                var stop = min(start+self._chunk_size,self.n_samples_fit_)
                for j in range(start,stop):
                    if query_is_train and i == j: continue
                    var d = self._distance(metric,q,i,j)
                    var pos = count
                    if count == k:
                        if not before(d,j,best.unsafe_get(k-1),ids[k-1]): continue
                        pos = k-1
                    while pos > 0 and before(d,j,best.unsafe_get(pos-1),ids[pos-1]):
                        best.unsafe_set(pos,best.unsafe_get(pos-1))
                        ids[pos] = ids[pos-1]
                        pos -= 1
                    best.unsafe_set(pos,d)
                    ids[pos] = j
                    if count < k: count += 1
            for j in range(k):
                indices.append(ids[j])
                if return_distance:
                    output.unsafe_set(i*k+j,best.unsafe_get(j).cast[Self.dtype]())
        if return_distance: distances = output^
        var result = TopKResult[Self.dtype](nq,k,indices^,distances^)
        result.validate(self.n_samples_fit_,self._allows_undefined())
        return result^

    def radius_neighbors(self, x: Optional[nm.NDArray[Self.dtype]], radius: Float64,
                         return_distance: Bool = True, sort_results: Bool = False,
                         ) raises -> RadiusResult[Self.dtype]:
        validate_radius_options(radius,return_distance,sort_results)
        self._validate_query(x)
        if x: return self._radius(x.value(),radius,return_distance,sort_results,False)
        return self._radius(self._fit_X.value(),radius,return_distance,sort_results,True)

    def _radius(self, q: nm.NDArray[Self.dtype], radius: Float64,
                return_distance: Bool, sort_results: Bool, query_is_train: Bool,
                ) raises -> RadiusResult[Self.dtype]:
        var metric = self._query_metric(query_is_train)
        var offsets = List[Int]()
        var indices = List[Int]()
        offsets.append(0)
        # First pass collects membership. Numeric output is allocated exactly once.
        for i in range(q.shape[0]):
            for start in range(0,self.n_samples_fit_,self._chunk_size):
                for j in range(start,min(start+self._chunk_size,self.n_samples_fit_)):
                    if query_is_train and i == j: continue
                    if self._distance(metric,q,i,j) <= radius: indices.append(j)
            offsets.append(len(indices))
        var distances = Optional[nm.NDArray[Self.dtype]](None)
        if return_distance:
            var d = nm.zeros[Self.dtype](nm.Shape(len(indices)))
            for i in range(q.shape[0]):
                for pos in range(offsets[i],offsets[i+1]):
                    d.unsafe_set(pos,self._distance(metric,q,i,indices[pos]).cast[Self.dtype]())
                if sort_results:
                    # In-place row sorting needs no numeric scratch allocation.
                    for pos in range(offsets[i]+1,offsets[i+1]):
                        var value = d.unsafe_get(pos)
                        var idx = indices[pos]
                        var dest = pos
                        while dest > offsets[i] and before(Float64(value),idx,Float64(d.unsafe_get(dest-1)),indices[dest-1]):
                            d.unsafe_set(dest,d.unsafe_get(dest-1))
                            indices[dest] = indices[dest-1]
                            dest -= 1
                        d.unsafe_set(dest,value)
                        indices[dest] = idx
            distances = d^
        var result = RadiusResult[Self.dtype](offsets^,indices^,distances^)
        result.validate(self.n_samples_fit_,sort_results)
        return result^


def distance_weights[dtype: DType](distances: nm.NDArray[dtype], offsets: List[Int]) raises -> nm.NDArray[dtype]:
    """Inverse-distance weights for flat ragged rows, including zero dominance."""
    if distances.ndim != 1 or len(offsets) == 0 or offsets[0] != 0:
        raise Error("ValueError: expected flat distances and ragged offsets")
    if offsets[len(offsets)-1] != distances.shape[0]: raise Error("ValueError: offset endpoint mismatch")
    var output = nm.zeros[dtype](nm.Shape(distances.shape[0]))
    for i in range(len(offsets)-1):
        if offsets[i+1] < offsets[i] or offsets[i+1] > distances.shape[0]:
            raise Error("ValueError: invalid offsets")
        var zero = False
        for j in range(offsets[i],offsets[i+1]):
            var d = distances.unsafe_get(j * Int(distances.strides[0]))
            if isnan(d) or d < 0: raise Error("ValueError: invalid distance")
            if d == 0: zero = True
        for j in range(offsets[i],offsets[i+1]):
            var d = distances.unsafe_get(j * Int(distances.strides[0]))
            var w: Scalar[dtype]
            if zero: w = Scalar[dtype](1) if d == 0 else Scalar[dtype](0)
            else: w = 1/d
            output.unsafe_set(j,w)
    return output^


trait NeighborWeights(Copyable, Movable):
    def apply(self, distances: nm.NDArray[DType.float64], offsets: List[Int]) raises -> nm.NDArray[DType.float64]:
        ...

@fieldwise_init
struct BuiltinWeights(NeighborWeights):
    var name: String
    def apply(self, distances: nm.NDArray[DType.float64], offsets: List[Int]) raises -> nm.NDArray[DType.float64]:
        if self.name == "distance": return distance_weights(distances, offsets)
        if self.name != "uniform": raise Error("ValueError: weights must be uniform or distance; use predict_with_weights for native callables")
        return nm.ones[DType.float64](nm.Shape(distances.shape[0]))
