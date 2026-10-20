"""Shared median spatial tree traversal with box or ball geometry."""
import numojo as nm
from std.math import abs, max, min, isfinite, isnan, exp, log
from moxie.neighbors._distance import Metric, at, metric_power
from moxie.neighbors._query_engine import before
from moxie.neighbors._contracts import DenseMetric, TopKResult, RadiusResult, fit_snapshot, validate_matrix, validate_k
from moxie.neighbors._binary_tree import NodeData, TreeStats, TreeArrays, validate_kernel, log_kernel, log_kernel_norm

from moxie._linalg import covariance_inverse
from moxie.neighbors._metrics import MetricNames, kd_eligible, ball_eligible, ball_prunable

@fieldwise_init
struct TreeRadiusResult[dtype: DType](Copyable, Movable):
    """Exactly one of neighbors or counts is present, selected by count_only."""
    var neighbors: Optional[RadiusResult[Self.dtype]]
    var counts: Optional[List[Int]]

struct SpatialTree[dtype: DType = DType.float64, ball: Bool = False, M: DenseMetric & Deinitable = Metric[dtype]](Copyable, Movable):
    comptime valid_metrics = MetricNames[Self.ball]()
    var data: nm.NDArray[Self.dtype]
    var idx_array: List[Int]
    var node_data: List[NodeData]
    var node_bounds: nm.NDArray[Self.dtype]
    var sample_weight: Optional[nm.NDArray[Self.dtype]]
    var sum_weight: Float64
    var leaf_size: Int
    var _metric: Self.M
    var _power: Float64
    var _stats: TreeStats

    def __init__(out self, data: nm.NDArray[Self.dtype], leaf_size: Int = 40,
                 metric: String = "minkowski", sample_weight: Optional[nm.NDArray[Self.dtype]] = None,
                 *, p: Float64 = 2, metric_params: Optional[Metric[Self.dtype]] = None) raises:
        comptime assert Self.M == Metric[Self.dtype]
        var prepared = Metric[Self.dtype]("minkowski" if metric == "p" else metric,p)
        if metric_params: prepared = metric_params.value().copy()
        self = Self(data,leaf_size,sample_weight=sample_weight.copy(),_prepared_metric=rebind[Self.M](prepared).copy())

    def __init__(out self, data: nm.NDArray[Self.dtype], leaf_size: Int = 40,
                 *, metric_params: Self.M, sample_weight: Optional[nm.NDArray[Self.dtype]] = None) raises:
        self = Self(data,leaf_size,sample_weight=sample_weight.copy(),_prepared_metric=metric_params.copy())

    def __init__(out self, data: nm.NDArray[Self.dtype], leaf_size: Int = 40,
                 *, sample_weight: Optional[nm.NDArray[Self.dtype]] = None, _prepared_metric: Self.M) raises:
        self.data = fit_snapshot(data)
        comptime assert Self.M.dtype == Self.dtype, "metric dtype must match tree dtype"
        self._power = 2
        comptime if Self.M == Metric[Self.dtype]:
            comptime assert Self.M == Metric[Self.dtype]
            var prepared = rebind[Metric[Self.dtype]](_prepared_metric).copy()
            comptime if Self.ball:
                if prepared.name == "mahalanobis" and not prepared.VI and prepared.V:
                    prepared.VI = covariance_inverse(prepared.V.value().astype[DType.float64]()).astype[Self.dtype]()
                    prepared.V = None
                if not ball_eligible(prepared): raise Error("ValueError: metric is not supported by BallTree")
                prepared.validate_cross_query()
            else:
                if not kd_eligible(prepared): raise Error("ValueError: metric is not supported by KDTree (unweighted Minkowski p>=1 only)")
            self._power = prepared.p
            if prepared.name == "euclidean": self._power = 2
            elif prepared.name == "manhattan": self._power = 1
            elif prepared.name == "chebyshev": self._power = Float64("inf")
            self._metric = rebind[Self.M](prepared).copy()
        else:
            if not Self.ball: raise Error("ValueError: KDTree does not support callable metrics")
            self._metric = _prepared_metric.copy()
        self._metric.validate_features(data.shape[1])
        if leaf_size < 1: raise Error("ValueError: leaf_size must be positive")
        self.leaf_size = leaf_size
        self.sample_weight = sample_weight.copy()
        self.sum_weight = Float64(data.shape[0])
        if sample_weight:
            ref w = sample_weight.value()
            if w.ndim != 1 or w.shape[0] != data.shape[0]: raise Error("ValueError: sample_weight shape mismatch")
            self.sum_weight = 0
            for i in range(w.shape[0]):
                var weight = Float64(w.unsafe_get(i*Int(w.strides[0])))
                if not isfinite(weight) or weight < 0: raise Error("ValueError: sample weights must be finite and nonnegative")
                self.sum_weight += weight
        self.idx_array = List[Int]()
        for i in range(data.shape[0]): self.idx_array.append(i)
        var leaves = 1
        var ratio = (data.shape[0]-1)//leaf_size
        while ratio >= 2:
            leaves *= 2
            ratio //= 2
        self.node_data = List[NodeData]()
        for _ in range(2*leaves-1): self.node_data.append(NodeData(0,0,False,0))
        self.node_bounds = nm.zeros[Self.dtype](nm.Shape(1 if Self.ball else 2,2*leaves-1,data.shape[1]))
        self._stats = TreeStats()
        var scratch = self.idx_array.copy()
        self._build(0,0,data.shape[0],scratch)

    def _prunable(self) -> Bool:
        comptime if Self.M == Metric[Self.dtype]:
            comptime assert Self.M == Metric[Self.dtype]
            return ball_prunable(rebind[Metric[Self.dtype]](self._metric))
        else:
            # Native callbacks need not satisfy the triangle inequality.
            return False

    def _distance(self, x: nm.NDArray[Self.dtype], i: Int, y: nm.NDArray[Self.dtype], j: Int) raises -> Float64:
        comptime assert Self.M.dtype == Self.dtype
        return self._metric.distance(rebind[nm.NDArray[Self.M.dtype]](x),i,rebind[nm.NDArray[Self.M.dtype]](y),j)

    def _point_distance(self, X: nm.NDArray[Self.dtype], row: Int, index: Int) raises -> Float64:
        var value = self._distance(X,row,self.data,index)
        if isnan(value) or value < 0: raise Error("ValueError: tree metric produced an undefined or negative distance")
        return value

    def _combine(self, total: Float64, delta: Float64) raises -> Float64:
        if self._power == Float64("inf"): return max(total,delta)
        return total+metric_power(delta,self._power)

    def _root(self, total: Float64) raises -> Float64:
        if self._power == Float64("inf"): return total
        return metric_power(total,1/self._power)

    def _bound(self, side: Int, node: Int, axis: Int) raises -> Float64:
        return Float64(self.node_bounds.unsafe_get((side*len(self.node_data)+node)*self.data.shape[1]+axis))

    def _less(self, a: Int, b: Int, axis: Int) raises -> Bool:
        var x = at(self.data,a,axis)
        var y = at(self.data,b,axis)
        return x < y or (x == y and a < b)

    def _sort(mut self, lo: Int, hi: Int, axis: Int, mut scratch: List[Int]) raises:
        if hi-lo <= 1: return
        var mid = lo+(hi-lo)//2
        self._sort(lo,mid,axis,scratch)
        self._sort(mid,hi,axis,scratch)
        var a = lo
        var b = mid
        for i in range(lo,hi):
            if b == hi or (a < mid and self._less(self.idx_array[a],self.idx_array[b],axis)):
                scratch[i] = self.idx_array[a]
                a += 1
            else:
                scratch[i] = self.idx_array[b]
                b += 1
        for i in range(lo,hi): self.idx_array[i] = scratch[i]

    def _build(mut self, node: Int, lo: Int, hi: Int, mut scratch: List[Int]) raises:
        var axis = 0
        var spread = Float64(-1)
        var radius = Float64(0)
        var d = self.data.shape[1]
        for j in range(d):
            var lower = Float64("inf")
            var upper = Float64("-inf")
            for i in range(lo,hi):
                var value = at(self.data,self.idx_array[i],j)
                lower = min(lower,value)
                upper = max(upper,value)
            comptime if not Self.ball:
                self.node_bounds.unsafe_set(node*d+j,lower.cast[Self.dtype]())
                self.node_bounds.unsafe_set((len(self.node_data)+node)*d+j,upper.cast[Self.dtype]())
                radius = self._combine(radius,upper/2-lower/2)
            else:
                var total_weight = Float64(0)
                if self.sample_weight:
                    for i in range(lo,hi): total_weight += Float64(self.sample_weight.value().unsafe_get(self.idx_array[i]*Int(self.sample_weight.value().strides[0])))
                var center = Float64(0)
                for i in range(lo,hi):
                    var weight = Float64(1)/Float64(hi-lo)
                    if self.sample_weight and total_weight > 0:
                        weight = Float64(self.sample_weight.value().unsafe_get(self.idx_array[i]*Int(self.sample_weight.value().strides[0])))/total_weight
                    center += at(self.data,self.idx_array[i],j)*weight
                self.node_bounds.unsafe_set(node*d+j,center.cast[Self.dtype]())
            if upper-lower > spread:
                spread = upper-lower
                axis = j
        var leaf = 2*node+1 >= len(self.node_data)
        comptime if Self.ball:
            var centers = self.node_bounds.view_with_layout(nm.Shape(len(self.node_data),d),nm.NDArrayStrides(d,1),0)
            for i in range(lo,hi): radius = max(radius,self._distance(centers,node,self.data,self.idx_array[i]))
            self.node_data[node] = NodeData(lo,hi,leaf,radius)
        else:
            self.node_data[node] = NodeData(lo,hi,leaf,self._root(radius))
        if leaf: return
        self._sort(lo,hi,axis,scratch)
        var mid = lo+(hi-lo)//2
        self._build(2*node+1,lo,mid,scratch)
        self._build(2*node+2,mid,hi,scratch)

    def _bounds(self, X: nm.NDArray[Self.dtype], row: Int, node: Int) raises -> Tuple[Float64,Float64]:
        comptime if Self.ball:
            if not self._prunable(): return (Float64(0),Float64("inf"))
            var centers = self.node_bounds.view_with_layout(nm.Shape(len(self.node_data),self.data.shape[1]),nm.NDArrayStrides(self.data.shape[1],1),0)
            var distance = self._distance(X,row,centers,node)
            var radius = self.node_data[node].radius
            if not isfinite(distance) or not isfinite(radius): return (Float64(0),Float64("inf"))
            var guard = (abs(distance)+radius)*1e-12
            return (max(Float64(0),distance-radius-guard),distance+radius+guard)
        var lower = Float64(0)
        var upper = Float64(0)
        for j in range(self.data.shape[1]):
            var x = at(X,row,j)
            var lo = self._bound(0,node,j)
            var hi = self._bound(1,node,j)
            lower = self._combine(lower,max(Float64(0),max(lo-x,x-hi)))
            upper = self._combine(upper,max(abs(x-lo),abs(x-hi)))
        return (self._root(lower),self._root(upper))

    def _prune(self, lower: Float64, cutoff: Float64) -> Bool:
        # Conservative guard for different rounding paths at a closed boundary.
        return lower > cutoff and lower-cutoff > abs(lower)*Float64(1e-14)

    def _add_stats(mut self, stats: TreeStats):
        self._stats.calls += stats.calls
        self._stats.trims += stats.trims
        self._stats.leaves += stats.leaves
        self._stats.splits += stats.splits

    def get_n_calls(self) -> Int: return self._stats.calls
    def reset_n_calls(mut self): self._stats.calls = 0
    def get_tree_stats(self) -> Tuple[Int,Int,Int]:
        return (self._stats.trims,self._stats.leaves,self._stats.splits)
    def get_arrays(self) -> TreeArrays[Self.dtype]:
        return TreeArrays(self.data.copy(),self.idx_array.copy(),self.node_data.copy(),self.node_bounds.copy())

    def _insert(self, row: Int, k: Int, id: Int, distance: Float64,
                mut best: nm.NDArray[DType.float64], mut ids: List[Int]) raises:
        var start = row*k
        var pos = start+k-1
        if not before(distance,id,best.unsafe_get(pos),ids[pos]): return
        while pos > start and before(distance,id,best.unsafe_get(pos-1),ids[pos-1]):
            best.unsafe_set(pos,best.unsafe_get(pos-1))
            ids[pos] = ids[pos-1]
            pos -= 1
        best.unsafe_set(pos,distance)
        ids[pos] = id

    def _query(self, X: nm.NDArray[Self.dtype], k: Int, return_distance: Bool,
               breadth_first: Bool, exclude_self: Bool, mut stats: TreeStats) raises -> TopKResult[Self.dtype]:
        validate_matrix(X,self.data.shape[1])
        validate_k(k,self.data.shape[0],exclude_self)
        var best = nm.full[DType.float64](nm.Shape(X.shape[0],k),Float64("inf"))
        var ids = List[Int]()
        for _ in range(X.shape[0]*k): ids.append(self.data.shape[0])
        for i in range(X.shape[0]):
            var nodes = List[Int]()
            nodes.append(0)
            var cursor = 0
            while cursor < len(nodes):
                var node: Int
                if breadth_first:
                    node = nodes[cursor]
                    cursor += 1
                else: node = nodes.pop()
                var lower = self._bounds(X,i,node)[0]
                if self._prune(lower,best.unsafe_get(i*k+k-1)):
                    stats.trims += 1
                    continue
                var info = self.node_data[node]
                if info.is_leaf:
                    stats.leaves += 1
                    for pos in range(info.idx_start,info.idx_end):
                        var id = self.idx_array[pos]
                        if exclude_self and id == i: continue
                        var distance = self._point_distance(X,i,id)
                        stats.calls += 1
                        self._insert(i,k,id,distance,best,ids)
                else:
                    stats.splits += 1
                    var left = 2*node+1
                    var right = left+1
                    if not breadth_first and self._bounds(X,i,left)[0] < self._bounds(X,i,right)[0]:
                        nodes.append(right)
                        nodes.append(left)
                    else:
                        nodes.append(left)
                        nodes.append(right)
        var distances = Optional[nm.NDArray[Self.dtype]](None)
        if return_distance: distances = best.astype[Self.dtype]()
        return TopKResult[Self.dtype](X.shape[0],k,ids^,distances^)

    def query(mut self, X: nm.NDArray[Self.dtype], k: Int = 1, return_distance: Bool = True,
              dualtree: Bool = False, breadth_first: Bool = False, sort_results: Bool = True) raises -> TopKResult[Self.dtype]:
        var stats = TreeStats()
        var result: TopKResult[Self.dtype]
        if dualtree: result = self._dual_query(X,k,return_distance,breadth_first,stats)
        else: result = self._query(X,k,return_distance,breadth_first,False,stats)
        # Sorted output is also a valid unspecified ordering when sort_results=False.
        self._add_stats(stats)
        return result^

    def _pair_bound(self, other: Self, node: Int, query_node: Int) raises -> Float64:
        comptime if Self.ball:
            if not self._prunable(): return 0
            var d = self.data.shape[1]
            var centers = self.node_bounds.view_with_layout(nm.Shape(len(self.node_data),d),nm.NDArrayStrides(d,1),0)
            var others = other.node_bounds.view_with_layout(nm.Shape(len(other.node_data),d),nm.NDArrayStrides(d,1),0)
            var distance = self._distance(centers,node,others,query_node)
            var radius = self.node_data[node].radius+other.node_data[query_node].radius
            if not isfinite(distance) or not isfinite(radius): return 0
            return max(Float64(0),distance-radius-(abs(distance)+radius)*1e-12)
        var total = Float64(0)
        for j in range(self.data.shape[1]):
            var delta = max(Float64(0),max(self._bound(0,node,j)-other._bound(1,query_node,j),other._bound(0,query_node,j)-self._bound(1,node,j)))
            total = self._combine(total,delta)
        return self._root(total)

    def _dual_query(self, X: nm.NDArray[Self.dtype], k: Int, return_distance: Bool,
                    breadth_first: Bool, mut stats: TreeStats) raises -> TopKResult[Self.dtype]:
        validate_matrix(X,self.data.shape[1])
        validate_k(k,self.data.shape[0],False)
        var other = Self(X,self.leaf_size,metric_params=self._metric.copy())
        var best = nm.full[DType.float64](nm.Shape(X.shape[0],k),Float64("inf"))
        var ids = List[Int]()
        for _ in range(X.shape[0]*k): ids.append(self.data.shape[0])
        var pairs = List[Tuple[Int,Int]]()
        pairs.append((0,0))
        var cursor = 0
        while cursor < len(pairs):
            var pair: Tuple[Int,Int]
            if breadth_first:
                pair = pairs[cursor]
                cursor += 1
            else: pair = pairs.pop()
            var node = pair[0]
            var qnode = pair[1]
            var info = self.node_data[node]
            var qinfo = other.node_data[qnode]
            var cutoff = Float64(0)
            for pos in range(qinfo.idx_start,qinfo.idx_end):
                cutoff = max(cutoff,best.unsafe_get(other.idx_array[pos]*k+k-1))
            if self._prune(self._pair_bound(other,node,qnode),cutoff):
                stats.trims += 1
                continue
            if info.is_leaf and qinfo.is_leaf:
                stats.leaves += 1
                for qpos in range(qinfo.idx_start,qinfo.idx_end):
                    var i = other.idx_array[qpos]
                    for pos in range(info.idx_start,info.idx_end):
                        var id = self.idx_array[pos]
                        var distance = self._point_distance(X,i,id)
                        stats.calls += 1
                        self._insert(i,k,id,distance,best,ids)
            else:
                stats.splits += 1
                if qinfo.is_leaf or (not info.is_leaf and info.radius >= qinfo.radius):
                    pairs.append((2*node+2,qnode))
                    pairs.append((2*node+1,qnode))
                else:
                    pairs.append((node,2*qnode+2))
                    pairs.append((node,2*qnode+1))
        var distances = Optional[nm.NDArray[Self.dtype]](None)
        if return_distance: distances = best.astype[Self.dtype]()
        return TopKResult[Self.dtype](X.shape[0],k,ids^,distances^)

    def _radius(self, X: nm.NDArray[Self.dtype], radii: nm.NDArray[DType.float64],
                return_distance: Bool, count_only: Bool, sort_results: Bool,
                exclude_self: Bool, mut stats: TreeStats) raises -> TreeRadiusResult[Self.dtype]:
        validate_matrix(X,self.data.shape[1])
        if count_only and return_distance: raise Error("ValueError: count_only conflicts with return_distance")
        if sort_results and not return_distance: raise Error("ValueError: sorting requires distances")
        if radii.ndim != 1 or (radii.shape[0] != 1 and radii.shape[0] != X.shape[0]): raise Error("ValueError: radii must be scalar or one per query")
        var ids = List[Int]()
        var offsets = List[Int]()
        var counts = List[Int]()
        offsets.append(0)
        for i in range(X.shape[0]):
            var radius = radii.unsafe_get((0 if radii.shape[0] == 1 else i)*Int(radii.strides[0]))
            if isnan(radius): raise Error("ValueError: radius must not be NaN")
            var count = 0
            var nodes = List[Int]()
            nodes.append(0)
            while len(nodes) > 0:
                var node = nodes.pop()
                var bounds = self._bounds(X,i,node)
                if radius < 0 or self._prune(bounds[0],radius): continue
                var info = self.node_data[node]
                if isfinite(bounds[1]) and bounds[1] <= radius:
                    for pos in range(info.idx_start,info.idx_end):
                        var id = self.idx_array[pos]
                        if exclude_self and id == i: continue
                        count += 1
                        if not count_only: ids.append(id)
                elif info.is_leaf:
                    for pos in range(info.idx_start,info.idx_end):
                        var id = self.idx_array[pos]
                        if exclude_self and id == i: continue
                        var distance = self._point_distance(X,i,id)
                        stats.calls += 1
                        if distance <= radius:
                            count += 1
                            if not count_only: ids.append(id)
                else:
                    nodes.append(2*node+2)
                    nodes.append(2*node+1)
            counts.append(count)
            offsets.append(len(ids))
        if count_only: return TreeRadiusResult[Self.dtype](None,counts^)
        var distances = Optional[nm.NDArray[Self.dtype]](None)
        if return_distance:
            var output = nm.zeros[Self.dtype](nm.Shape(len(ids)))
            for i in range(X.shape[0]):
                for pos in range(offsets[i],offsets[i+1]):
                    output.unsafe_set(pos,self._point_distance(X,i,ids[pos]).cast[Self.dtype]())
                    stats.calls += 1
                if sort_results:
                    for pos in range(offsets[i]+1,offsets[i+1]):
                        var id = ids[pos]
                        var value = output.unsafe_get(pos)
                        var dest = pos
                        while dest > offsets[i] and before(Float64(value),id,Float64(output.unsafe_get(dest-1)),ids[dest-1]):
                            output.unsafe_set(dest,output.unsafe_get(dest-1))
                            ids[dest] = ids[dest-1]
                            dest -= 1
                        output.unsafe_set(dest,value)
                        ids[dest] = id
            distances = output^
        return TreeRadiusResult[Self.dtype](RadiusResult[Self.dtype](offsets^,ids^,distances^),None)

    def query_radius(mut self, X: nm.NDArray[Self.dtype], r: Float64,
                     return_distance: Bool = False, count_only: Bool = False,
                     sort_results: Bool = False) raises -> TreeRadiusResult[Self.dtype]:
        var radii = nm.full[DType.float64](nm.Shape(1),r)
        return self.query_radius(X,radii,return_distance,count_only,sort_results)

    def query_radius(mut self, X: nm.NDArray[Self.dtype], r: nm.NDArray[DType.float64],
                     return_distance: Bool = False, count_only: Bool = False,
                     sort_results: Bool = False) raises -> TreeRadiusResult[Self.dtype]:
        var stats = TreeStats()
        var result = self._radius(X,r,return_distance,count_only,sort_results,False,stats)
        self._add_stats(stats)
        return result^

    def _density(self, X: nm.NDArray[Self.dtype], h: Float64, kernel: String,
                 atol: Float64, rtol: Float64, breadth_first: Bool, return_log: Bool,
                 mut stats: TreeStats) raises -> nm.NDArray[DType.float64]:
        validate_matrix(X,self.data.shape[1])
        validate_kernel(h,kernel,atol,rtol)
        var result = nm.zeros[DType.float64](nm.Shape(X.shape[0]))
        var normalization = log_kernel_norm(self.data.shape[1],h,kernel)
        var compact = kernel != "gaussian" and kernel != "exponential"
        for i in range(X.shape[0]):
            var highest = Float64("-inf")
            var scaled = Float64(0)
            var nodes = List[Int]()
            nodes.append(0)
            var cursor = 0
            while cursor < len(nodes):
                var node: Int
                if breadth_first:
                    node = nodes[cursor]
                    cursor += 1
                else: node = nodes.pop()
                if compact and self._prune(self._bounds(X,i,node)[0],h): continue
                var info = self.node_data[node]
                if not info.is_leaf:
                    nodes.append(2*node+2)
                    nodes.append(2*node+1)
                    continue
                for pos in range(info.idx_start,info.idx_end):
                    var id = self.idx_array[pos]
                    var weight = Float64(1)
                    if self.sample_weight: weight = Float64(self.sample_weight.value().unsafe_get(id*Int(self.sample_weight.value().strides[0])))
                    if weight == 0: continue
                    var distance = self._point_distance(X,i,id)
                    stats.calls += 1
                    var value = log_kernel(distance,h,kernel)+log(weight)
                    if value == Float64("-inf"): continue
                    if value > highest:
                        scaled = scaled*exp(highest-value)+1
                        highest = value
                    else: scaled += exp(value-highest)
            var value = highest+log(scaled)+normalization if scaled > 0 else Float64("-inf")
            result.unsafe_set(i,value if return_log else exp(value))
        return result^

    def kernel_density(mut self, X: nm.NDArray[Self.dtype], h: Float64,
                       kernel: String = "gaussian", atol: Float64 = 0, rtol: Float64 = 1e-8,
                       breadth_first: Bool = True, return_log: Bool = False) raises -> nm.NDArray[Self.dtype]:
        var stats = TreeStats()
        var result = self._density(X,h,kernel,atol,rtol,breadth_first,return_log,stats)
        self._add_stats(stats)
        return result.astype[Self.dtype]()

    def two_point_correlation(mut self, X: nm.NDArray[Self.dtype], r: nm.NDArray[DType.float64],
                              dualtree: Bool = False) raises -> List[Int]:
        validate_matrix(X,self.data.shape[1])
        if r.ndim != 1: raise Error("ValueError: radii must be rank one")
        var counts = List[Int]()
        var stats = TreeStats()
        if dualtree:
            var other = Self(X,self.leaf_size,metric_params=self._metric.copy())
            for ri in range(r.shape[0]):
                var radius = r.unsafe_get(ri*Int(r.strides[0]))
                if isnan(radius): raise Error("ValueError: radius must not be NaN")
                var count = 0
                var pairs = List[Tuple[Int,Int]]()
                pairs.append((0,0))
                while len(pairs) > 0:
                    var pair = pairs.pop()
                    var node = pair[0]
                    var qnode = pair[1]
                    if radius < 0 or self._prune(self._pair_bound(other,node,qnode),radius): continue
                    var info = self.node_data[node]
                    var qinfo = other.node_data[qnode]
                    if info.is_leaf and qinfo.is_leaf:
                        for qi in range(qinfo.idx_start,qinfo.idx_end):
                            for ti in range(info.idx_start,info.idx_end):
                                if self._point_distance(X,other.idx_array[qi],self.idx_array[ti]) <= radius: count += 1
                                stats.calls += 1
                    elif qinfo.is_leaf or (not info.is_leaf and info.radius >= qinfo.radius):
                        pairs.append((2*node+1,qnode))
                        pairs.append((2*node+2,qnode))
                    else:
                        pairs.append((node,2*qnode+1))
                        pairs.append((node,2*qnode+2))
                counts.append(count)
        else:
            for ri in range(r.shape[0]):
                var radii = nm.full[DType.float64](nm.Shape(1),r.unsafe_get(ri*Int(r.strides[0])))
                var result = self._radius(X,radii,False,True,False,False,stats)
                var total = 0
                for count in result.counts.value(): total += count
                counts.append(total)
        self._add_stats(stats)
        return counts^
