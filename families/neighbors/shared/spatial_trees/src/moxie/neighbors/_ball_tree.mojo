"""Public tree adapter; search and geometry remain in the shared engine."""
import numojo as nm
from moxie.neighbors._tree_engine import SpatialTree, TreeRadiusResult
from moxie.neighbors._contracts import DenseMetric, TopKResult
from moxie.neighbors._distance import Metric
from moxie.neighbors._binary_tree import NodeData, TreeStats, TreeArrays
from moxie.neighbors._metrics import MetricNames, kd_eligible, ball_eligible

struct BallTree[dtype: DType = DType.float64, M: DenseMetric & Deinitable = Metric[dtype]](Copyable, Movable):
    comptime valid_metrics = MetricNames[True]()
    var _engine: SpatialTree[Self.dtype,True,Self.M]
    var data: nm.NDArray[Self.dtype]
    var idx_array: List[Int]
    var node_data: List[NodeData]
    var node_bounds: nm.NDArray[Self.dtype]
    var sample_weight: Optional[nm.NDArray[Self.dtype]]
    var sum_weight: Float64
    var leaf_size: Int

    def __init__[d: DType](out self: BallTree[d,Metric[d]], data: nm.NDArray[d], leaf_size: Int = 40,
                 metric: String = "minkowski", sample_weight: Optional[nm.NDArray[d]] = None,
                 *, p: Float64 = 2, metric_params: Optional[Metric[d]] = None) raises:
        self._engine = SpatialTree[d,True,Metric[d]](data,leaf_size,metric,sample_weight.copy(),p=p,metric_params=metric_params.copy())
        self.data = self._engine.data.view()
        self.idx_array = self._engine.idx_array.copy()
        self.node_data = self._engine.node_data.copy()
        self.node_bounds = self._engine.node_bounds.view()
        self.sample_weight = self._engine.sample_weight.copy()
        self.sum_weight = self._engine.sum_weight
        self.leaf_size = self._engine.leaf_size

    def __init__(out self, data: nm.NDArray[Self.dtype], leaf_size: Int = 40,
                 *, metric_params: Self.M, sample_weight: Optional[nm.NDArray[Self.dtype]] = None) raises:
        self._engine = SpatialTree[Self.dtype,True,Self.M](data,leaf_size,_prepared_metric=metric_params.copy(),sample_weight=sample_weight.copy())
        self.data = self._engine.data.view()
        self.idx_array = self._engine.idx_array.copy()
        self.node_data = self._engine.node_data.copy()
        self.node_bounds = self._engine.node_bounds.view()
        self.sample_weight = self._engine.sample_weight.copy()
        self.sum_weight = self._engine.sum_weight
        self.leaf_size = self._engine.leaf_size

    def get_n_calls(self) -> Int:
        return self._engine.get_n_calls()

    def reset_n_calls(mut self):
        self._engine.reset_n_calls()

    def get_tree_stats(self) -> Tuple[Int,Int,Int]:
        return self._engine.get_tree_stats()

    def get_arrays(self) -> TreeArrays[Self.dtype]:
        return self._engine.get_arrays()

    def _query(self, X: nm.NDArray[Self.dtype], k: Int, return_distance: Bool,
               breadth_first: Bool, exclude_self: Bool, mut stats: TreeStats) raises -> TopKResult[Self.dtype]:
        return self._engine._query(X,k,return_distance,breadth_first,exclude_self,stats)

    def query(mut self, X: nm.NDArray[Self.dtype], k: Int = 1, return_distance: Bool = True,
              dualtree: Bool = False, breadth_first: Bool = False, sort_results: Bool = True) raises -> TopKResult[Self.dtype]:
        return self._engine.query(X,k,return_distance,dualtree,breadth_first,sort_results)

    def _radius(self, X: nm.NDArray[Self.dtype], radii: nm.NDArray[DType.float64],
                return_distance: Bool, count_only: Bool, sort_results: Bool,
                exclude_self: Bool, mut stats: TreeStats) raises -> TreeRadiusResult[Self.dtype]:
        return self._engine._radius(X,radii,return_distance,count_only,sort_results,exclude_self,stats)

    def query_radius(mut self, X: nm.NDArray[Self.dtype], r: Float64,
                     return_distance: Bool = False, count_only: Bool = False,
                     sort_results: Bool = False) raises -> TreeRadiusResult[Self.dtype]:
        return self._engine.query_radius(X,r,return_distance,count_only,sort_results)

    def query_radius(mut self, X: nm.NDArray[Self.dtype], r: nm.NDArray[DType.float64],
                     return_distance: Bool = False, count_only: Bool = False,
                     sort_results: Bool = False) raises -> TreeRadiusResult[Self.dtype]:
        return self._engine.query_radius(X,r,return_distance,count_only,sort_results)

    def _density(self, X: nm.NDArray[Self.dtype], h: Float64, kernel: String,
                 atol: Float64, rtol: Float64, breadth_first: Bool, return_log: Bool,
                 mut stats: TreeStats) raises -> nm.NDArray[DType.float64]:
        return self._engine._density(X,h,kernel,atol,rtol,breadth_first,return_log,stats)

    def kernel_density(mut self, X: nm.NDArray[Self.dtype], h: Float64,
                       kernel: String = "gaussian", atol: Float64 = 0, rtol: Float64 = 1e-8,
                       breadth_first: Bool = True, return_log: Bool = False) raises -> nm.NDArray[Self.dtype]:
        return self._engine.kernel_density(X,h,kernel,atol,rtol,breadth_first,return_log)

    def two_point_correlation(mut self, X: nm.NDArray[Self.dtype], r: nm.NDArray[DType.float64],
                              dualtree: Bool = False) raises -> List[Int]:
        return self._engine.two_point_correlation(X,r,dualtree)
