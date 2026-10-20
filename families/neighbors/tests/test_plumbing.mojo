"""Native storage/validation contracts, not numerical query parity."""
import numojo as nm
from std.testing import assert_equal, assert_raises, assert_false, assert_true
from std.utils import Variant
from moxie.neighbors import KNeighborsQuery, RadiusNeighborsQuery
from moxie.neighbors import (
    TopKResult, RadiusResult, fit_snapshot, validate_matrix,
    validate_k, validate_radius_options, DenseMetric, DenseQueryBackend,
)

@fieldwise_init
struct MetricProbe(DenseMetric):
    comptime dtype = DType.float64
    def validate_features(self, n_features: Int) raises:
        if n_features <= 0:
            raise Error("ValueError: expected features")
    def distance(self, x: nm.NDArray[Self.dtype], x_row: Int,
                 y: nm.NDArray[Self.dtype], y_row: Int) raises -> Float64:
        raise Error("Contract probe: metric engine is not implemented")

@fieldwise_init
struct BackendProbe(DenseQueryBackend):
    comptime dtype = DType.float64
    def fit(mut self, x: nm.NDArray[Self.dtype]) raises:
        raise Error("Contract probe: query engine is not implemented")
    def kneighbors(self, x: Optional[nm.NDArray[Self.dtype]], k: Int,
                   return_distance: Bool = True) raises -> TopKResult[Self.dtype]:
        raise Error("Contract probe: query engine is not implemented")
    def radius_neighbors(self, x: Optional[nm.NDArray[Self.dtype]], radius: Float64,
                         return_distance: Bool = True, sort_results: Bool = False,
                         ) raises -> RadiusResult[Self.dtype]:
        raise Error("Contract probe: query engine is not implemented")


def backend_contract[B: DenseQueryBackend](mut b: B) raises:
    var x = nm.zeros[B.dtype](nm.Shape(1, 1))
    with assert_raises(): b.fit(x)
    with assert_raises(): _ = b.kneighbors(None, 1)
    with assert_raises(): _ = b.radius_neighbors(None, 1)


def metric_contract[M: DenseMetric](m: M) raises:
    var x = nm.zeros[M.dtype](nm.Shape(1, 1))
    m.validate_features(1)
    with assert_raises(): _ = m.distance(x, 0, x, 0)



@fieldwise_init
struct PublicProbe(KNeighborsQuery, RadiusNeighborsQuery):
    comptime Input = nm.NDArray[DType.float64]
    comptime Params = Int
    comptime NeighborResult = Variant[TopKResult[DType.float64], RadiusResult[DType.float64]]

    def get_params(self, deep: Bool = True) raises -> Int:
        return 0
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Int) raises -> ref[self] Self:
        pass
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return False
    def kneighbors(self, X: Optional[Self.Input] = None,
                   n_neighbors: Optional[Int] = None,
                   return_distance: Bool = True) raises -> Self.NeighborResult:
        raise Error("Contract probe: no search implemented")
    def radius_neighbors(self, X: Optional[Self.Input] = None,
                         radius: Optional[Float64] = None, return_distance: Bool = True,
                         sort_results: Bool = False) raises -> Self.NeighborResult:
        raise Error("Contract probe: no search implemented")


def public_contract(p: PublicProbe) raises:
    with assert_raises(): _ = p.kneighbors()
    with assert_raises(): _ = p.radius_neighbors()
    var ids: List[Int] = [0]
    var result = TopKResult[DType.float64](1, 1, ids^, None)
    var wrapped = PublicProbe.NeighborResult(result^)
    assert_true(wrapped.isa[TopKResult[DType.float64]]())
    wrapped[TopKResult[DType.float64]].validate(1)
    var offsets: List[Int] = [0, 0]
    var empty = RadiusResult[DType.float64](offsets^, List[Int](), None)
    wrapped = PublicProbe.NeighborResult(empty^)
    assert_true(wrapped.isa[RadiusResult[DType.float64]]())
    wrapped[RadiusResult[DType.float64]].validate(1)


def contracts[dtype: DType]() raises:
    var x = nm.full[dtype](nm.Shape(3, 2), 2)
    var fitted = fit_snapshot(x)
    var alias_view = x.view()
    alias_view.unsafe_set(0, 9)
    assert_equal(x.item(0, 0), 9)
    assert_equal(fitted.item(0, 0), 2)
    var strided = x.view_with_layout(nm.Shape(2, 1), nm.NDArrayStrides(2, 1), 2)
    var saved = fit_snapshot(strided)
    strided.unsafe_set(0, 7)
    assert_equal(saved.item(0, 0), 2)
    validate_matrix(strided, 1)
    var reversed_view = x.view_with_layout(nm.Shape(3, 2), nm.NDArrayStrides(-2, 1), 4)
    var reverse_copy = fit_snapshot(reversed_view)
    assert_equal(reverse_copy.item(2, 0), 9)
    with assert_raises(): validate_matrix(x, 3)
    with assert_raises(): _ = fit_snapshot(x, precomputed=True)
    var square = nm.zeros[dtype](nm.Shape(2, 2))
    validate_matrix(square, precomputed=True, training=True)
    square.unsafe_set(0, -1)
    with assert_raises(): validate_matrix(square, precomputed=True)
    var empty = nm.zeros[dtype](nm.Shape(0, 2))
    with assert_raises(): validate_matrix(empty)
    var vector = nm.zeros[dtype](nm.Shape(2))
    with assert_raises(): validate_matrix(vector)
    var special = nm.full[dtype](nm.Shape(1, 1), Float64("nan").cast[dtype]())
    with assert_raises(): validate_matrix(special)
    validate_matrix(special, allow_nan=True)
    with assert_raises(): validate_matrix(special, precomputed=True, allow_nan=True)
    special.unsafe_set(0, Float64("inf").cast[dtype]())
    with assert_raises(): validate_matrix(special, allow_nan=True)

    var indices: List[Int] = [0, 2, 1, 2]
    var distances = nm.full[dtype](nm.Shape(2, 2), 1)
    var top = TopKResult[dtype](2, 2, indices^, distances^)
    top.validate(3)
    var cloned = top.copy()
    cloned.indices[0] = 1
    cloned.distances.value().unsafe_set(0, 0)
    assert_equal(top.indices[0], 0)
    assert_equal(top.distances.value().item(0, 0), 1)
    cloned.validate(3)
    cloned.distances.value().unsafe_set(0, 2)
    with assert_raises(): cloned.validate(3)
    cloned.distances.value().unsafe_set(0, -0.5)
    with assert_raises(): cloned.validate(3)
    cloned.distances.value().unsafe_set(0, 0)
    cloned.distances.value().unsafe_set(1, Float64("nan").cast[dtype]())
    with assert_raises(): cloned.validate(3)
    cloned.validate(3, allow_nan=True)
    cloned.distances.value().unsafe_set(0, Float64("nan").cast[dtype]())
    cloned.distances.value().unsafe_set(1, 1)
    with assert_raises(): cloned.validate(3, allow_nan=True)
    cloned.distances = None
    cloned.validate(3)
    assert_false(Bool(cloned.distances))
    cloned.indices[0] = 3
    with assert_raises(): cloned.validate(3)
    cloned.indices[0] = 0
    cloned.n_queries = 3
    with assert_raises(): cloned.validate(3)
    cloned.n_queries = 2
    cloned.distances = nm.zeros[dtype](nm.Shape(4))
    with assert_raises(): cloned.validate(3)

    var offsets: List[Int] = [0, 0, 2, 2]
    var ri: List[Int] = [2, 0]
    var rd = nm.full[dtype](nm.Shape(2), 1)
    var ragged = RadiusResult[dtype](offsets^, ri^, rd^)
    ragged.validate(3, True)
    ragged.distances.value().unsafe_set(1, 0)
    ragged.validate(3)
    with assert_raises(): ragged.validate(3, True)
    var rc = ragged.copy()
    rc.offsets[1] = 1
    assert_equal(ragged.offsets[1], 0)
    rc.offsets[2] = 0
    with assert_raises(): rc.validate(3)
    rc.offsets[2] = 2
    rc.offsets[3] = 1
    with assert_raises(): rc.validate(3)
    rc.offsets[3] = 2
    rc.distances = nm.zeros[dtype](nm.Shape(3))
    with assert_raises(): rc.validate(3)
    rc.distances = None
    rc.validate(3)
    with assert_raises(): rc.validate(3, True)
    var eo: List[Int] = [0, 0]
    var ei = List[Int]()
    var er = RadiusResult[dtype](eo^, ei^, None)
    er.validate(3)


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    validate_k(3, 3, False)
    validate_k(2, 3, True)
    with assert_raises(): validate_k(3, 3, True)
    with assert_raises(): validate_k(0, 3, False)
    with assert_raises(): validate_k(1, 0, False)
    with assert_raises(): validate_k(1, 1, True)
    validate_radius_options(0, True, True)
    validate_radius_options(Float64("inf"), False, False)
    with assert_raises(): validate_radius_options(-1, True, False)
    with assert_raises(): validate_radius_options(Float64("nan"), True, False)
    with assert_raises(): validate_radius_options(1, False, True)
    var backend = BackendProbe()
    backend_contract(backend)
    metric_contract(MetricProbe())
    public_contract(PublicProbe())
    print("PASS: Float32/Float64 dense query contracts and raising extension probes")
