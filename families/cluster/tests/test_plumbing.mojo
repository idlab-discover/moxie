from std.testing import assert_equal, assert_raises
import numojo as nm
from moxie.cluster._contracts import Labels, CentroidResult, MergeTree, Neighborhoods, validate_labels, validate_centers, snapshot_centers, validate_matrix, DenseDistanceAdapter, RadiusQueryAdapter
from moxie.cluster._contracts import AssignmentResult, CenterUpdate, CentroidAssignment, CentroidUpdate
from moxie._dense import row_squared_norms
from moxie._random import RandomState

@fieldwise_init
struct DistanceProbe(DenseDistanceAdapter):
    comptime dtype = DType.float64
    def pairwise(self, x: nm.NDArray[Self.dtype], y: nm.NDArray[Self.dtype]) raises -> nm.NDArray[Self.dtype]:
        raise Error("contract probe: distance engine deferred")

@fieldwise_init
struct RadiusProbe(RadiusQueryAdapter):
    comptime dtype = DType.float64
    def radius_neighbors(self, x: nm.NDArray[Self.dtype], radius: Float64) raises -> Neighborhoods:
        raise Error("contract probe: query engine deferred")

@fieldwise_init
struct CentroidProbe(CentroidAssignment, CentroidUpdate):
    comptime dtype = DType.float64
    def assign(self, x: nm.NDArray[Self.dtype], centers: nm.NDArray[Self.dtype], weights: nm.NDArray[Self.dtype]) raises -> AssignmentResult:
        raise Error("contract probe: assignment engine deferred")
    def update(self, x: nm.NDArray[Self.dtype], labels: Labels, weights: nm.NDArray[Self.dtype], previous: nm.NDArray[Self.dtype]) raises -> CenterUpdate[Self.dtype]:
        raise Error("contract probe: update engine deferred")

def dense[dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(2, 2))
    x[nm.Item(0, 0)] = 3
    var centers = snapshot_centers(x)
    var view = x.view()
    view[nm.Item(0, 0)] = 4
    assert_equal(centers.item(0, 0), 3)
    assert_equal(x.item(0, 0), 4)
    validate_centers(x, centers)
    var result = CentroidResult[dtype](nm.zeros[DType.int64](nm.Shape(2)), centers^, 0, 1)
    var copied = result.copy()
    result.centers[nm.Item(0, 0)] = 7
    assert_equal(copied.centers.item(0, 0), 3)
    var norms = row_squared_norms(copied.centers)
    assert_equal(norms.item(0), 9)
    var strided = x.view_with_layout(nm.Shape(1, 2), nm.NDArrayStrides(2, -1), 1)
    var strided_copy = snapshot_centers(strided)
    strided[nm.Item(0, 0)] = 8
    assert_equal(strided_copy.item(0, 0), 0)
    assert_equal(x.item(0, 1), 8)
    var bad = nm.zeros[dtype](nm.Shape(1, 3))
    with assert_raises():
        validate_centers(x, bad)
    var empty = nm.zeros[dtype](nm.Shape(0, 2))
    with assert_raises():
        validate_matrix(empty)
    var vector = nm.zeros[dtype](nm.Shape(2))
    with assert_raises():
        validate_matrix(vector)

def main() raises:
    dense[DType.float32]()
    dense[DType.float64]()
    var labels = nm.zeros[DType.int64](nm.Shape(2))
    validate_labels(labels, 2, 1)
    labels[nm.Item(0)] = -1
    validate_labels(labels, 2, 1, -1)
    with assert_raises():
        validate_labels(labels, 2, 1)
    labels[nm.Item(0)] = -3
    validate_labels(labels, 2, 1, -3)
    with assert_raises():
        validate_labels(labels, 2, 1, -1)
    with assert_raises():
        validate_labels(labels, 3, 1, -3)
    labels[nm.Item(0)] = 1
    with assert_raises():
        validate_labels(labels, 2, 1)
    labels[nm.Item(0)] = -1
    labels[nm.Item(1)] = -1
    validate_labels(labels, 2, 0, -1)
    labels[nm.Item(0)] = -2
    validate_labels(labels, 2, 0, -3)
    labels[nm.Item(0)] = -4
    with assert_raises():
        validate_labels(labels, 2, 0, -3)
    with assert_raises():
        validate_labels(labels, 2, 0, -2)
    var singleton = MergeTree(nm.zeros[DType.int64](nm.Shape(0, 2)), 1, 1, None, None)
    singleton.validate()
    var children = nm.zeros[DType.int64](nm.Shape(2, 2))
    children[nm.Item(0, 1)] = 1
    children[nm.Item(1, 0)] = 2
    children[nm.Item(1, 1)] = 3
    var tree = MergeTree(children^, 3, 1, None, None)
    tree.validate()
    var parents = nm.zeros[DType.int64](nm.Shape(5))
    parents[nm.Item(0)] = 3
    parents[nm.Item(1)] = 3
    parents[nm.Item(2)] = 4
    parents[nm.Item(3)] = 4
    parents[nm.Item(4)] = 4
    tree.parents = parents^
    tree.distances = nm.zeros[DType.float64](nm.Shape(2))
    tree.validate()
    var bad_parent = tree.copy()
    bad_parent.parents.value()[nm.Item(0)] = 4
    with assert_raises():
        bad_parent.validate()
    var bad_distances = tree.copy()
    bad_distances.distances = nm.zeros[DType.float64](nm.Shape(1))
    with assert_raises():
        bad_distances.validate()
    tree.children[nm.Item(1, 1)] = 0
    with assert_raises():
        tree.validate()
    var rows = List[List[Int]]()
    rows.append([0, 1])
    rows.append([])
    var graph = Neighborhoods(rows^, 2)
    graph.validate(2)
    var graph_copy = graph.copy()
    graph.indices[0][0] = 2
    graph_copy.validate(2)
    with assert_raises():
        graph.validate(2)
    var x = nm.zeros[DType.float64](nm.Shape(1, 1))
    with assert_raises():
        _ = DistanceProbe().pairwise(x, x)
    with assert_raises():
        _ = RadiusProbe().radius_neighbors(x, 1)
    var weights = nm.zeros[DType.float64](nm.Shape(1))
    with assert_raises():
        _ = CentroidProbe().assign(x, x, weights)
    with assert_raises():
        _ = CentroidProbe().update(x, labels, weights, x)
    var rng = RandomState(42)
    var same = rng.copy()
    assert_equal(rng.next_uint32(), same.next_uint32())
    print("cluster plumbing contracts passed")
