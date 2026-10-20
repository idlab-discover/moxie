from moxie.svm._cache import ColumnCache
import numojo as nm
from std.testing import assert_equal, assert_raises
from moxie.svm._contracts import (
    TrainingSnapshot, KernelSpec, pair_index, validate_matrix, validate_vector,
    KernelEngine, LinearEngine, KernelModel, LinearModel, KernelOptions, LinearOptions,
    Dense, KernelProvider, ClassPacking, WorkingSetState, KernelColumnCache,
)
from moxie._dense import row_squared_norms

# Raising probes establish signatures only. They are deliberately not solvers.
@fieldwise_init
struct KernelProbe(KernelEngine):
    def solve(self, x: Dense, y: Dense, sample_weight: Dense,
              class_weight: Dense, options: KernelOptions) raises -> KernelModel:
        raise Error("NotImplementedError: contract probe")

    def decision_values(self, x: Dense, model: KernelModel,
                        kernel: KernelSpec) raises -> Dense:
        raise Error("NotImplementedError: contract probe")

    def class_probabilities(self, decisions: Dense, model: KernelModel) raises -> Dense:
        raise Error("NotImplementedError: contract probe")

@fieldwise_init
struct LinearProbe(LinearEngine):
    def solve(self, x: Dense, y: Dense, sample_weight: Dense,
              class_weight: Dense, options: LinearOptions) raises -> LinearModel:
        raise Error("NotImplementedError: contract probe")

    def decision_values(self, x: Dense, model: LinearModel) raises -> Dense:
        raise Error("NotImplementedError: contract probe")

@fieldwise_init
struct CallableProbe(KernelProvider):
    def evaluate(self, x: Dense, train: Dense) raises -> Dense:
        raise Error("NotImplementedError: contract probe")

def first_column_value[T: KernelColumnCache](mut cache: T) raises -> Float32:
    return cache.column(0).unsafe_get(0)

def main() raises:
    var x = nm.ones[DType.float64](nm.Shape(3, 2))
    var snapshot = TrainingSnapshot(x)
    x.unsafe_set(0, 4)
    assert_equal(snapshot.x.unsafe_get(0), 1.0)
    var copied = snapshot.copy()
    copied.x.unsafe_set(0, 7)
    assert_equal(snapshot.x.unsafe_get(0), 1.0)
    snapshot.validate_query(x)
    var norms = row_squared_norms(snapshot.x)
    assert_equal(norms.unsafe_get(0), 2.0)
    var owner = nm.arange[DType.float64](12).reshape(nm.Shape(3, 4))
    var strided = owner.view_with_layout(nm.Shape(3, 2), nm.NDArrayStrides(4, -2), 3)
    var strided_snapshot = TrainingSnapshot(strided)
    owner.unsafe_set(3, 99)
    assert_equal(strided_snapshot.x.item(0, 0), 3.0)
    assert_equal(strided_snapshot.x.item(2, 1), 9.0)
    var shared = snapshot.x.view()
    shared.unsafe_set(0, 5)
    assert_equal(snapshot.x.unsafe_get(0), 5.0)
    snapshot = TrainingSnapshot(replacement_input())
    assert_equal(snapshot.n_features, 1)
    var index = 0
    for i in range(5):
        for j in range(i + 1, 5):
            assert_equal(pair_index(5, i, j), index)
            index += 1
    with assert_raises():
        _ = pair_index(3, 1, 1)
    with assert_raises():
        _ = pair_index(3, -1, 2)
    with assert_raises():
        _ = pair_index(3, 0, 3)
    with assert_raises():
        _ = TrainingSnapshot(x, True)
    var gram = nm.ones[DType.float64](nm.Shape(3, 3))
    var precomputed = TrainingSnapshot(gram, True)
    precomputed.validate_query(gram)
    with assert_raises():
        precomputed.validate_query(x)
    var empty = nm.zeros[DType.float64](nm.Shape(0, 2))
    with assert_raises():
        validate_matrix(empty)
    var y = nm.ones[DType.float64](nm.Shape(3))
    validate_vector(y, 3)
    with assert_raises():
        validate_matrix(y)
    with assert_raises():
        validate_vector(y, 2)
    with assert_raises():
        validate_vector(x, 3)
    x.unsafe_set(0, Float64("nan"))
    with assert_raises():
        validate_matrix(x)
    y.unsafe_set(0, Float64("inf"))
    with assert_raises():
        validate_vector(y, 3)
    var spec = KernelSpec("rbf", 0.5, 3, 0)
    spec.validate()
    spec.kind = "sparse"
    with assert_raises():
        spec.validate()
    spec.kind = "rbf"
    spec.gamma = -1
    with assert_raises():
        spec.validate()
    var opts = KernelOptions("c_svc", KernelSpec("linear", 1, 3, 0),
                             1, 0.5, 0.1, 0.001, 200, True, False, -1, 42)
    with assert_raises():
        _ = KernelProbe().solve(gram, y, y, y, opts)
    var linear = LinearOptions(1, 1, 0, 0.0001, True, 1, 1000, 42)
    with assert_raises():
        _ = LinearProbe().solve(gram, y, y, y, linear)
    with assert_raises():
        _ = CallableProbe().evaluate(gram, gram)
    var cache = ColumnCache(nm.ones[DType.float64](nm.Shape(3, 2)), KernelSpec("linear", 1, 3, 0), 1)
    assert_equal(first_column_value(cache), Float32(2))
    var counts = nm.ones[DType.int64](nm.Shape(3))
    var offsets = nm.arange[DType.int64](4)
    var packing = ClassPacking(counts^, offsets^)
    packing.validate(3)
    with assert_raises():
        packing.validate(-1)
    with assert_raises():
        packing.validate(4)
    packing.counts.unsafe_set(0, -1)
    with assert_raises():
        packing.validate(1)
    var state = WorkingSetState(
        nm.zeros[DType.float64](nm.Shape(2)), nm.zeros[DType.float64](nm.Shape(2)),
        nm.zeros[DType.float64](nm.Shape(2)), nm.zeros[DType.float64](nm.Shape(2)),
        nm.zeros[DType.float64](nm.Shape(2)), nm.ones[DType.float64](nm.Shape(2)),
        nm.ones[DType.int64](nm.Shape(2)), nm.zeros[DType.int64](nm.Shape(2)),
        nm.arange[DType.int64](2), 2, False)
    assert_equal(state.active_size, 2)
    print("PASS dense SVM plumbing")

def replacement_input() raises -> Dense:
    return nm.ones[DType.float64](nm.Shape(1, 1))
