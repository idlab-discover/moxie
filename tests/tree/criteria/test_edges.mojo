import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.tree._criterion import Gini, MSE, MAE, Poisson
from moxie.tree._records import CriterionConfig, TrainingData

def data4() raises -> TrainingData:
    var y = nm.zeros[DType.float64](nm.Shape(4, 1))
    for i in range(4):
        y.unsafe_set(i, Float64(i))
    return TrainingData(nm.zeros[DType.float32](nm.Shape(4, 1)), y^, None,
        nm.zeros[DType.uint8](nm.Shape(1)), nm.full[DType.int64](nm.Shape(1), -1),
        nm.zeros[DType.int8](nm.Shape(1)))

def retained() raises -> MSE:
    var data = data4()
    var samples = nm.zeros[DType.int64](nm.Shape(4))
    for i in range(4):
        samples.unsafe_set(i, Int64(i))
    var criterion = MSE(CriterionConfig(1, [1], 4))
    criterion.initialize(data, samples, 0, 4, 4)
    return criterion^

def main() raises:
    with assert_raises():
        _ = MSE(CriterionConfig(0, [], 4))
    with assert_raises():
        _ = MSE(CriterionConfig(1, [2], 4))
    var data = data4()
    var indices = nm.zeros[DType.int64](nm.Shape(8))
    for i in range(4):
        indices.unsafe_set(2 * i, Int64(i))
    var samples = indices.view_with_layout(nm.Shape(4), nm.NDArrayStrides(2), 0)
    var c = MSE(CriterionConfig(1, [1], 4))
    c.initialize(data, samples, 0, 4, 4)
    assert_equal(c.node_impurity(), 1.25)
    c.update(2)
    assert_equal(c.children_impurity().left, 0.25)
    assert_equal(c.children_impurity().right, 0.25)
    # Node retained views keep storage alive after its original owner dies.
    var independent = retained()
    assert_equal(independent.node_impurity(), 1.25)
    c.initialize(data, samples, 1, 3, 4)
    assert_equal(c.node_impurity(), 0.25)
    assert_equal(independent.node_impurity(), 1.25)
    var mae = MAE(CriterionConfig(1, [1], 4))
    mae.initialize(data, samples, 0, 4, 4)
    assert_equal(mae.node_impurity(), 1.0)
    # Precomputed MAE must retain the checked child-query contract.
    with assert_raises():
        _ = mae.children_impurity()
    mae.update(4)
    with assert_raises():
        _ = mae.children_impurity()
    mae.reset()
    var dest = nm.zeros[DType.float64](nm.Shape(1, 1))
    mae.node_value(dest)
    assert_equal(dest.unsafe_get(0), 1.5)
    mae.initialize(data, samples, 1, 3, 4)
    # Known discrepancy: the pinned Cython oracle returns stale-root numerator / 2.
    assert_equal(mae.node_impurity(), 0.5)
    mae.node_value(dest)
    assert_equal(dest.unsafe_get(0), 1.5)
    # Missing-feature routing is represented by permutation changes before reset.
    data.X.unsafe_set(0, Float64("nan").cast[DType.float32]())
    data.missing_values_in_feature_mask.unsafe_set(0, 1)
    c.initialize(data, samples, 0, 4, 4)
    c.update(1)
    assert_equal(c.children_impurity().left, 0.0)
    indices.unsafe_set(0, 3)
    indices.unsafe_set(6, 0)
    c.reset()
    c.update(3)
    assert_equal(c.children_impurity().right, 0.0)
    var poisson = Poisson(CriterionConfig(1, [1], 4))
    poisson.initialize(data, samples, 0, 4, 4)
    poisson.update(3)
    assert_equal(poisson.proxy_impurity_improvement(), -Float64("inf"))
    assert_equal(poisson.children_impurity().right, Float64("inf"))
    data.y.unsafe_set(0, -1)
    with assert_raises():
        poisson.initialize(data, samples, 0, 4, 4)
    data.y.unsafe_set(0, Float64("nan"))
    with assert_raises():
        c.initialize(data, samples, 0, 4, 4)
    data.y.unsafe_set(0, 0)
    var gini = Gini(CriterionConfig(1, [2], 4))
    with assert_raises():
        gini.initialize(data, samples, 0, 4, 4)
    c.initialize(data, samples, 0, 0, 4)
    assert_true(c.node_impurity() != c.node_impurity())
    c.initialize(data, samples, 0, 4, 0)
    data.sample_weight = nm.full[DType.float64](nm.Shape(4), -1)
    with assert_raises():
        mae.initialize(data, samples, 0, 4, 4)
    c.initialize(data, samples, 0, 4, 4)
    assert_equal(c.node_impurity(), 1.25)
    # A guard slot detects accidental binary-complement writes for one class.
    var single = Gini(CriterionConfig(1, [1], 4))
    data.y = nm.zeros[DType.float64](nm.Shape(4, 1))
    data.sample_weight = None
    single.initialize(data, samples, 0, 4, 4)
    var guarded = nm.full[DType.float64](nm.Shape(2), 9876)
    var probability = guarded.view_with_layout(nm.Shape(1, 1), nm.NDArrayStrides(1, 1), 0)
    single.node_value(probability)
    single.clip_node_value(probability, 0.2, 0.8)
    assert_equal(probability.unsafe_get(0), 0.8)
    assert_equal(guarded.unsafe_get(1), 9876)
    print("PASS edges, missing permutations, strided indices and retained lifetime")
