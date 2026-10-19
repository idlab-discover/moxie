from std.testing import assert_equal, assert_true, assert_raises
import numojo as nm
from moxie.tree._criterion import WeightedCriterion
from moxie.tree._engine_api import Criterion
from moxie.tree._records import CriterionConfig, TrainingData, Values

def check_dispatch[C: Criterion](c: C) raises -> Float64:
    return c.node_impurity()

def check[kind: Int]() raises:
    var nc = 2 if kind < 2 else 1
    var c = WeightedCriterion[kind](CriterionConfig(1, [nc], 4))
    with assert_raises():
        _ = c.node_impurity()
    var y = nm.zeros[DType.float64](nm.Shape(4, 1))
    var weights = nm.zeros[DType.float64](nm.Shape(4))
    var samples = nm.zeros[DType.int64](nm.Shape(4))
    for i in range(4):
        y.unsafe_set(i, Float64(i // 2 if kind < 2 else i))
        weights.unsafe_set(i, Float64(i + 1))
        samples.unsafe_set(i, Int64(i))
    var data = TrainingData(nm.zeros[DType.float32](nm.Shape(4, 1)), y^, weights^,
        nm.zeros[DType.uint8](nm.Shape(1)), nm.full[DType.int64](nm.Shape(1), -1),
        nm.zeros[DType.int8](nm.Shape(1)))
    c.initialize(data, samples, 0, 4, 10)
    var value = nm.zeros[DType.float64](nm.Shape(1, nc))
    c.node_value(value)
    assert_equal(c.weighted_n_node_samples, 10.0)
    assert_equal(check_dispatch(c), c.node_impurity())
    comptime if kind < 2:
        assert_equal(value.unsafe_get(0), 0.3)
        assert_equal(value.unsafe_get(1), 0.7)
    else:
        assert_equal(value.unsafe_get(0), 2.0)
    c.update(2)
    assert_equal(c.weighted_n_left, 3.0)
    assert_equal(c.weighted_n_right, 7.0)
    var children = c.children_impurity()
    assert_true(c.impurity_improvement(c.node_impurity(),children.left,children.right) >= 0)
    _ = c.proxy_impurity_improvement()
    _ = c.middle_value()
    assert_true(c.check_monotonicity(Int8(-1 if kind < 2 else 1), -100, 100))
    assert_true(not c.check_monotonicity(Int8(1 if kind < 2 else -1), -100, 100))
    c.clip_node_value(value, 0.4, 0.6)
    assert_true(value.unsafe_get(0) >= 0.4 and value.unsafe_get(0) <= 0.6)
    with assert_raises():
        c.update(1)
    with assert_raises():
        c.update(5)
    with assert_raises():
        c.clip_node_value(value, 1, 0)
    c.reset()
    c.update(3)
    assert_equal(c.weighted_n_left, 6.0)
    comptime if kind == 3:
        with assert_raises():
            c.reverse_reset()
    else:
        c.reverse_reset()
        assert_equal(c.weighted_n_right, 0.0)
    # Reordering the shared permutation and resetting must be observed.
    samples.unsafe_set(0, 3)
    samples.unsafe_set(3, 0)
    c.reset()
    c.update(1)
    assert_equal(c.weighted_n_left, 4.0)
    # Repeat initialization replaces retained state.
    c.initialize(data, samples, 1, 3, 10)
    assert_equal(c.weighted_n_node_samples, 5.0)
    samples.unsafe_set(1, 99)
    with assert_raises():
        c.initialize(data, samples, 1, 3, 10)
    with assert_raises():
        _ = c.node_impurity()
    print("PASS criterion", kind)

def main() raises:
    comptime for kind in range(5):
        check[kind]()
