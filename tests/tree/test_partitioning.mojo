import numojo as nm
from std.testing import assert_equal, assert_true
from moxie.tree._records import TrainingData, SplitWorkspace, SplitRecord, CategoryBits, SplitPosition
from moxie.tree._partitioner import DensePartitioner
from moxie.tree._engine_api import Partitioner, TreeBuilder
from moxie.tree._builders import DepthFirstTreeBuilder
from moxie.tree import DecisionTreeClassifier

def main() raises:
    comptime assert conforms_to(DensePartitioner, Partitioner)
    comptime assert conforms_to(DepthFirstTreeBuilder[0], TreeBuilder)
    var X = nm.zeros[DType.float32](nm.Shape(4, 2))
    var y = nm.zeros[DType.float64](nm.Shape(4))
    for i in range(4):
        X.unsafe_set(i * 2, Float32(i))
        X.unsafe_set(i * 2 + 1, Float32(i))
        y.unsafe_set(i, Float64(i // 2))
    var c = DecisionTreeClassifier(random_state=0)
    _ = c.fit(X, y)
    assert_equal(c.tree_.nodes[0].feature, 1)
    assert_equal(c.tree_.nodes[0].threshold, 1.5)
    var work = SplitWorkspace(nm.zeros[DType.int64](nm.Shape(4)), nm.zeros[DType.int64](nm.Shape(2)), nm.zeros[DType.int64](nm.Shape(2)), nm.zeros[DType.float32](nm.Shape(4)), 0, 4, 0)
    for i in range(4):
        work.samples.unsafe_set(i, Int64(3 - i))
    X.unsafe_set(6, Float32(Float64("nan")))
    # The mask declares feature 0's NaN; like the reference, sorting compacts only flagged features.
    var missing = nm.zeros[DType.uint8](nm.Shape(2))
    missing.unsafe_set(0, 1)
    var data = TrainingData(X^, y.reshape(nm.Shape(4, 1)), None, missing^, nm.full[DType.int64](nm.Shape(2), -1), nm.zeros[DType.int8](nm.Shape(2)))
    var partitioner = DensePartitioner()
    partitioner.init_node_split(work, 0, 4)
    assert_true(not partitioner.sort_samples_and_feature_values(data, work, 0))
    assert_equal(work.n_missing, 1)
    assert_equal(work.samples.unsafe_get(3), 3)
    var position = partitioner.next_p(work, SplitPosition(0, 0), False)
    assert_equal(position.current, 1)
    partitioner.shift_missing_to_the_left(work)
    assert_equal(work.samples.unsafe_get(0), 3)
    var split = SplitRecord(0, 3, 1.5, CategoryBits(0), 0, 0, 0, 0, 0, 1, 1)
    partitioner.partition_samples_final(data, work, split)
    for i in range(3):
        assert_true(work.samples.unsafe_get(i) != 2)
    assert_equal(work.samples.unsafe_get(3), 2)
    data.n_categories.unsafe_set(0, 3)
    data.y.unsafe_set(0, 2)
    data.y.unsafe_set(1, 0)
    data.y.unsafe_set(2, 1)
    for i in range(4):
        work.samples.unsafe_set(i, Int64(i))
    partitioner.init_node_split(work, 0, 4)
    assert_true(not partitioner.sort_samples_and_feature_values(data, work, 0))
    assert_equal(work.feature_values.unsafe_get(0), 1)
    assert_equal(work.feature_values.unsafe_get(1), 2)
    assert_equal(work.feature_values.unsafe_get(2), 0)
    position = partitioner.next_p(work, SplitPosition(0, 0), False)
    position = partitioner.next_p(work, position, False)
    assert_equal(position.current, 2)
    split = SplitRecord(0, 2, 0, CategoryBits(0), 1, 0, 0, 0, 0, 1, 0)
    partitioner.cat_position_to_split_bitset(work, 2, False, split)
    assert_equal(split.left_cat_bitset[0], UInt32(6))
    partitioner.partition_samples_final(data, work, split)
    print("PASS numeric/categorical partitioner protocols, missing partitions and feature ties")
