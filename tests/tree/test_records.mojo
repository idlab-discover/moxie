from std.testing import assert_equal, assert_raises
from numojo import NDArray, Shape, Item, full
from moxie.tree._records import (
    CategoryBits, ParentInfo, SplitRecord, Node, Indices, Mask, Values, TreeData,
    TrainingData, validate_training_layout, validate_parent, validate_range, validate_split, value_offset, snapshot_values,
)
from moxie.tree._tree import Tree, new_tree

def assert_vector[dtype: DType](actual: NDArray[dtype], expected: List[Scalar[dtype]]) raises:
    assert_equal(actual.ndim, 1)
    assert_equal(actual.shape[0], len(expected))
    for i in range(len(expected)):
        assert_equal(actual.item(i), expected[i])

def test_inspection() raises:
    var tree = new_tree(2, [1], [-1, -1])
    assert_vector(tree.children_left(), [])
    assert_vector(tree.threshold(), [])
    assert_equal(tree.n_leaves(), 0)

    var storage = Tree()
    _ = storage.add_node(tree, -2, False, Node(-1, -1, 1, 0.25, CategoryBits(0), 0.5, 3, 4.5, 1, 0))
    _ = storage.add_node(tree, 0, True, Node(-1, -1, -1, 0, CategoryBits(0), 0, 2, 3, 0, -1))
    _ = storage.add_node(tree, 0, False, Node(-1, -1, -1, 0, CategoryBits(0), 0, 1, 1.5, 0, -1))
    storage.resize(tree, 8)
    assert_equal(tree.capacity, 8)

    var left: Indices = tree.children_left()
    var right: Indices = tree.children_right()
    var feature: Indices = tree.feature()
    var threshold: Values = tree.threshold()
    var impurity: Values = tree.impurity()
    var counts: Indices = tree.n_node_samples()
    var weights: Values = tree.weighted_n_node_samples()
    var missing: Mask = tree.missing_go_to_left()
    assert_vector(left, [1, -1, -1])
    assert_vector(right, [2, -1, -1])
    assert_vector(feature, [1, -2, -2])
    assert_vector(threshold, [0.25, -2.0, -2.0])
    assert_vector(impurity, [0.5, 0.0, 0.0])
    assert_vector(counts, [3, 2, 1])
    assert_vector(weights, [4.5, 3.0, 1.5])
    assert_vector(missing, [1, 0, 0])
    assert_equal(tree.n_leaves(), 2)

    left[Item(0)] = 8
    assert_equal(tree.nodes[0].left_child, 1)
    tree.nodes[0].threshold = 0.75
    assert_equal(threshold.item(0), 0.25)
    assert_equal(tree.threshold().item(0), 0.75)

def main() raises:
    test_inspection()
    validate_range(0, 2, 3, 3)
    validate_range(0, 0, 0, 0)
    with assert_raises():
        validate_range(-1, 0, 1, 1)
    with assert_raises():
        validate_range(0, 3, 2, 3)
    with assert_raises():
        validate_range(0, 0, 4, 3)
    validate_parent(ParentInfo(-1, 1, 0.5, 1), 2)
    with assert_raises():
        validate_parent(ParentInfo(1, -1, 0.5, 0), 2)
    with assert_raises():
        validate_parent(ParentInfo(0, 1, 0.5, 3), 2)
    assert_equal(value_offset(2, 1, 3, 4, 2, 5), 28)
    with assert_raises():
        _ = value_offset(4, 0, 0, 4, 2, 5)
    with assert_raises():
        _ = value_offset(0, 0, 0, 9223372036854775807, 2, 5)
    with assert_raises():
        _ = value_offset(0, 0, 0, 1, 0, 5)
    var split = SplitRecord(0, 2, 0.5, CategoryBits(0), 0, 0.2, 0.1, 0.1, -1, 1, 0)
    for kind in range(3):
        split.split_kind = Int8(kind)
        validate_split(split, 0, 4, 1)
    split.split_kind = 3
    with assert_raises():
        validate_split(split, 0, 4, 1)
    split.split_kind = -1
    with assert_raises():
        validate_split(split, 0, 4, 1)
    split.pos = 4
    validate_split(split, 0, 4, 1)
    split.missing_go_to_left = 2
    with assert_raises():
        validate_split(split, 0, 4, 1)
    var values = full[DType.float64](Shape(2, 1, 2), 3.0)
    var snapshot = snapshot_values(values)
    var view = values.view()
    view[Item(0, 0, 0)] = 8.0
    assert_equal(values.item(0, 0, 0), 8.0)
    assert_equal(snapshot.item(0, 0, 0), 3.0)
    var nodes = List[Node]()
    nodes.append(Node(-1, -1, -2, -2, CategoryBits(0), 0, 2, 2, 0, -1))
    var tree = TreeData(1, [2], [-1], 1, 2, 0, 1, 2, nodes^, snapshot^)
    assert_equal(tree.node_count, 1)
    assert_equal(tree.nodes[0].left_child, -1)
    assert_equal(tree.value.item(0, 0, 0), 3.0)
    var data = TrainingData(
        full[DType.float32](Shape(2, 1), 1.0),
        full[DType.float64](Shape(2, 1), 1.0),
        None,
        full[DType.uint8](Shape(1), 0),
        full[DType.int64](Shape(1), -1),
        full[DType.int8](Shape(1), 0),
    )
    validate_training_layout(data)
    data.sample_weight = full[DType.float64](Shape(2), 1.0)
    validate_training_layout(data)
    data.sample_weight = full[DType.float64](Shape(1), 1.0)
    with assert_raises():
        validate_training_layout(data)
    data.sample_weight = None
    data.n_categories = full[DType.int64](Shape(2), -1)
    with assert_raises():
        validate_training_layout(data)
    data.n_categories = full[DType.int64](Shape(1), -1)
    data.y = full[DType.float64](Shape(1, 1), 1.0)
    with assert_raises():
        validate_training_layout(data)
    print("Tree record validation and ownership passed")
