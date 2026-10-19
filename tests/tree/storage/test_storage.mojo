from std.testing import assert_equal, assert_true, assert_raises
from numojo import Shape, NDArrayStrides, full, zeros
from moxie.tree._records import Node, CategoryBits, TreeData, Features
from moxie.tree._engine_api import TreeStorage
from moxie.tree._tree import Tree, new_tree, copy_tree, validate_tree

def leaf(value: Float64 = 0) -> Node:
    return Node(-1, -1, -2, -2, CategoryBits(0), value, 1, 1, 0, -1)

def branch(feature: Int = 0) -> Node:
    return Node(-1, -1, feature, 0.5, CategoryBits(0), 1, 3, 3, 0, 0)

def dispatch[T: TreeStorage](engine: T, tree: TreeData, X: Features) raises:
    var ids = engine.apply(tree, X)
    assert_equal(ids.item(0), 1)

def main() raises:
    var engine = Tree()
    var tree = new_tree(2, [1, 3], [-1, -1])
    var X = zeros[DType.float32](Shape(3, 2))
    with assert_raises():
        _ = engine.apply(tree, X)
    with assert_raises():
        _ = engine.compute_node_depths(tree)
    with assert_raises():
        _ = new_tree(0, [1], [])
    with assert_raises():
        _ = new_tree(1, [0], [-1])
    with assert_raises():
        _ = new_tree(1, [1], [16777217])
    with assert_raises():
        _ = engine.add_node(tree, 0, False, leaf())
    assert_equal(engine.add_node(tree, -2, False, branch()), 0)
    assert_equal(tree.capacity, 3)
    with assert_raises():
        _ = engine.apply(tree, X)
    assert_equal(engine.add_node(tree, 0, True, leaf()), 1)
    assert_equal(engine.add_node(tree, 0, False, branch(1)), 2)
    with assert_raises():
        _ = engine.add_node(tree, 0, True, leaf())
    with assert_raises():
        _ = engine.add_node(tree, 1, True, leaf())
    assert_equal(engine.add_node(tree, 2, False, leaf()), 3)
    assert_equal(tree.capacity, 6)
    assert_equal(engine.add_node(tree, 2, True, leaf()), 4)
    tree.max_depth = 2
    validate_tree(tree)
    for i in range(tree.node_count):
        tree.value.unsafe_set(i * 6, Float64(i))
        for c in range(3):
            tree.value.unsafe_set(i * 6 + 3 + c, Float64(i * 10 + c))
    X.unsafe_set(2, 1)
    X.unsafe_set(4, 1)
    X.unsafe_set(5, 1)
    dispatch(engine, tree, X)
    var ids = engine.apply(tree, X)
    assert_equal(ids.item(0), 1)
    assert_equal(ids.item(1), 4)
    assert_equal(ids.item(2), 3)
    # Borrowed layouts preserve offsets and signed element strides.
    var backing = zeros[DType.float32](Shape(20))
    var strided = backing.view_with_layout(Shape(3, 2), NDArrayStrides(6, 2), 1)
    for row in range(3):
        for col in range(2):
            strided.unsafe_set(row * 6 + col * 2, X.item(row, col))
    var strided_ids = engine.apply(tree, strided)
    var reversed = backing.view_with_layout(Shape(3, 2), NDArrayStrides(-6, 2), 13)
    var reversed_ids = engine.apply(tree, reversed)
    for row in range(3):
        assert_equal(strided_ids.item(row), ids.item(row))
        assert_equal(reversed_ids.item(row), ids.item(2 - row))
    var column_major = backing.view_with_layout(Shape(3, 2), NDArrayStrides(1, 3), 0)
    for row in range(3):
        for col in range(2):
            column_major.unsafe_set(row + col * 3, X.item(row, col))
    var column_ids = engine.apply(tree, column_major)
    for row in range(3):
        assert_equal(column_ids.item(row), ids.item(row))
    var pred = engine.predict(tree, X)
    assert_equal(pred.shape[0], 3)
    assert_equal(pred.shape[1], 2)
    assert_equal(pred.shape[2], 3)
    assert_equal(pred.item(1, 0, 0), 4)
    assert_equal(pred.item(1, 0, 1), 0)
    assert_equal(pred.item(2, 1, 2), 32)
    pred.unsafe_set(0, 999)
    assert_equal(tree.value.unsafe_get(6), 1)
    var depths = engine.compute_node_depths(tree)
    assert_equal(depths.item(0), 1)
    assert_equal(depths.item(2), 2)
    assert_equal(depths.item(3), 3)
    assert_equal(depths.item(4), 3)
    var importance = engine.compute_feature_importances(tree)
    assert_equal(importance.item(0), 0)
    assert_equal(importance.item(1), 1)
    var raw = engine.compute_feature_importances(tree, False)
    assert_equal(raw.item(1), 1)
    var snapshot = copy_tree(tree)
    tree.value.unsafe_set(6, -10)
    tree.nodes[0].threshold = -1
    tree.n_classes[0] = 2
    assert_equal(snapshot.value.unsafe_get(6), 1)
    assert_equal(snapshot.nodes[0].threshold, 0.5)
    assert_equal(snapshot.n_classes[0], 1)
    tree = snapshot^
    engine.resize(tree, 20)
    assert_equal(tree.capacity, 20)
    assert_equal(tree.value.unsafe_get(6), 1)
    assert_equal(tree.value.unsafe_get(19 * 6), 0)
    engine.resize(tree, 5)
    assert_equal(tree.node_count, 5)
    validate_tree(tree)
    with assert_raises():
        engine.resize(tree, -3)
    with assert_raises():
        engine.resize(tree, 9223372036854775807)
    assert_equal(tree.capacity, 5)
    var wrong = zeros[DType.float32](Shape(2, 3))
    with assert_raises():
        _ = engine.apply(tree, wrong)
    wrong = zeros[DType.float32](Shape(2))
    with assert_raises():
        _ = engine.apply(tree, wrong)
    var empty = zeros[DType.float32](Shape(0, 2))
    assert_equal(engine.apply(tree, empty).size, 0)
    assert_equal(engine.predict(tree, empty).size, 0)
    tree.nodes[2].left_child = 2
    with assert_raises():
        _ = engine.apply(tree, X)
    tree.nodes[2].left_child = 4
    tree.nodes[2].right_child = 4
    with assert_raises():
        validate_tree(tree)
    tree.nodes[2].right_child = 3
    engine.resize(tree, 2)
    with assert_raises():
        _ = engine.apply(tree, X)
    engine.resize(tree, 0)
    assert_equal(tree.node_count, 0)
    assert_equal(tree.max_depth, 0)
    _ = engine.add_node(tree, -2, False, leaf())
    assert_equal(engine.compute_feature_importances(tree).item(0), 0)
    assert_equal(engine.compute_node_depths(tree).item(0), 1)
    assert_equal(engine.apply(tree, X).item(2), 0)
    var cat = new_tree(1, [1], [256])
    var node = branch()
    node.split_kind = 1
    node.left_cat_bitset[7] = UInt32(1) << 31
    node.missing_go_to_left = 1
    _ = engine.add_node(cat, -2, False, node)
    _ = engine.add_node(cat, 0, True, leaf())
    _ = engine.add_node(cat, 0, False, leaf())
    var C = full[DType.float32](Shape(1, 1), 255)
    assert_equal(engine.apply(cat, C).item(0), 1)
    C.unsafe_set(0, 254)
    assert_equal(engine.apply(cat, C).item(0), 2)
    C.unsafe_set(0, Float32(0) / Float32(0))
    assert_equal(engine.apply(cat, C).item(0), 1)
    for invalid in [-1.0, 0.5, 256.0]:
        C.unsafe_set(0, Float32(invalid))
        with assert_raises():
            _ = engine.apply(cat, C)
    C.unsafe_set(0, Float32(1) / Float32(0))
    with assert_raises():
        _ = engine.apply(cat, C)
    cat.nodes[0].split_kind = 3
    with assert_raises():
        _ = engine.apply(cat, C)
    print("PASS native tree storage, ownership, growth, shapes, topology and errors")
