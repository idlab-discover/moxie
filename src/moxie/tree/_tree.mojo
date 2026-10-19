# Adapted from scikit-learn tree/_tree.pyx and tree/_utils.pxd.
# Copyright the scikit-learn developers; BSD-3-Clause, LICENSES/scikit-learn.txt.
"""Allocation and dense inspection over the family's single TreeData record."""
from numojo import Shape, zeros
from moxie.tree._records import (
    TreeData, Node, CategoryBits, Values, Features, Indices, value_offset,
    TREE_LEAF, TREE_UNDEFINED, SPLIT_LEAF, SPLIT_NUMERIC,
    SPLIT_CATEGORICAL_BITSET, SPLIT_CATEGORICAL_HASH,
)
from moxie.tree._engine_api import TreeStorage


def _dimensions(n_features: Int, n_classes: List[Int], n_categories: List[Int]) raises -> Int:
    if n_features <= 0 or len(n_categories) != n_features or len(n_classes) == 0:
        raise Error("ValueError: invalid tree dimensions")
    var largest = 0
    for n in n_classes:
        if n <= 0:
            raise Error("ValueError: n_classes must be positive")
        largest = max(largest, n)
    for n in n_categories:
        if n < -1 or n > 16777216:
            raise Error("ValueError: invalid category count")
    return largest


def new_tree(n_features: Int, n_classes: List[Int], n_categories: List[Int]) raises -> TreeData:
    var largest = _dimensions(n_features, n_classes, n_categories)
    _ = value_offset(0, 0, 0, 1, len(n_classes), largest)
    return TreeData(n_features, n_classes.copy(), n_categories.copy(), len(n_classes),
                    largest, 0, 0, 0, List[Node](),
                    zeros[DType.float64](Shape(0, len(n_classes), largest)))


def _layout(tree: TreeData) raises:
    var largest = _dimensions(tree.n_features, tree.n_classes, tree.n_categories)
    if tree.n_outputs != len(tree.n_classes) or tree.max_n_classes != largest:
        raise Error("ValueError: inconsistent output metadata")
    if tree.node_count != len(tree.nodes) or tree.capacity < tree.node_count or tree.capacity < 0:
        raise Error("ValueError: inconsistent node capacity")
    _ = value_offset(0, 0, 0, max(1, tree.capacity), tree.n_outputs, largest)
    if tree.value.ndim != 3:
        raise Error("ValueError: values must have rank three")
    if tree.value.shape[0] != tree.capacity or tree.value.shape[1] != tree.n_outputs or tree.value.shape[2] != largest:
        raise Error("ValueError: invalid value shape")
    if tree.value.strides[2] != 1 or tree.value.strides[1] != largest or tree.value.strides[0] != tree.n_outputs * largest:
        raise Error("ValueError: values must be C contiguous")


def _node(tree: TreeData, node: Node) raises:
    if node.missing_go_to_left > 1 or node.n_node_samples < 0:
        raise Error("ValueError: invalid node metadata")
    if node.split_kind == SPLIT_LEAF:
        return
    if node.feature < 0 or node.feature >= tree.n_features:
        raise Error("ValueError: invalid split feature")
    if node.split_kind == SPLIT_NUMERIC:
        if node.threshold != node.threshold or tree.n_categories[node.feature] != -1:
            raise Error("ValueError: invalid numeric split")
    elif node.split_kind == SPLIT_CATEGORICAL_BITSET:
        if tree.n_categories[node.feature] < 0 or tree.n_categories[node.feature] > 256:
            raise Error("ValueError: invalid bitset category count")
    elif node.split_kind == SPLIT_CATEGORICAL_HASH:
        if tree.n_categories[node.feature] < 0:
            raise Error("ValueError: invalid hash category count")
        for i in range(1, 8):
            if node.left_cat_bitset[i] != 0:
                raise Error("ValueError: hash seed uses only word zero")
    else:
        raise Error("ValueError: unknown split kind")


def validate_tree(tree: TreeData) raises:
    """Validate complete creation-ordered topology before dereferencing children."""
    _layout(tree)
    if tree.node_count == 0:
        raise Error("NotFittedError: tree has no nodes")
    var parents = zeros[DType.int64](Shape(tree.node_count))
    for i in range(tree.node_count):
        _node(tree, tree.nodes[i])
        if tree.nodes[i].split_kind == SPLIT_LEAF:
            if tree.nodes[i].left_child != TREE_LEAF or tree.nodes[i].right_child != TREE_LEAF:
                raise Error("ValueError: leaf has children")
        else:
            var left = tree.nodes[i].left_child
            var right = tree.nodes[i].right_child
            if left <= i or right <= i or left >= tree.node_count or right >= tree.node_count or left == right:
                raise Error("ValueError: children must follow their parent")
            parents.unsafe_set(left, parents.unsafe_get(left) + 1)
            parents.unsafe_set(right, parents.unsafe_get(right) + 1)
    for i in range(1, tree.node_count):
        if parents.unsafe_get(i) != 1:
            raise Error("ValueError: every non-root node needs one parent")


def copy_tree(tree: TreeData) raises -> TreeData:
    _layout(tree)
    return TreeData(tree.n_features, tree.n_classes.copy(), tree.n_categories.copy(),
        tree.n_outputs, tree.max_n_classes, tree.max_depth, tree.node_count, tree.capacity,
        tree.nodes.copy(), tree.value.copy())


def _go_left(node: Node, value: Float32, categories: Int) raises -> Bool:
    if value != value:
        return node.missing_go_to_left != 0
    if node.split_kind == SPLIT_NUMERIC:
        return Float64(value) <= node.threshold
    # Check before float-to-integer conversion: out-of-domain codes are not
    # silently wrapped. Estimator encoders map unknown categories to NaN.
    if value < 0 or value >= Float32(categories):
        raise Error("ValueError: categorical code out of range; encode unknown as NaN")
    var code = Int(value)
    if Float32(code) != value:
        raise Error("ValueError: categorical codes must be integers")
    if node.split_kind == SPLIT_CATEGORICAL_BITSET:
        return (node.left_cat_bitset[code // 32] & (UInt32(1) << UInt32(code % 32))) != 0
    var mixed = node.left_cat_bitset[0] ^ UInt32(code)
    mixed ^= mixed >> 16
    mixed *= UInt32(0x7feb352d)
    mixed ^= mixed >> 15
    mixed *= UInt32(0x846ca68b)
    mixed ^= mixed >> 16
    return (mixed & 1) != 0


struct Tree(TreeStorage):
    """Stateless engine; mutable state lives exclusively in shared TreeData."""
    def __init__(out self):
        pass

    def resize(mut self, mut tree: TreeData, capacity: Int) raises:
        _layout(tree)
        var requested = capacity
        if requested == -1:
            if tree.capacity > 9223372036854775807 // 2:
                raise Error("ValueError: capacity overflow")
            requested = 3 if tree.capacity == 0 else 2 * tree.capacity
        if requested < 0:
            raise Error("ValueError: negative capacity")
        _ = value_offset(0, 0, 0, max(1, requested), tree.n_outputs, tree.max_n_classes)
        var stride = tree.n_outputs * tree.max_n_classes
        if requested > 9223372036854775807 // 8 // stride:
            raise Error("ValueError: value byte capacity overflow")
        if requested == tree.capacity:
            return
        var values = zeros[DType.float64](Shape(requested, tree.n_outputs, tree.max_n_classes))
        for i in range(min(requested, tree.capacity) * stride):
            values.unsafe_set(i, tree.value.unsafe_get(i))
        while len(tree.nodes) > requested:
            _ = tree.nodes.pop()
        tree.value = values^
        tree.capacity = requested
        tree.node_count = len(tree.nodes)
        if tree.node_count == 0:
            tree.max_depth = 0
        # A destructive shrink can leave dangling child IDs, like sklearn.
        # Traversal rejects that incomplete topology until the caller repairs it.

    def add_node(mut self, mut tree: TreeData, parent: Int, is_left: Bool, node: Node) raises -> Int:
        _layout(tree)
        _node(tree, node)
        if tree.node_count == 0:
            if parent != TREE_UNDEFINED:
                raise Error("ValueError: root parent must be TREE_UNDEFINED")
        else:
            if parent < 0 or parent >= tree.node_count:
                raise Error("ValueError: invalid parent")
            if tree.nodes[parent].split_kind == SPLIT_LEAF:
                raise Error("ValueError: cannot attach to leaf")
            var previous = tree.nodes[parent].left_child if is_left else tree.nodes[parent].right_child
            if previous != TREE_LEAF:
                raise Error("ValueError: child already attached")
        var added = node.copy()
        added.left_child = TREE_LEAF
        added.right_child = TREE_LEAF
        if added.split_kind == SPLIT_LEAF:
            added.feature = TREE_UNDEFINED
            added.threshold = TREE_UNDEFINED
            added.missing_go_to_left = 0
            added.left_cat_bitset = CategoryBits(0)
        elif added.split_kind != SPLIT_NUMERIC:
            added.threshold = -Float64(1.0) / Float64(0.0)
        var node_id = tree.node_count
        if node_id == tree.capacity:
            self.resize(tree, -1)
        tree.nodes.append(added^)
        tree.node_count += 1
        if parent != TREE_UNDEFINED:
            if is_left:
                tree.nodes[parent].left_child = node_id
            else:
                tree.nodes[parent].right_child = node_id
        return node_id

    def apply(self, tree: TreeData, X: Features) raises -> Indices:
        validate_tree(tree)
        if X.ndim != 2:
            raise Error("ValueError: X must have rank two")
        if X.shape[1] != tree.n_features:
            raise Error("ValueError: X feature count mismatch")
        var result = zeros[DType.int64](Shape(X.shape[0]))
        var row_stride = Int(X.strides[0])
        var col_stride = Int(X.strides[1])
        for row in range(X.shape[0]):
            var node_id = 0
            while tree.nodes[node_id].split_kind != SPLIT_LEAF:
                var feature = tree.nodes[node_id].feature
                var value = X.unsafe_get(row * row_stride + feature * col_stride)
                var left = _go_left(tree.nodes[node_id], value, tree.n_categories[feature])
                node_id = tree.nodes[node_id].left_child if left else tree.nodes[node_id].right_child
            result.unsafe_set(row, Int64(node_id))
        return result^

    def predict(self, tree: TreeData, X: Features) raises -> Values:
        var leaves = self.apply(tree, X)
        var stride = tree.n_outputs * tree.max_n_classes
        var result = zeros[DType.float64](Shape(X.shape[0], tree.n_outputs, tree.max_n_classes))
        for row in range(X.shape[0]):
            var leaf = Int(leaves.unsafe_get(row))
            for j in range(stride):
                result.unsafe_set(row * stride + j, tree.value.unsafe_get(leaf * stride + j))
        return result^

    def compute_node_depths(self, tree: TreeData) raises -> Indices:
        validate_tree(tree)
        var depths = zeros[DType.int64](Shape(tree.node_count))
        depths.unsafe_set(0, 1)
        for i in range(tree.node_count):
            if tree.nodes[i].split_kind != SPLIT_LEAF:
                var depth = depths.unsafe_get(i) + 1
                depths.unsafe_set(tree.nodes[i].left_child, depth)
                depths.unsafe_set(tree.nodes[i].right_child, depth)
        return depths^

    def compute_feature_importances(self, tree: TreeData, normalize: Bool = True) raises -> Values:
        validate_tree(tree)
        var result = zeros[DType.float64](Shape(tree.n_features))
        for i in range(tree.node_count):
            if tree.nodes[i].split_kind != SPLIT_LEAF:
                var left = tree.nodes[i].left_child
                var right = tree.nodes[i].right_child
                var decrease = tree.nodes[i].weighted_n_node_samples * tree.nodes[i].impurity
                decrease -= tree.nodes[left].weighted_n_node_samples * tree.nodes[left].impurity
                decrease -= tree.nodes[right].weighted_n_node_samples * tree.nodes[right].impurity
                var feature = tree.nodes[i].feature
                result.unsafe_set(feature, result.unsafe_get(feature) + decrease)
        var total = Float64(0)
        for f in range(tree.n_features):
            var importance = result.unsafe_get(f) / tree.nodes[0].weighted_n_node_samples
            result.unsafe_set(f, importance)
            total += importance
        if normalize and total > 0:
            for f in range(tree.n_features):
                result.unsafe_set(f, result.unsafe_get(f) / total)
        return result^
