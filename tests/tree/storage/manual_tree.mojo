from numojo import Shape, full
from moxie.tree._tree import Tree, new_tree
from moxie.tree._records import Node, CategoryBits

def main() raises:
    var engine = Tree()
    var tree = new_tree(1, [1], [-1])
    _ = engine.add_node(tree, -2, False, Node(-1, -1, 0, 0.5, CategoryBits(0), 1, 2, 2, 0, 0))
    _ = engine.add_node(tree, 0, True, Node(-1, -1, -2, -2, CategoryBits(0), 0, 1, 1, 0, -1))
    _ = engine.add_node(tree, 0, False, Node(-1, -1, -2, -2, CategoryBits(0), 0, 1, 1, 0, -1))
    tree.value.unsafe_set(1, 10)
    tree.value.unsafe_set(2, 20)
    tree.max_depth = 1  # Builders publish the maximum edge depth.
    var X = full[DType.float32](Shape(2, 1), 0)
    X.unsafe_set(1, 1)
    var predictions = engine.predict(tree, X)
    print(predictions.item(0, 0, 0), predictions.item(1, 0, 0))
