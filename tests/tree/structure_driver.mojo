from numojo import Shape, empty, full
from moxie.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier
from moxie.tree._records import TreeData


def print_structure(tree: TreeData, case_id: Int) raises:
    var left = tree.children_left()
    var right = tree.children_right()
    var feature = tree.feature()
    var threshold = tree.threshold()
    var impurity = tree.impurity()
    var counts = tree.n_node_samples()
    var weights = tree.weighted_n_node_samples()
    var missing = tree.missing_go_to_left()
    print("CASE", case_id, tree.node_count, tree.max_depth, tree.n_leaves())
    for i in range(tree.node_count):
        print(
            "NODE", case_id, left.item(i), right.item(i), feature.item(i),
            threshold.item(i), impurity.item(i), counts.item(i), weights.item(i),
            missing.item(i),
        )


def main() raises:
    var X = empty[DType.float32](Shape(4, 1))
    var labels = empty[DType.int64](Shape(4))
    var targets = empty[DType.float64](Shape(4))
    for i in range(4):
        labels.unsafe_set(i, Int64(i // 2))
        targets.unsafe_set(i, Float64(i // 2))
        X.unsafe_set(i, Float32(i))

    var best = DecisionTreeClassifier[DType.int64](random_state=0)
    X.unsafe_set(0, Float32(Float64("nan")))
    _ = best.fit(X, labels)
    print_structure(best.tree_, 0)

    var regressor = DecisionTreeRegressor(random_state=0)
    X.unsafe_set(0, 0)
    X.unsafe_set(3, Float32(Float64("nan")))
    var weights = full[DType.float64](Shape(4), 1.5)
    _ = regressor.fit(X, targets, weights^)
    print_structure(regressor.tree_, 1)

    X.unsafe_set(1, 0)
    X.unsafe_set(3, 2)
    var extra = ExtraTreeClassifier[DType.int64](random_state=0)
    _ = extra.fit(X, labels)
    print_structure(extra.tree_, 2)

    var ordered = empty[DType.float32](Shape(8, 1))
    var classes = empty[DType.int64](Shape(8))
    for i in range(8):
        ordered.unsafe_set(i, Float32(i))
        classes.unsafe_set(i, Int64(0 if i < 4 else i // 2 - 1))

    var limited = DecisionTreeClassifier[DType.int64](random_state=0, max_leaf_nodes=3)
    _ = limited.fit(ordered, classes)
    print_structure(limited.tree_, 3)

    var pruned = DecisionTreeClassifier[DType.int64](random_state=0, ccp_alpha=0.3)
    _ = pruned.fit(ordered, classes)
    print_structure(pruned.tree_, 4)

    var constant = full[DType.float32](Shape(4, 1), 0)
    _ = best.fit(constant, labels)
    print_structure(best.tree_, 5)
