import numojo as nm
from moxie.tree._classes import BaseDecisionTree

def run[classifier: Bool, extra: Bool](criterion: String) raises:
    var nan = Float32(Float64("nan"))
    for case_id in range(8):
        var sign = -1 if case_id % 2 == 1 else 1
        var best_first = case_id % 4 >= 2
        var missing_low = case_id < 4
        var X = nm.zeros[DType.float32](nm.Shape(48, 1))
        var y = nm.zeros[DType.float64](nm.Shape(48))
        var weights = nm.zeros[DType.float64](nm.Shape(48))
        for i in range(48):
            var group = i // 8
            X.unsafe_set(i, Float32(group) if group < 5 else nan)
            var target = group + 1 if group < 5 else (1 if missing_low else 5)
            comptime if classifier:
                var positives = 0 if target == 1 else (8 if target == 5 else target)
                var positive = i % 8 < positives
                y.unsafe_set(i, Float64(13 if positive == (sign > 0) else 7))
            else:
                y.unsafe_set(i, Float64(target if sign > 0 else 6 - target))
            weights.unsafe_set(i, Float64(1 + (i * 3 + group) % 7) / 4)
        var seed = case_id
        comptime if classifier and extra:
            seed = 2 if missing_low else 0
        var estimator = BaseDecisionTree[classifier, extra](
            criterion=criterion, max_depth=4, random_state=seed,
            monotonic_cst=Optional[List[Int]]([sign]),
        )
        if best_first:
            estimator.params.max_leaf_nodes = 4
        _ = estimator.fit(X, y, weights^)
        ref tree = estimator.tree_
        var thresholds = tree.threshold()
        var missing = tree.missing_go_to_left()
        print("CASE", criterion, extra, case_id, tree.node_count, tree.max_depth, tree.n_leaves())
        for i in range(tree.node_count):
            ref node = tree.nodes[i]
            print("NODE", criterion, extra, case_id, i, node.left_child,
                node.right_child, node.feature, thresholds.item(i), node.impurity,
                node.n_node_samples, node.weighted_n_node_samples, missing.item(i))
            for j in range(tree.max_n_classes):
                print("VALUE", criterion, extra, case_id, i, j,
                    tree.value.unsafe_get(i * tree.max_n_classes + j))
        var query = nm.zeros[DType.float32](nm.Shape(12, 1))
        for i in range(11):
            query.unsafe_set(i, Float32(i - 1) / 2)
        query.unsafe_set(11, nan)
        var labels = estimator.predict(query)
        var leaves = estimator.apply(query)
        for i in range(query.size):
            print("PRED", criterion, extra, case_id, i, leaves.item(i), labels.item(i))
        comptime if classifier:
            var probabilities = estimator.predict_proba(query)
            for i in range(query.size):
                print("PROBA", criterion, extra, case_id, i, probabilities[0].unsafe_get(2 * i + 1))

def main() raises:
    run[True, False]("gini")
    run[True, True]("entropy")
    run[False, False]("squared_error")
    run[False, True]("poisson")
    run[False, False]("absolute_error")
    run[False, True]("absolute_error")
