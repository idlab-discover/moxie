from std.sys import argv
import numojo as nm
from balsa import save
from moxie.tree._classes import BaseDecisionTree
from moxie.tree import export_treelite

def run[classifier: Bool, extra: Bool = False](kind: Int, treelite: String) raises:
    var tag = kind + (4 if extra else 0)
    for case_id in range(26):
        var n = 8 + case_id
        var p = 1 + case_id % 4
        var outputs = 1 if case_id >= 18 else 1 + case_id % 3
        var X = nm.zeros[DType.float32](nm.Shape(n, p))
        var y = nm.zeros[DType.float64](nm.Shape(n, outputs))
        var weights = nm.zeros[DType.float64](nm.Shape(n))
        for i in range(n):
            weights.unsafe_set(i, Float64(0 if i % 7 == 0 else 1 + i % 3))
            if case_id >= 24 and i % 7 == 0:
                weights.unsafe_set(i, -0.125)
            for j in range(p):
                var x = Float32((i * (j + 3) + j * 5 + case_id) % 23) / 4
                if case_id % 5 == 0 and i % 4 == 0:
                    x = Float32(Float64("nan"))
                if case_id % 7 == 0 and j == 0:
                    x = 1
                X.unsafe_set(i * p + j, x)
            for k in range(outputs):
                var v = Float64((i * 7 + k * 3 + case_id) % 17) / 4
                comptime if classifier:
                    v = Float64(10 + 5 * ((i + i // 3 + k) % (2 + k % 2)))
                y.unsafe_set(i * outputs + k, v)
        var estimator = BaseDecisionTree[classifier, extra](random_state=case_id)
        estimator.params.criterion = "gini" if kind == 0 else ("entropy" if kind == 1 else ("squared_error" if kind == 2 else "poisson"))
        estimator.params.max_depth = 4
        if case_id % 6 == 2:
            estimator.params.max_leaf_nodes = 4
        estimator.params.min_samples_leaf = 1 + case_id % 2
        if not extra or case_id % 3 != 0:
            estimator.params.max_features = 1 + case_id % p
        if not extra:
            estimator.params.splitter = "random" if case_id % 3 == 1 else "best"
        estimator.params.ccp_alpha = 0.02 if case_id % 4 == 2 else 0
        comptime if classifier:
            if case_id % 4 == 3:
                estimator.params.class_weight = "balanced"
        if case_id >= 18 and case_id < 24:
            var signs = List[Int]()
            for j in range(p):
                signs.append(1 if j % 2 == 0 else -1)
            estimator.params.monotonic_cst = signs^
            if case_id == 18:
                estimator.params.max_features = "sqrt"
            elif case_id == 19:
                estimator.params.max_features = "log2"
            elif case_id == 21:
                estimator.params.min_samples_leaf = Float64(0.15)
                estimator.params.min_samples_split = Float64(0.3)
            elif case_id == 22:
                estimator.params.max_features = Float64(0.6)
            elif case_id == 23:
                estimator.params.min_impurity_decrease = 0.015
        if case_id == 20:
            estimator.params.monotonic_cst = None
        _ = estimator.fit(X, y, weights.copy())
        print("CASE", tag, case_id, estimator.tree_.node_count, estimator.get_depth(), estimator.get_n_leaves())
        for i in range(estimator.tree_.node_count):
            var node = estimator.tree_.nodes[i].copy()
            print("NODE", tag, case_id, i, node.left_child, node.right_child, node.feature, node.threshold, node.impurity, node.n_node_samples, node.weighted_n_node_samples, node.missing_go_to_left)
            for j in range(estimator.n_outputs_ * estimator.tree_.max_n_classes):
                print("VALUE", tag, case_id, i, j, estimator.tree_.value.unsafe_get(i * estimator.n_outputs_ * estimator.tree_.max_n_classes + j))
        var pred = estimator.predict(X)
        var leaves = estimator.apply(X)
        for i in range(pred.size):
            print("PRED", tag, case_id, i, pred.unsafe_get(i))
        for i in range(leaves.size):
            print("LEAF", tag, case_id, i, leaves.unsafe_get(i))
        for j in range(p):
            print("IMPORTANCE", tag, case_id, j, estimator.feature_importances_.unsafe_get(j))
        if not classifier or outputs == 1:
            print("SCORE", tag, case_id, estimator.score(X, y.copy() if outputs > 1 else y.reshape(nm.Shape(n)), weights.copy()))
        save(export_treelite(estimator), treelite + "/" + String(tag) + "_" + String(case_id) + ".tl")
        var path = estimator.cost_complexity_pruning_path(X, y, weights.copy())
        for i in range(path.ccp_alphas.size):
            print("PATH", tag, case_id, i, path.ccp_alphas.unsafe_get(i), path.impurities.unsafe_get(i))

def main() raises:
    # Treelite checkpoints of every fitted case go to argv[1].
    var treelite = String(argv()[1])
    run[True](0, treelite)
    run[True](1, treelite)
    run[False](2, treelite)
    run[False](3, treelite)
    run[True, True](0, treelite)
    run[True, True](1, treelite)
    run[False, True](2, treelite)
    run[False, True](3, treelite)
