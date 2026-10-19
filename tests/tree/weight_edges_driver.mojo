import numojo as nm
from moxie.tree._classes import BaseDecisionTree

def run[classifier: Bool, extra: Bool](kind: Int) raises:
    for scenario in range(12):
        var X = nm.zeros[DType.float32](nm.Shape(4, 1))
        var y = nm.zeros[DType.float64](nm.Shape(4))
        var w = nm.zeros[DType.float64](nm.Shape(4))
        for i in range(4):
            X.unsafe_set(i, Float32(i))
            y.unsafe_set(i, Float64(1 if scenario >= 8 else 1 + i % 2))
            w.unsafe_set(i, Float64(0 if scenario % 4 == 0 else (-1 if scenario % 4 == 1 else (1 if i % 2 == 0 else -1))))
            if scenario % 4 == 3:
                w.unsafe_set(i, 1)
        var model = BaseDecisionTree[classifier, extra](random_state=7, criterion="gini" if kind == 0 else ("entropy" if kind == 1 else ("squared_error" if kind == 2 else ("poisson" if kind == 3 else "absolute_error"))))
        if scenario >= 4:
            model.params.max_leaf_nodes = 3
            model.params.min_weight_fraction_leaf = 0.5 if scenario >= 8 else 0.25
            model.params.ccp_alpha = 0.02
        if scenario >= 8:
            model.params.monotonic_cst = Optional[List[Int]]([1])
        try:
            _ = model.fit(X, y, w.copy())
        except e:
            print("ERROR", kind + 5 * Int(extra), scenario)
            continue
        print("CASE", kind + 5 * Int(extra), scenario, model.tree_.node_count)
        for i in range(model.tree_.node_count):
            var node = model.tree_.nodes[i].copy()
            print("NODE", kind + 5 * Int(extra), scenario, i, node.impurity, node.n_node_samples, node.weighted_n_node_samples)
        for i in range(model.tree_.value.size):
            print("VALUE", kind + 5 * Int(extra), scenario, i, model.tree_.value.unsafe_get(i))
        var pred = model.predict(X)
        for i in range(pred.size):
            print("PRED", kind + 5 * Int(extra), scenario, i, pred.unsafe_get(i))

def main() raises:
    run[True, False](0)
    run[True, False](1)
    run[False, False](2)
    run[False, False](3)
    run[False, False](4)
    run[True, True](0)
    run[True, True](1)
    run[False, True](2)
    run[False, True](3)
    run[False, True](4)
