import numojo as nm
from moxie._labels import StringTarget
from moxie.tree._classes import BaseDecisionTree
from moxie.tree._records import TreeTable, TreeColumn

# Ordered deliberately differently from classes_. Int64 neighbours exceed 2**53.
def numeric_label[dtype: DType](code: Int) -> Scalar[dtype]:
    comptime if dtype == DType.int64:
        if code == 0:
            return Scalar[dtype](9007199254740993)
        if code == 1:
            return Scalar[dtype](-9223372036854775807 - 1)
        return Scalar[dtype](9007199254740992)
    else:
        return Scalar[dtype](8 if code == 0 else (-3 if code == 1 else 2))

def run[extra: Bool, dtype: DType, strings: Bool](tag: Int) raises:
    comptime Model = BaseDecisionTree[True, extra, dtype, strings]
    for scenario in range(8):
        var n = 9 + scenario
        var outputs = 1 if scenario < 4 else 2
        var X = nm.zeros[DType.float32](nm.Shape(n, 2))
        var weights = nm.full[DType.float64](nm.Shape(n), 1)
        var y: Model.Target
        comptime if strings:
            var labels = List[String]()
            var names: List[String] = ["zebra", "amber", "é"]
            for i in range(n):
                for k in range(outputs):
                    labels.append(names[(i + i // 3 + k) % 3].copy())
            var target = StringTarget(labels) if outputs == 1 else StringTarget(labels, n, outputs)
            y = rebind[Model.Target](target).copy()
        else:
            var target = nm.zeros[dtype](nm.Shape(n) if outputs == 1 else nm.Shape(n, outputs))
            for i in range(n):
                for k in range(outputs):
                    target.unsafe_set(i * outputs + k, numeric_label[dtype]((i + i // 3 + k) % 3))
            y = rebind[Model.Target](target).copy()
        for i in range(n):
            weights.unsafe_set(i, Float64(0 if i % 7 == 0 else 1 + i % 3))
            for j in range(2):
                X.unsafe_set(i * 2 + j, Float32((i * (j + 3) + scenario + j * 2) % 19) / 4)
        var model = Model(random_state=scenario, max_depth=3, criterion="entropy" if scenario % 2 else "gini")
        if scenario % 3 == 1:
            model.params.class_weight = "balanced"
        if scenario == 2:
            var class_weights = model.WeightMap()
            comptime if strings:
                rebind[Dict[String, Float64]](class_weights)["zebra"] = 2
            else:
                rebind[Dict[Scalar[dtype], Float64]](class_weights)[numeric_label[dtype](0)] = 2
            model.params.class_weight = class_weights^
        if scenario == 6:
            model.params.max_leaf_nodes = 4
        _ = model.fit(X, y, weights.copy())
        var prediction = model.predict(X)
        print("CASE", tag, scenario, model.n_outputs_, model.tree_.node_count)
        for k in range(outputs):
            for c in range(model.n_classes_[k]):
                print("CLASS", tag, scenario, k, c, model._class_label(k, c))
        for i in range(n):
            for k in range(outputs):
                print("PRED", tag, scenario, i, k, model._target_value(prediction, i, k))
        var probabilities = model.predict_proba(X)
        for k in range(outputs):
            for i in range(probabilities[k].size):
                print("PROBA", tag, scenario, k, i, probabilities[k].unsafe_get(i))
        if outputs == 1:
            print("SCORE", tag, scenario, model.score(X, y, weights.copy()))
        for i in range(model.tree_.node_count):
            var node = model.tree_.nodes[i].copy()
            print("NODE", tag, scenario, i, node.left_child, node.right_child, node.feature, node.threshold, node.impurity)

def main() raises:
    run[False, DType.float64, True](0)
    run[True, DType.float64, True](1)
    run[False, DType.int64, False](2)
    run[True, DType.int64, False](3)
    run[False, DType.float32, False](4)
    run[True, DType.float32, False](5)
