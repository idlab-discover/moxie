import numojo as nm
from moxie._labels import StringTarget
from moxie._random import RandomState
from moxie.tree._classes import BaseDecisionTree, CategoricalFeatures
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
    var model = Model(random_state=0)
    for scenario in range(32):
        var n = 17 + scenario
        var outputs = 1
        var X = nm.zeros[DType.float32](nm.Shape(n, 2))
        var weights = nm.full[DType.float64](nm.Shape(n), 1)
        var rng = RandomState(7300 + scenario)
        var codes = nm.zeros[DType.int64](nm.Shape(n))
        for i in range(n):
            codes.unsafe_set(i, Int64(rng.randint(0, 2) * (2 if scenario % 2 else 1)))
            weights.unsafe_set(i, Float64(rng.randint(1, 5)))
            if i % 7 == 0:
                weights.unsafe_set(i, -0.125)
            for j in range(2):
                X.unsafe_set(i * 2 + j, Float32(rng.randint(0, 7 if j == 0 else 127)) / 4)
        var y: Model.Target
        comptime if strings:
            var labels = List[String]()
            var names: List[String] = ["zebra", "amber", "é"]
            for i in range(n):
                for _ in range(outputs):
                    labels.append(names[Int(codes.unsafe_get(i))].copy())
            var target = StringTarget(labels) if outputs == 1 else StringTarget(labels, n, outputs)
            y = rebind[Model.Target](target).copy()
        else:
            var target = nm.zeros[dtype](nm.Shape(n) if outputs == 1 else nm.Shape(n, outputs))
            for i in range(n):
                for k in range(outputs):
                    target.unsafe_set(i * outputs + k, numeric_label[dtype](Int(codes.unsafe_get(i))))
            y = rebind[Model.Target](target).copy()
        var selection = CategoricalFeatures(Optional[List[Int]]([0]))
        if scenario % 4 == 1:
            var mask: List[Bool] = [True, False]
            selection = mask^
        elif scenario % 4 == 2:
            var names: List[String] = ["category"]
            selection = names^
        elif scenario % 4 == 3:
            selection = "from_dtype"
        var configured = Model(random_state=scenario, max_depth=4, criterion="entropy" if scenario % 2 else "gini", categorical_features=selection^)
        configured.params.ccp_alpha = 0.02 if scenario % 3 else 0.1
        if scenario % 2:
            configured.params.max_leaf_nodes = 5
        _ = model.set_params(configured.get_params())
        var columns = List[TreeColumn]()
        for j in range(2):
            var column = nm.zeros[DType.float32](nm.Shape(n))
            for i in range(n):
                column.unsafe_set(i, X.unsafe_get(i * 2 + j))
            columns.append(TreeColumn(column^))
        var table = TreeTable(columns, feature_names=Optional[List[String]](["category", "numeric"]), categorical_mask=Optional[List[Bool]]([True, False]))
        _ = model.fit(table, y, weights.copy())
        var prediction = model.predict(table)
        print("CASE", tag, scenario, model.n_outputs_, model.tree_.node_count)
        for k in range(outputs):
            for c in range(model.n_classes_[k]):
                print("CLASS", tag, scenario, k, c, model._class_label(k, c))
        for i in range(n):
            for k in range(outputs):
                print("PRED", tag, scenario, i, k, model._target_value(prediction, i, k))
        var probabilities = model.predict_proba(table)
        for k in range(outputs):
            for i in range(probabilities[k].size):
                print("PROBA", tag, scenario, k, i, probabilities[k].unsafe_get(i))
        if outputs == 1:
            print("SCORE", tag, scenario, model.score(table, y, weights.copy()))
        var leaves = model.apply(table)
        for i in range(n):
            print("LEAF", tag, scenario, i, leaves.unsafe_get(i))
        for i in range(model.tree_.node_count):
            var node = model.tree_.nodes[i].copy()
            print("NODE", tag, scenario, i, node.left_child, node.right_child, node.feature, node.threshold, node.impurity, node.weighted_n_node_samples)
            for b in range(8):
                print("BITS", tag, scenario, i * 8 + b, node.left_cat_bitset[b])

def main() raises:
    run[False, DType.float64, True](0)
    run[True, DType.float64, True](1)
    run[False, DType.int64, False](2)
    run[True, DType.int64, False](3)
    run[False, DType.float32, False](4)
    run[True, DType.float32, False](5)
