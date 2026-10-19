import numojo as nm
from moxie.tree._classes import BaseDecisionTree, CategoricalFeatures
from moxie.tree import TreeTable, TreeColumn, export_text
from moxie._categorical import categorical_column

def mixed(X: nm.NDArray[DType.float32]) raises -> TreeTable:
    var strings = List[Optional[String]]()
    var columns = List[TreeColumn]()
    for i in range(X.shape[0]):
        var x = X.unsafe_get(i * 3)
        var code = Int((x + 13) / 2) if x == x else 0
        strings.append(Optional[String](String("v") + ("0" if code < 10 else "") + String(code)) if x == x else None)
    columns.append(TreeColumn(categorical_column(strings)))
    for j in range(1, 3):
        var column = nm.zeros[DType.float32](nm.Shape(X.shape[0]))
        for i in range(X.shape[0]):
            column.unsafe_set(i, X.unsafe_get(i * 3 + j))
        columns.append(TreeColumn(column^))
    return TreeTable(columns, feature_names=Optional[List[String]](["color", "height", "group"]), categorical_mask=Optional[List[Bool]]([True, False, True]))

def run[classifier: Bool, extra: Bool]() raises:
    var tag = Int(classifier) + 2 * Int(extra)
    for case_id in range(40):
        var n = 16 + case_id
        var p = 3
        var outputs = 2 if extra and case_id % 4 == 0 else 1
        var X = nm.zeros[DType.float32](nm.Shape(n, p))
        var y = nm.zeros[DType.float64](nm.Shape(n, outputs))
        var w = nm.zeros[DType.float64](nm.Shape(n))
        for i in range(n):
            w.unsafe_set(i, Float64(0 if i % 11 == 0 else 1 + i % 3))
            if case_id >= 30 and i % 11 == 0:
                w.unsafe_set(i, -0.125)
            for j in range(p):
                var value = Float32((i * (3 + j) + case_id + j) % (2 + case_id % 19)) * 2 - 13
                if j == 1:
                    value = Float32(i % 7) / 4
                if case_id % 3 == 0 and i % 5 == 0:
                    value = Float32(Float64("nan"))
                if case_id == 1 and j == 0:
                    value = Float32(Float64("nan"))
                X.unsafe_set(i * p + j, value)
            for k in range(outputs):
                var value = Float64((i * 7 + case_id + k) % 13) / 4
                comptime if classifier:
                    value = Float64((i + i // 3 + k) % (3 if extra else 2))
                y.unsafe_set(i * outputs + k, value)
        var selection = CategoricalFeatures(Optional[List[Int]]([0, 2]))
        if case_id % 4 == 1:
            var mask: List[Bool] = [True, False, True]
            selection = mask^
        elif case_id % 4 == 2:
            var names: List[String] = ["color", "group"]
            selection = names^
        elif case_id % 4 == 3:
            selection = "from_dtype"
        var model = BaseDecisionTree[classifier, extra](categorical_features=selection^, random_state=case_id, max_depth=4)
        var params = model.get_params()
        params.max_features = 1 + case_id % 3
        if case_id % 4 == 0:
            params.max_leaf_nodes = 4
        if case_id % 5 == 0:
            params.ccp_alpha = 0.02
        if case_id % 2 == 0:
            params.criterion = "entropy" if classifier else "poisson"
        _ = model.set_params(params)
        _ = model.fit(mixed(X), y, sample_weight=w)
        var fitted_names = model.feature_names_in_.value().copy()
        print("NAMES", tag, case_id, fitted_names[0], fitted_names[1], fitted_names[2])
        print("TEXT", tag, case_id, "{!r}".format(export_text(model)))
        print("CASE", tag, case_id, model.tree_.node_count)
        for i in range(model.tree_.node_count):
            var node = model.tree_.nodes[i].copy()
            print("NODE", tag, case_id, i, node.left_child, node.right_child, node.feature, node.impurity, node.n_node_samples, node.weighted_n_node_samples, node.missing_go_to_left, node.split_kind)
            print("THRESHOLD", tag, case_id, i, node.threshold)
            for b in range(8):
                print("BITS", tag, case_id, i * 8 + b, node.left_cat_bitset[b])
        for i in range(model.tree_.value.size):
            print("VALUE", tag, case_id, i, model.tree_.value.unsafe_get(i))
        var query = X.copy()
        query.unsafe_set(0, 9999)
        query.unsafe_set(2, Float32(Float64("nan")))
        var prediction = model.predict(mixed(query))
        var leaves = model.apply(mixed(query))
        comptime if classifier:
            var probabilities = model.predict_proba(mixed(query))
            var offset = 0
            for column in probabilities:
                for i in range(column.size):
                    print("PROBA", tag, case_id, offset + i, column.unsafe_get(i))
                offset += column.size
        for i in range(prediction.size):
            print("PRED", tag, case_id, i, prediction.unsafe_get(i))
        for i in range(leaves.size):
            print("LEAF", tag, case_id, i, leaves.unsafe_get(i))

def main() raises:
    run[False, False]()
    run[True, False]()
    run[False, True]()
    run[True, True]()
