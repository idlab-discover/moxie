import numojo as nm
from moxie.tree._classes import BaseDecisionTree
from moxie.tree import export_graphviz, plot_tree

def run[classifier: Bool, extra: Bool]() raises:
    var tag = Int(classifier) + 2 * Int(extra)
    for case_id in range(20):
        var n = 16 + case_id
        var outputs = 2 if case_id % 3 == 0 else 1
        var X = nm.zeros[DType.float32](nm.Shape(n, 2))
        var y = nm.zeros[DType.float64](nm.Shape(n, outputs))
        var w = nm.zeros[DType.float64](nm.Shape(n))
        for i in range(n):
            X.unsafe_set(i * 2, Float32((i * 3 + case_id) % 13) / 4)
            X.unsafe_set(i * 2 + 1, Float32(i % 7) / 4)
            w.unsafe_set(i, Float64(1 + i % 3) / 4)
            for k in range(outputs):
                var v = Float64((i * 7 + case_id + k) % 13) / 4
                comptime if classifier:
                    v = Float64((i + i // 3 + k) % (1 if case_id == 1 else 7 if case_id == 19 else 3))
                y.unsafe_set(i * outputs + k, v)
        var criterion = String("gini" if classifier else "squared_error")
        if case_id % 4 == 2:
            criterion = "entropy" if classifier else "absolute_error"
        elif case_id % 4 == 3:
            criterion = "log_loss" if classifier else "poisson"
        var model = BaseDecisionTree[classifier, extra](criterion=criterion, random_state=case_id, max_depth=14 if case_id == 19 else 4, max_leaf_nodes=Optional[Int](5) if case_id % 4 == 2 else None)
        _ = model.fit(X, y, sample_weight=w)
        for mode in range(16):
            print("REPORT", tag, case_id, mode)
            var names: Optional[List[String]] = None
            if mode == 1:
                names = Optional[List[String]](["height", "colour intensity"])
            var classes: Optional[List[String]] = None
            comptime if classifier:
                if outputs == 1 and case_id != 1 and mode == 1:
                    var names_for_classes = List[String]()
                    for c in range(model.n_classes_[0]):
                        names_for_classes.append("class " + String(c))
                    classes = names_for_classes^
            var palette: Optional[List[String]] = None
            if mode == 14:
                var custom = List[String]()
                for c in range(model.tree_.n_classes[0]):
                    custom.append("#ff0000" if c == 0 else "#00ff00" if c == 1 else "#0000ff")
                palette = custom^
            print(export_graphviz(model, feature_names=names, class_names=classes, max_depth=Optional[Int](0) if mode == 2 or mode == 9 or mode == 13 else None, precision=3, fill_colors=palette, label="root" if mode == 3 else "none" if mode == 4 else "all", impurity=mode != 5, node_ids=mode == 6, proportion=mode == 7, rounded=mode == 6 or mode == 12, rotate=mode == 7 or mode == 12, filled=mode >= 8, leaves_parallel=mode == 10 or mode == 12, special_characters=mode == 11 or mode == 12 or mode == 13))
            print("SVG")
            print(plot_tree(model, feature_names=names, class_names=classes, max_depth=Optional[Int](0) if mode == 2 or mode == 9 or mode == 13 else None, precision=3, fontsize=Optional[Float64](Float64(10)) if mode == 15 else None, fill_colors=palette, label="root" if mode == 3 else "none" if mode == 4 else "all", impurity=mode != 5, node_ids=mode == 6, proportion=mode == 7, rounded=mode == 6 or mode == 12, filled=mode >= 8))

def main() raises:
    run[False, False]()
    run[True, False]()
    run[False, True]()
    run[True, True]()
