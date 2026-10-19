import numojo as nm
from moxie.tree._classes import BaseDecisionTree
from moxie.tree import export_text
from moxie.tree._export import _fixed
from std.memory import bitcast

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
                    v = Float64((i + i // 3 + k) % (1 if case_id == 1 else 3))
                y.unsafe_set(i * outputs + k, v)
        var model = BaseDecisionTree[classifier, extra](random_state=case_id, max_depth=4)
        _ = model.fit(X, y, sample_weight=w)
        for mode in range(4):
            print("REPORT", tag, case_id, mode)
            var names: Optional[List[String]] = None
            if mode == 1:
                names = Optional[List[String]](["height", "colour intensity"])
            var classes: Optional[List[String]] = None
            comptime if classifier:
                if outputs == 1 and case_id != 1 and mode == 1:
                    classes = Optional[List[String]](["first", "second", "third"])
            print(export_text(model, feature_names=names, class_names=classes, max_depth=0 if mode == 2 else 10, spacing=1 if mode == 3 else 3, decimals=mode, show_weights=mode % 2 == 1), end="")

def main() raises:
    run[False, False]()
    run[True, False]()
    run[False, True]()
    run[True, True]()
    print("FORMATS")
    var state: UInt64 = 1234
    for i in range(512):
        state = state * UInt64(6364136223846793005) + 1
        var value = bitcast[DType.float64](state)
        var decimals = i % 21
        print(state, decimals, _fixed(value, decimals))
