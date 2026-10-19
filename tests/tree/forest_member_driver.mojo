"""Builds every member of a reference forest from one prepared input; see forest_member_parity.py."""
import numojo as nm
from std.memory import bitcast
from std.sys import argv
from moxie.tree._classes import BaseDecisionTree, FeatureCount, SampleCount
from moxie.tree._records import Features, Values

def f64(token: String) raises -> Float64:
    return bitcast[DType.float64](Int64(atol(token)))

def f32(token: String) raises -> Float32:
    return bitcast[DType.float32](Int32(atol(token)))

def bits(value: Float64) -> String:
    return String(bitcast[DType.int64](value))

def floats[dtype: DType](fields: List[String], shape: nm.Shape, first: Int = 1) raises -> nm.NDArray[dtype]:
    var result = nm.zeros[dtype](shape)
    for i in range(result.size):
        comptime if dtype == DType.float32:
            result.unsafe_set(i, rebind[Scalar[dtype]](f32(fields[i + first])))
        else:
            result.unsafe_set(i, rebind[Scalar[dtype]](f64(fields[i + first])))
    return result^

def emit(tag: String, index: Int, member: Int, values: List[String]):
    var line = tag + " " + String(index) + " " + String(member)
    for value in values:
        line += " " + value
    print(line)

def run_case[classifier: Bool, extra: Bool](lines: List[List[String]], mut at: Int) raises:
    ref head = lines[at]
    var index = atol(head[1])
    var criterion = String(head[4])
    var n = atol(head[5])
    var p = atol(head[6])
    var outputs = atol(head[7])
    var q = atol(head[8])
    var depth = atol(head[9])
    var mf = String(head[10])
    var features = FeatureCount(NoneType())
    if mf == "sqrt" or mf == "log2":
        features = FeatureCount(mf)
    elif mf.startswith("f"):
        features = FeatureCount(f64(String(mf[byte=1:])))
    elif mf.startswith("i"):
        features = FeatureCount(atol(mf[byte=1:]))
    var msl = String(head[11])
    var leaf = SampleCount(f64(String(msl[byte=1:]))) if msl.startswith("f") else SampleCount(atol(msl[byte=1:]))
    var leaves = atol(head[12])
    var mono = String(head[16])
    var monotonic = Optional[List[Int]]()
    if mono != "none":
        var signs = List[Int]()
        for sign in mono.split(","):
            signs.append(atol(sign))
        monotonic = signs^
    var members = atol(head[17])
    comptime Model = BaseDecisionTree[classifier, extra]
    var model = Model(criterion=criterion, max_depth=Optional[Int](depth) if depth > 0 else None,
        min_samples_leaf=leaf, max_features=features, max_leaf_nodes=Optional[Int](leaves) if leaves > 0 else None,
        ccp_alpha=f64(head[13]), min_weight_fraction_leaf=f64(head[14]), min_impurity_decrease=f64(head[15]),
        monotonic_cst=monotonic)
    var X = floats[DType.float32](lines[at + 1], nm.Shape(n, p))
    var y = floats[DType.float64](lines[at + 2], nm.Shape(n) if outputs == 1 else nm.Shape(n, outputs))
    var Q = floats[DType.float32](lines[at + 3], nm.Shape(q, p))
    at += 5  # The caller weight line is folded into each member's weights by the forest.
    # Prepare and encode once; every member reuses both.
    var prepared = model.prepare_fit(X, y)
    var encoded = prepared.encoder.encode(Q, criterion == "absolute_error", Bool(monotonic))
    for m in range(members):
        ref fields = lines[at]
        at += 1
        var tree = model.__sklearn_clone__()
        var params = tree.get_params()
        params.random_state = atol(fields[1])
        _ = tree.set_params(params)
        if fields[2] == "none":
            _ = tree.fit_prepared(prepared)
        else:
            _ = tree.fit_prepared(prepared, sample_weight=floats[DType.float64](fields, nm.Shape(n), 2))
        ref t = tree.tree_
        var structure = List[String]()
        var stats = List[String]()
        for node in t.nodes:
            structure.append(String(node.left_child))
            structure.append(String(node.right_child))
            structure.append(String(node.feature))
            structure.append(String(node.n_node_samples))
            structure.append(String(node.missing_go_to_left))
            stats.append(bits(node.threshold if node.left_child >= 0 else 0))
            stats.append(bits(node.impurity))
            stats.append(bits(node.weighted_n_node_samples))
        emit("STRUCT", index, m, structure)
        emit("STAT", index, m, stats)
        var values = List[String]()
        for i in range(t.node_count * t.n_outputs * t.max_n_classes):
            values.append(bits(t.value.unsafe_get(i)))
        emit("VALUE", index, m, values)
        var importances = List[String]()
        for j in range(p):
            importances.append(bits(tree.feature_importances_.unsafe_get(j)))
        emit("IMP", index, m, importances)
        var leaf_ids = tree.apply_encoded(encoded)
        var applied = List[String]()
        for i in range(q):
            applied.append(String(leaf_ids.unsafe_get(i)))
        emit("LEAF", index, m, applied)
        var predicted = List[String]()
        comptime if classifier:
            var proba = tree.predict_proba_encoded(encoded)
            for i in range(q):
                for k in range(outputs):
                    for c in range(tree.n_classes_[k]):
                        predicted.append(bits(proba[k].unsafe_get(i * tree.n_classes_[k] + c)))
        else:
            var prediction = tree.predict_encoded(encoded)
            for i in range(q * outputs):
                predicted.append(bits(prediction.unsafe_get(i)))
        emit("PRED", index, m, predicted)

def main() raises:
    var text = open(argv()[1], "r").read()
    var lines = List[List[String]]()
    for line in text.split("\n"):
        if line.byte_length() > 0:
            var tokens = List[String]()
            for token in line.split(" "):
                tokens.append(String(token))
            lines.append(tokens^)
    var at = 0
    while at < len(lines):
        ref head = lines[at]
        var classifier = head[2] == "1"
        var extra = head[3] == "1"
        if classifier and extra:
            run_case[True, True](lines, at)
        elif classifier:
            run_case[True, False](lines, at)
        elif extra:
            run_case[False, True](lines, at)
        else:
            run_case[False, False](lines, at)
