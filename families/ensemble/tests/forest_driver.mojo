"""Native side of forest_parity.py: fits ForestEngine cases and prints every compared quantity."""
import numojo as nm
from std.memory import bitcast
from std.sys import argv
from moxie._random import RandomState
from moxie.tree._classes import BaseDecisionTree, FeatureCount, SampleCount
from moxie.ensemble._records import Values, Indices
from moxie.ensemble._sampling import MaxSamples
from moxie.ensemble._forest_engine import ForestEngine, ForestClassWeight

def f64(token: String) raises -> Float64:
    return bitcast[DType.float64](Int64(atol(token)))

def f32(token: String) raises -> Float32:
    return bitcast[DType.float32](Int32(atol(token)))

def bits(value: Float64) -> String:
    return String(bitcast[DType.int64](value))

def floats[dtype: DType](fields: List[String], shape: nm.Shape) raises -> nm.NDArray[dtype]:
    var result = nm.zeros[dtype](shape)
    for i in range(result.size):
        comptime if dtype == DType.float32:
            result.unsafe_set(i, rebind[Scalar[dtype]](f32(fields[i + 1])))
        else:
            result.unsafe_set(i, rebind[Scalar[dtype]](f64(fields[i + 1])))
    return result^

def float_line(tag: String, values: Values) raises -> String:
    var line = tag
    for i in range(values.size):
        line += " " + bits(values.unsafe_get(i))
    return line

def int_line(tag: String, values: Indices) raises -> String:
    var line = tag
    for i in range(values.size):
        line += " " + String(values.unsafe_get(i))
    return line

def count(option: String) raises -> SampleCount:
    return SampleCount(f64(String(option[byte=1:]))) if option.startswith("f") else SampleCount(atol(option[byte=1:]))

def report[classifier: Bool, extra: Bool](mut forest: ForestEngine[classifier, extra], Q: nm.NDArray[DType.float32], phase: Int):
    try:
        var seeds = List[String]()
        for m in range(len(forest)):
            seeds.append(String(forest.base.estimators_[m].params.random_state.value()))
        var line = "SEEDS " + String(phase)
        for seed in seeds:
            line += " " + seed
        print(line)
        var samples = forest.estimators_samples_()
        for m in range(len(samples)):
            print(int_line("SAMPLE " + String(phase) + " " + String(m), samples[m]))
        print(int_line("APPLY " + String(phase), forest.apply(Q)))
        comptime if classifier:
            var probabilities = forest.predict_proba(Q)
            for k in range(len(probabilities)):
                print(float_line("PROBA " + String(phase) + " " + String(k), probabilities[k]))
        else:
            print(float_line("PRED " + String(phase), forest.predict(Q)))
        for m in range(len(forest)):
            ref t = forest.base.estimators_[m].tree_
            var stats = String("MSTAT ") + String(phase) + " " + String(m)
            for node in t.nodes:
                stats += " " + String(node.left_child) + " " + String(node.feature) + " " + bits(node.impurity) + " " + bits(node.weighted_n_node_samples)
            print(stats)
            print(float_line("MIMP " + String(phase) + " " + String(m), forest.base.estimators_[m].feature_importances_))
        print(float_line("IMP " + String(phase), forest.feature_importances_()))
        if forest.has_oob():
            print("OOB " + String(phase) + " " + bits(forest.oob_score_))
            comptime if classifier:
                for k in range(len(forest.oob_decision_function_)):
                    print(float_line("OOBD " + String(phase) + " " + String(k), forest.oob_decision_function_[k]))
            else:
                print(float_line("OOBP " + String(phase), forest.oob_prediction_))
        var warned = String()
        for warning in forest.warnings_:
            warned += " " + String(warning.split(":")[0])
        print("WARN " + String(phase) + warned)
    except e:
        print("ERROR " + String(phase) + " " + String(e))

def run_case[classifier: Bool, extra: Bool](lines: List[List[String]], mut at: Int) raises:
    ref head = lines[at]
    var options = Dict[String, String]()
    for i in range(9, len(head)):
        var pair = head[i].split("=")
        options[String(pair[0])] = String(pair[1])
    var index = atol(head[1])
    var n = atol(head[4])
    var p = atol(head[5])
    var outputs = atol(head[6])
    var q = atol(head[7])
    var n_estimators = atol(head[8])
    var mf = options["mf"]
    var features = FeatureCount(NoneType())
    if mf == "sqrt" or mf == "log2":
        features = FeatureCount(mf)
    elif mf.startswith("f"):
        features = FeatureCount(f64(String(mf[byte=1:])))
    elif mf.startswith("i"):
        features = FeatureCount(atol(mf[byte=1:]))
    var monotonic = Optional[List[Int]]()
    if options["mono"] != "none":
        var signs = List[Int]()
        for sign in options["mono"].split(","):
            signs.append(atol(sign))
        monotonic = signs^
    var depth = atol(options["depth"])
    var leaves = atol(options["leaves"])
    var template = BaseDecisionTree[classifier, extra](criterion=options["criterion"],
        max_depth=Optional[Int](depth) if depth > 0 else None, min_samples_split=count(options["mss"]),
        min_samples_leaf=count(options["msl"]), min_weight_fraction_leaf=f64(options["mwfl"]),
        max_features=features, max_leaf_nodes=Optional[Int](leaves) if leaves > 0 else None,
        min_impurity_decrease=f64(options["mid"]), ccp_alpha=f64(options["ccp"]), monotonic_cst=monotonic)
    var max_samples = MaxSamples(NoneType())
    if options["ms"].startswith("i"):
        max_samples = MaxSamples(atol(options["ms"][byte=1:]))
    elif options["ms"].startswith("f"):
        max_samples = MaxSamples(f64(String(options["ms"][byte=1:])))
    var X = floats[DType.float32](lines[at + 1], nm.Shape(n, p))
    var y = floats[DType.float64](lines[at + 2], nm.Shape(n) if outputs == 1 else nm.Shape(n, outputs))
    var Q = floats[DType.float32](lines[at + 3], nm.Shape(q, p))
    var weights = Optional[Values]()
    if lines[at + 4][1] != "none":
        weights = floats[DType.float64](lines[at + 4], nm.Shape(n))
    at += 5
    var class_weight = ForestClassWeight(NoneType())
    if options["cw"] == "factors":
        var factors = List[Values]()
        for _ in range(outputs):
            factors.append(floats[DType.float64](lines[at], nm.Shape(len(lines[at]) - 1)))
            at += 1
        class_weight = ForestClassWeight(factors^)
    elif options["cw"] != "none":
        class_weight = ForestClassWeight(options["cw"])
    var forest = ForestEngine[classifier, extra](template^, n_estimators, bootstrap=options["bootstrap"] == "1",
        oob_score=options["oob"] == "1", warm_start=False, class_weight=class_weight, max_samples=max_samples)
    var seed = atol(options["seed"])
    var share = options["shared"] == "1"
    var grow = atol(options["warm"])
    var shared = RandomState(seed)
    print("CASE", index)
    try:
        if not share:
            var rng = RandomState(seed)
            forest.fit(X, y, weights, rng)
        else:
            forest.fit(X, y, weights, shared)
    except e:
        print("ERROR 0 " + String(e))
        return
    report(forest, Q, 0)
    if grow > 0:  # warm-start growth, then a refit with the same estimator count
        forest.warm_start = True
        forest.base.n_estimators = grow
        try:
            if not share:
                var rng = RandomState(seed)
                forest.fit(X, y, weights, rng)
            else:
                forest.fit(X, y, weights, shared)
        except e:
            print("ERROR 1 " + String(e))
            return
        report(forest, Q, 1)
        try:
            if not share:
                var rng = RandomState(seed)
                forest.fit(X, y, weights, rng)
            else:
                forest.fit(X, y, weights, shared)
        except e:
            print("ERROR 2 " + String(e))
            return
        report(forest, Q, 2)

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
