import numojo as nm
from std.sys import argv
from moxie.svm._contracts import Dense, KernelOptions, KernelSpec
from moxie.svm._libsvm import LibSVM


def emit(name: String, values: Dense):
    print(name, end="")
    for i in range(values.size):
        print("", values.unsafe_get(i), end="")
    print()


def main() raises:
    var args = argv()
    var form = Int(args[1])
    var kind = Int(args[2])
    var n = Int(args[3])
    var p = Int(args[4])
    var k = Int(args[5])
    var weight_kind = Int(args[6])
    var weighted = weight_kind != 0
    var formulations: List[String] = ["c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr"]
    var kinds: List[String] = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
    var raw = nm.zeros[DType.float64](nm.Shape(n, p))
    var y = nm.zeros[DType.float64](nm.Shape(n))
    var weights = nm.zeros[DType.float64](nm.Shape(n))
    var classes = nm.zeros[DType.float64](nm.Shape(k if form < 2 else 0))
    for i in range(n):
        y.unsafe_set(i, Float64(i % k) if form < 2 else (1.0 if form == 2 else Float64((i * 13) % 23 - 11) / 7))
        weights.unsafe_set(i, ((-0.5 if weight_kind == 2 else 0.0) if i % 7 == 0 else 0.5 + Float64(i % 5) * 0.4) if weighted else 1.0)
        for j in range(p):
            raw.unsafe_set(i * p + j, Float64((i * 17 + j * 11) % 37 - 18) / 13 + Float64(i % k) * 2.0 + (2.0 if form == 2 else 0.0))
    for i in range(classes.size):
        classes.unsafe_set(i, 0.7 + Float64(i) * 0.4 if weighted else 1.0)
    var x = raw.copy()
    if kind == 4:
        x = nm.zeros[DType.float64](nm.Shape(n, n))
        for i in range(n):
            for j in range(n):
                var value = Float64(0)
                for c in range(p):
                    value += raw.unsafe_get(i * p + c) * raw.unsafe_get(j * p + c)
                x.unsafe_set(i * n + j, value)
    var kernel = KernelSpec(kinds[kind].copy(), gamma=0.4, degree=3, coef0=0.2)
    var options = KernelOptions(formulations[form].copy(), kernel.copy(), 1.7, 0.35, 0.12, 1e-7, Float64(args[11]), Int(args[7]) != 0, Int(args[9]) != 0, Int(args[8]), Int(args[10]))
    var engine = LibSVM()
    var model = engine.solve(x, y, weights, classes, options)
    print("support", end="")
    for i in range(model.support.size):
        print("", model.support.unsafe_get(i), end="")
    print()
    print("n_support", end="")
    for i in range(model.n_support.size):
        print("", model.n_support.unsafe_get(i), end="")
    print()
    emit("dual", model.dual_coef)
    emit("rho", model.rho)
    print("iterations", end="")
    for i in range(model.iterations.size):
        print("", model.iterations.unsafe_get(i), end="")
    print()
    print("status", model.fit_status)
    emit("prob_a", model.prob_a)
    emit("prob_b", model.prob_b)
    var decisions = engine.decision_values(x, model, kernel)
    emit("decisions", decisions)
    if form < 2 and options.probability:
        var probabilities = engine.class_probabilities(decisions, model)
        emit("probabilities", probabilities)
    else:
        print("probabilities")
