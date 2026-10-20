"""Validation-only text bridge to public Mojo estimators."""
from std.sys import argv
import numojo as nm
from moxie.svm import LinearSVC, LinearSVR, Dual, ClassWeight
from moxie.svm._contracts import Dense

def emit(a: Dense) raises:
    for i in range(a.size):
        print(a.unsafe_get(i))

def main() raises:
    var args = argv()
    var contents: String
    with open(args[1], "r") as file:
        contents = file.read()
    var t = contents.split()
    var n = Int(t[0])
    var p = Int(t[1])
    var regression = Int(t[2]) != 0
    var loss = String(t[3])
    var penalty = String(t[4])
    var dual = Dual(String(t[5]))
    var multi = String(t[6])
    var mode = String(t[7])
    var fit = Int(t[8]) != 0
    var scale = Float64(t[9])
    var c = Float64(t[10])
    var epsilon = Float64(t[11])
    var tol = Float64(t[12])
    var limit = Int(t[13])
    var seed = Int(t[14])
    var single = Int(t[15]) != 0
    var k = Int(t[16])
    var x = nm.zeros[DType.float64](nm.Shape(n, p))
    var y = nm.zeros[DType.float64](nm.Shape(n))
    var labels = List[Float64]()
    var sw = nm.zeros[DType.float64](nm.Shape(n))
    var cursor = 17
    for i in range(n * p):
        x.unsafe_set(i, Float64(t[cursor]))
        cursor += 1
    for i in range(n):
        var value = Float64(t[cursor])
        y.unsafe_set(i, value)
        labels.append(value)
        cursor += 1
    for i in range(n):
        sw.unsafe_set(i, Float64(t[cursor]))
        cursor += 1
    var costs = ClassWeight[Float64](mode)
    if mode == "explicit":
        for _ in range(k):
            costs.labels.append(Float64(t[cursor]))
            costs.values.append(Float64(t[cursor+1]))
            cursor += 2
    if regression:
        var model = LinearSVR(loss=loss, dual=dual.copy(), C=c, epsilon=epsilon, tol=tol, fit_intercept=fit, intercept_scaling=scale, max_iter=limit, random_state=seed)
        if single:
            model.fit(x.astype[DType.float32](), y, sw.copy())
        else:
            model.fit(x, y, sw.copy())
        print("RESULT", 1, p, model.n_iter_, model.fit_status_)
        emit(model.coef_)
        emit(model.intercept_)
        emit(model.predict(x))
        print(model.score(x, y, sw.copy()))
    else:
        var model = LinearSVC(loss=loss, penalty=penalty, dual=dual.copy(), multi_class=multi, class_weight=costs.copy(), C=c, tol=tol, fit_intercept=fit, intercept_scaling=scale, max_iter=limit, random_state=seed)
        if single:
            model.fit(x.astype[DType.float32](), labels, sw.copy())
        else:
            model.fit(x, labels, sw.copy())
        print("RESULT", model.coef_.shape[0], p, model.n_iter_, model.fit_status_)
        emit(model.coef_)
        emit(model.intercept_)
        emit(model.decision_function(x))
        var predicted = model.predict(x)
        for i in range(len(predicted)):
            print(predicted[i])
        print(model.score(x, labels, sw.copy()))
