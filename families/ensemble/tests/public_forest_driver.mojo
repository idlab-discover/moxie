"""Read raw-bit fixtures and exercise public forest estimators."""
import numojo as nm
from std.sys import argv
from moxie.ensemble._forest import Forest
from moxie.tree._classes import ClassWeight, ClassWeightMap
from moxie.ensemble._records import Values
from forest_driver import floats, float_line, int_line


def report[classifier: Bool, extra: Bool, multi: Bool](model: Forest[classifier, extra, multi], X: nm.NDArray[DType.float32], y: Values, Q: nm.NDArray[DType.float32]) raises:
    var prediction = model.predict(Q)
    print("SHAPE", prediction.ndim, prediction.shape[0], model.n_outputs_, model.n_features_in_)
    print(float_line("PRED", prediction))
    print(float_line("IMP", model.feature_importances_))
    print("SCORE", model.score(X, y))
    print(int_line("APPLY", model.apply(Q)))
    var samples = model.estimators_samples_()
    for rows in samples:
        print(int_line("SAMPLES", rows))
    comptime if classifier:
        var probabilities = model.predict_proba(Q)
        var logarithms = model.predict_log_proba(Q)
        comptime if multi:
            for k in range(model.n_outputs_):
                print(float_line("CLASSES", rebind[List[Values]](model.classes_)[k]))
                print(float_line("PROBA", rebind[List[Values]](probabilities)[k]))
                print(float_line("LOG", rebind[List[Values]](logarithms)[k]))
        else:
            print(float_line("CLASSES", rebind[Values](model.classes_)))
            print(float_line("PROBA", rebind[Values](probabilities)))
            print(float_line("LOG", rebind[Values](logarithms)))
    if model.oob_score:
        print("OOB", model.oob_score_)
        comptime if classifier:
            comptime if multi:
                for k in range(model.n_outputs_):
                    print(float_line("OOBD", rebind[List[Values]](model.oob_decision_function_)[k]))
            else:
                print(float_line("OOBD", rebind[Values](model.oob_decision_function_)))
        else:
            print(float_line("OOBP", model.oob_prediction_))


def run_case[classifier: Bool, extra: Bool, multi: Bool](lines: List[List[String]], at: Int) raises:
    ref h = lines[at]
    var n = atol(h[3])
    var p = atol(h[4])
    var outputs = atol(h[5])
    var q = atol(h[6])
    var X = floats[DType.float32](lines[at+1], nm.Shape(n, p))
    var y = floats[DType.float64](lines[at+2], nm.Shape(n) if outputs == 1 else nm.Shape(n, outputs))
    var Q = floats[DType.float32](lines[at+3], nm.Shape(q, p))
    var weights = Optional[Values]()
    if lines[at+4][1] != "none":
        weights = floats[DType.float64](lines[at+4], nm.Shape(n))
    var cw = ClassWeight(NoneType())
    if h[12] == "dict":
        var mapping = ClassWeightMap()
        mapping[-3] = 2.5
        mapping[7] = 0.5
        cw = ClassWeight(mapping^)
    elif h[12] != "none":
        cw = ClassWeight(h[12])
    var model = Forest[classifier, extra, multi]()
    _ = model.set_params(n_estimators=atol(h[7]), criterion=h[8], bootstrap=h[9] == "1", oob_score=h[10] == "1", random_state=atol(h[11]), class_weight=cw.copy())
    _ = model.fit(X, y, weights)
    report(model, X, y, Q)
    if h[13] != "0":
        _ = model.set_params(warm_start=True, n_estimators=atol(h[13]))
        _ = model.fit(X, y, weights)
        report(model, X, y, Q)


def main() raises:
    var lines = List[List[String]]()
    for line in open(argv()[1], "r").read().split("\n"):
        if line.byte_length() > 0:
            var tokens = List[String]()
            for token in line.split(" "):
                tokens.append(String(token))
            lines.append(tokens^)
    for at in range(0, len(lines), 5):
        ref h = lines[at]
        print("CASE", h[0])
        if h[1] == "1":
            if h[2] == "1":
                if h[5] == "1":
                    run_case[True, True, False](lines, at)
                else:
                    run_case[True, True, True](lines, at)
            elif h[5] == "1":
                run_case[True, False, False](lines, at)
            else:
                run_case[True, False, True](lines, at)
        elif h[2] == "1":
            run_case[False, True, False](lines, at)
        else:
            run_case[False, False, False](lines, at)
