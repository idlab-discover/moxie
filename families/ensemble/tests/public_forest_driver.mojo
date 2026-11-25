"""Read raw-bit fixtures and exercise public forest estimators."""
import numojo as nm
from std.sys import argv
from moxie.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier,
    RandomForestRegressor,
)
from moxie.ensemble._forest import Forest
from moxie.impute import SimpleImputer
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler
from moxie.tree._classes import ClassWeight, ClassWeightMap
from moxie.ensemble._records import Values
from forest_driver import floats, float_line, int_line


def report[classifier: Bool, extra: Bool, multi: Bool, dtype: DType](
    model: Forest[classifier, extra, multi, feature_dtype=dtype],
    X: nm.NDArray[dtype], y: Values, Q: nm.NDArray[dtype],
) raises:
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


def run_case[classifier: Bool, extra: Bool, multi: Bool, dtype: DType](
    var model: Forest[classifier, extra, multi, feature_dtype=dtype],
    lines: List[List[String]], at: Int,
) raises:
    ref h = lines[at]
    var n = atol(h[3])
    var p = atol(h[4])
    var outputs = atol(h[5])
    var q = atol(h[6])
    var X = floats[dtype](lines[at + 1], nm.Shape(n, p))
    var y = floats[DType.float64](
        lines[at + 2], nm.Shape(n) if outputs == 1 else nm.Shape(n, outputs)
    )
    var Q = floats[dtype](lines[at + 3], nm.Shape(q, p))
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
    _ = model.set_params(
        n_estimators=atol(h[7]), criterion=h[8], bootstrap=h[9] == "1",
        oob_score=h[10] == "1", random_state=atol(h[11]), class_weight=cw.copy(),
    )
    comptime if dtype == DType.float64 and not extra and not multi:
        if h[15] == "pipeline":
            comptime if classifier:
                var pipe = make_pipeline(StandardScaler[dtype](), model^)
                _ = pipe.fit(X, y)
                print(float_line("PIPE_PRED", pipe.predict(Q)))
                print("PIPE_SCORE", pipe.score(X, y))
                report(
                    pipe.estimator, pipe.transformer.transform(X), y,
                    pipe.transformer.transform(Q),
                )
            else:
                var pipe = make_pipeline(SimpleImputer[dtype](), model^)
                _ = pipe.fit(X, y)
                print(float_line("PIPE_PRED", pipe.predict(Q)))
                print("PIPE_SCORE", pipe.score(X, y))
                report(
                    pipe.estimator, pipe.transformer.transform(X), y,
                    pipe.transformer.transform(Q),
                )
            return
    _ = model.fit(X, y, weights)
    report(model, X, y, Q)
    if h[13] != "0":
        _ = model.set_params(warm_start=True, n_estimators=atol(h[13]))
        _ = model.fit(X, y, weights)
        report(model, X, y, Q)


def dispatch[dtype: DType](lines: List[List[String]], at: Int) raises:
    ref h = lines[at]
    if h[1] == "1":
        if h[2] == "1":
            if h[5] == "1":
                run_case(ExtraTreesClassifier[feature_dtype=dtype](), lines, at)
            else:
                run_case(ExtraTreesClassifier[multi_output=True, feature_dtype=dtype](), lines, at)
        elif h[5] == "1":
            run_case(RandomForestClassifier[feature_dtype=dtype](), lines, at)
        else:
            run_case(RandomForestClassifier[multi_output=True, feature_dtype=dtype](), lines, at)
    elif h[2] == "1":
        run_case(ExtraTreesRegressor[feature_dtype=dtype](), lines, at)
    else:
        run_case(RandomForestRegressor[feature_dtype=dtype](), lines, at)


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
        if h[14] == "64":
            dispatch[DType.float64](lines, at)
        else:
            dispatch[DType.float32](lines, at)
