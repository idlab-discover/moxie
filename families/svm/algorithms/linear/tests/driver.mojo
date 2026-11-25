"""Validation-only text bridge to public Mojo estimators."""
from std.sys import argv
from std.testing import assert_raises, assert_true
import numojo as nm
from moxie.base import clone
from moxie.compose import ColumnTable
from moxie.svm import LinearSVC, LinearSVR, Dual, ClassWeight
from moxie.svm._contracts import Dense
from moxie.serialization import dumps, loads
from moxie.svm.linear_serialization import LinearSVCSchema, LinearSVRSchema

def emit(a: Dense) raises:
    for i in range(a.size):
        print(a.unsafe_get(i))

def strided_float32(X: Dense) raises -> nm.NDArray[DType.float32]:
    var width = X.shape[1]
    var storage = nm.zeros[DType.float32](nm.Shape(X.shape[0], width + 1))
    for i in range(X.shape[0]):
        for j in range(width):
            storage.unsafe_set(i * (width + 1) + j + 1, Float32(X.unsafe_get(i * width + j)))
    return storage.view_with_layout(X.shape, storage.strides, 1)


def mixed_numeric_table[dtype: DType](
    X: Dense, names: List[String],
) raises -> ColumnTable[dtype]:
    # The mixed fixture has a real-valued first feature and a Boolean second feature.
    var rows = X.shape[0]
    var matrix = nm.zeros[dtype](nm.Shape(rows, 1))
    var flag = nm.zeros[DType.bool](nm.Shape(rows))
    for i in range(rows):
        matrix.unsafe_set(i, X.unsafe_get(2 * i).cast[dtype]())
        flag.unsafe_set(i, X.unsafe_get(2 * i + 1) != 0)
    var table = ColumnTable(matrix, [names[0].copy()])
    table.add_numeric(names[1], flag)
    return table^


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
    var named = len(args) > 2
    var table_input = named and args[2] != "named"
    var mixed_input = table_input and args[2] == "mixed"
    var names = List[String]()
    if named:
        for i in range(p):
            names.append(String("feature_") + String(i))
    var table = Optional[ColumnTable[]]()
    var table32 = Optional[ColumnTable[DType.float32]]()
    var reordered = Optional[ColumnTable[]]()
    var mixed = Optional[ColumnTable[]]()
    if table_input:
        table = mixed_numeric_table[DType.float64](x, names) if mixed_input else ColumnTable(x, names)
        var reversed_names: List[String] = [names[1].copy(), names[0].copy()]
        reordered = ColumnTable(x, reversed_names)
        var mixed_table = table.value().copy()
        mixed_table.add_strings("added", List[String](length=n, fill="text"))
        mixed = mixed_table^
        if single:
            table32 = mixed_numeric_table[DType.float32](x, names) if mixed_input else ColumnTable(strided_float32(x), names)
    if regression:
        var model = LinearSVR(loss=loss, dual=dual.copy(), C=c, epsilon=epsilon, tol=tol, fit_intercept=fit, intercept_scaling=scale, max_iter=limit, random_state=seed)
        # Requests govern parent routing only. Direct weighted calls still work.
        _ = model.set_fit_request(sample_weight=False).set_score_request(
            sample_weight="evaluation_weight"
        )
        if table_input:
            with assert_raises(contains="NotFittedError"):
                _ = model.predict(table.value())
            if single:
                _ = model.fit(table32.value(), y, sample_weight=sw.copy())
            else:
                _ = model.fit(table.value(), y, sample_weight=sw.copy())
            with assert_raises(contains="same order"):
                _ = model.predict(reordered.value())
            with assert_raises(contains="requires numeric storage"):
                _ = model.fit(mixed.value(), y)
            assert_true(model.feature_names_in_.value() == names)
            var cloned = clone(model)
            assert_true(not cloned.feature_names_in_)
            _ = cloned.fit(x, y)
            _ = cloned.predict(table.value())
            _ = cloned.fit(table.value(), y).fit(x, y)
            assert_true(not cloned.feature_names_in_)
        elif named:
            if single:
                _ = model.fit(
                    x.astype[DType.float32](), y, feature_names=names, sample_weight=sw.copy()
                )
            else:
                _ = model.fit(x, y, feature_names=names, sample_weight=sw.copy())
        elif single:
            _ = model.fit(x.astype[DType.float32](), y, sw.copy())
        else:
            _ = model.fit(x, y, sw.copy())
        model = LinearSVRSchema.decode(loads(dumps(LinearSVRSchema.encode(model))))
        if named:
            for name in model.feature_names_in_.value():
                print("FEATURE_NAME", name)
        for warning in model.fit_warnings:
            print("WARNING", warning.category, warning.message)
        print("RESULT", 1, p, model.n_iter_, model.fit_status_)
        emit(model.coef_)
        emit(model.intercept_)
        if table_input:
            if single:
                emit(model.predict(table32.value()))
                print(model.score(table32.value(), y, sw.copy()))
            else:
                emit(model.predict(table.value()))
                print(model.score(table.value(), y, sw.copy()))
        elif named:
            emit(model.predict(x, feature_names=names))
            print(model.score(x, y, feature_names=names, sample_weight=sw.copy()))
        else:
            emit(model.predict(x))
            print(model.score(x, y, sw.copy()))
    else:
        var model = LinearSVC(loss=loss, penalty=penalty, dual=dual.copy(), multi_class=multi, class_weight=costs.copy(), C=c, tol=tol, fit_intercept=fit, intercept_scaling=scale, max_iter=limit, random_state=seed)
        # Requests govern parent routing only. Direct weighted calls still work.
        _ = model.set_fit_request(sample_weight=False).set_score_request(
            sample_weight="evaluation_weight"
        )
        if table_input:
            with assert_raises(contains="NotFittedError"):
                _ = model.predict(table.value())
            if single:
                _ = model.fit(table32.value(), labels, sample_weight=sw.copy())
            else:
                _ = model.fit(table.value(), labels, sample_weight=sw.copy())
            with assert_raises(contains="same order"):
                _ = model.predict(reordered.value())
            with assert_raises(contains="requires numeric storage"):
                _ = model.fit(mixed.value(), labels)
            assert_true(model.feature_names_in_.value() == names)
            var cloned = clone(model)
            assert_true(not cloned.feature_names_in_)
            _ = cloned.fit(x, labels)
            _ = cloned.predict(table.value())
            _ = cloned.fit(table.value(), labels).fit(x, labels)
            assert_true(not cloned.feature_names_in_)
        elif named:
            if single:
                _ = model.fit(
                    x.astype[DType.float32](), labels, feature_names=names, sample_weight=sw.copy()
                )
            else:
                _ = model.fit(x, labels, feature_names=names, sample_weight=sw.copy())
        elif single:
            _ = model.fit(x.astype[DType.float32](), labels, sw.copy())
        else:
            _ = model.fit(x, labels, sw.copy())
        model = LinearSVCSchema[Float64].decode(loads(dumps(LinearSVCSchema[Float64].encode(model))))
        if named:
            for name in model.feature_names_in_.value():
                print("FEATURE_NAME", name)
        for warning in model.fit_warnings:
            print("WARNING", warning.category, warning.message)
        print("RESULT", model.coef_.shape[0], p, model.n_iter_, model.fit_status_)
        emit(model.coef_)
        emit(model.intercept_)
        if table_input:
            if single:
                emit(model.decision_function(table32.value()))
            else:
                emit(model.decision_function(table.value()))
        elif named:
            emit(model.decision_function(x, feature_names=names))
        else:
            emit(model.decision_function(x))
        var predicted: List[Float64]
        if table_input:
            predicted = model.predict(table32.value()) if single else model.predict(table.value())
        elif named:
            predicted = model.predict(x, feature_names=names)
        else:
            predicted = model.predict(x)
        for i in range(len(predicted)):
            print(predicted[i])
        if table_input:
            if single:
                print(model.score(table32.value(), labels, sw.copy()))
            else:
                print(model.score(table.value(), labels, sw.copy()))
        elif named:
            print(model.score(x, labels, feature_names=names, sample_weight=sw.copy()))
        else:
            print(model.score(x, labels, sw.copy()))
