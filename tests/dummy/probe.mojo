from std.sys import argv
from moxie.dummy import DummyRegressor, DummyClassifier, Dense, Label
from moxie._random import RandomState, seed_global_random_state
from numojo import NDArray, Shape, Item
import numojo as nm

def emit[dtype: DType](a: NDArray[dtype]) raises:
    print(32 if dtype == DType.float32 else 64)
    print(a.ndim)
    for d in range(a.ndim):
        print(a.shape[d])
    if a.ndim == 1:
        for i in range(a.shape[0]):
            print(a.item(i))
    else:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                print(a.item(i, j))

def label(value: String, kind: Int) raises -> Label:
    if kind == 2:
        return Label(value)
    if kind == 1:
        return Label(Float64(value))
    return Label(Int(value))

def emit_labels(y: List[List[Label]]) raises:
    for row in y:
        for v in row:
            if v.kind == 2:
                print(v.text)
            elif v.kind == 1:
                print(v.number)
            else:
                print(v.integer)

def emit_labels(y: List[Label]) raises:
    for v in y:
        if v.kind == 2:
            print(v.text)
        elif v.kind == 1:
            print(v.number)
        else:
            print(v.integer)


def main() raises:
    var mode = argv()[1]
    if mode == "reg":
        regressor_probe[DType.float64]()
    elif mode == "reg32":
        regressor_probe[DType.float32]()
    elif mode == "flat":
        classifier_probe[False, DType.float64]()
    elif mode == "flat32":
        classifier_probe[False, DType.float32]()
    elif mode == "clf32" or mode == "mutable32" or mode == "global32":
        classifier_probe[True, DType.float32]()
    else:
        classifier_probe[True, DType.float64]()


def regressor_probe[dtype: DType]() raises:
    var args = argv()
    var strategy = args[2]
    var seed = Int(args[3])
    var q = Float64(args[4])
    var n = Int(args[5])
    var outputs = Int(args[6])
    var count = Int(args[7])
    var kind = Int(args[8])
    var weighted = Int(args[9]) != 0
    var constants = Int(args[10]) != 0
    var rank = Int(args[11])
    var pos = 12
    var X = nm.zeros[dtype](Shape(n, 2))
    var test = nm.zeros[dtype](Shape(count, 5))
    var shape = Shape(n) if rank == 1 else Shape(n, outputs)
    var y = nm.zeros[dtype](shape)
    for i in range(n * outputs):
        y.unsafe_set(i, Scalar[dtype](Float64(args[pos])))
        pos += 1
    var weights = nm.ones[dtype](Shape(n))
    for i in range(n):
        weights[Item(i)] = Scalar[dtype](Float64(args[pos]))
        pos += 1
    var constant = Optional[NDArray[dtype]]()
    if constants:
        var values = nm.zeros[dtype](Shape(outputs))
        for j in range(outputs):
            values[Item(j)] = Scalar[dtype](Float64(args[pos]))
            pos += 1
        constant = values^
    var quantile = Optional[Float64]()
    if q >= 0:
        quantile = q
    var model = DummyRegressor[dtype](strategy=strategy, constant=constant, quantile=quantile)
    if weighted:
        model.fit(X, y, sample_weight=weights)
    else:
        model.fit(X, y)
    emit(model.constant_.value())
    var result = model.predict(test, return_std=True)
    emit(result.prediction)
    emit(result.std.value())
    if weighted:
        print(model.score(X, y, weights.copy()))
    else:
        print(model.score(X, y))


def classifier_probe[multioutput: Bool, dtype: DType]() raises:
    var args = argv()
    comptime Model = DummyClassifier[multioutput=multioutput, dtype=dtype]
    var mode = args[1]
    var strategy = args[2]
    var seed = Int(args[3])
    var q = Float64(args[4])
    var n = Int(args[5])
    var outputs = Int(args[6])
    var count = Int(args[7])
    var kind = Int(args[8])
    var weighted = Int(args[9]) != 0
    var constants = Int(args[10]) != 0
    var rank = Int(args[11])
    var pos = 12
    var X = nm.zeros[dtype](Shape(n, 2))
    var test = nm.zeros[dtype](Shape(count, 5))
    var y = List[List[Label]]()
    for i in range(n):
        var row = List[Label]()
        for j in range(outputs):
            row.append(label(args[pos], kind))
            pos += 1
        y.append(row^)
    var weights = nm.ones[dtype](Shape(n))
    for i in range(n):
        weights[Item(i)] = Scalar[dtype](Float64(args[pos]))
        pos += 1
    var constant = List[Label]()
    if constants:
        for j in range(outputs):
            constant.append(label(args[pos], kind))
            pos += 1
    var model = DummyClassifier[multioutput=multioutput, dtype=dtype](strategy=strategy, constant=constant.copy(), random_state=seed)
    if (mode == "global" or mode == "global32"):
        seed_global_random_state(seed)
        model.set_params(random_state=Optional[Optional[Int]](Optional[Int]()))
    var targets = Model.Target()
    comptime if multioutput:
        targets = rebind[Model.Target](y).copy()
    else:
        var flat = List[Label]()
        for row in y:
            flat.append(row[0].copy())
        targets = rebind[Model.Target](flat).copy()
    if weighted:
        model.fit(X, targets, sample_weight=weights)
    else:
        model.fit(X, targets)
    comptime if multioutput:
        for j in range(outputs):
            print(rebind[List[Int]](model.n_classes_)[j])
            emit_labels(rebind[List[List[Label]]](model.classes_)[j])
            emit(rebind[List[Dense]](model.class_prior_)[j])
    else:
        print(rebind[Int](model.n_classes_))
        emit_labels(rebind[List[Label]](model.classes_))
        emit(rebind[Dense](model.class_prior_))
    var rng = RandomState(seed)
    for iteration in range(2):
        if (mode == "global" or mode == "global32"):
            emit_prediction[multioutput, dtype](model.predict(test))
            var p = model.predict_proba(test)
            comptime if multioutput:
                for j in range(outputs):
                    emit(rebind[List[Dense]](p)[j])
            else:
                emit(rebind[Dense](p))
        elif (mode == "mutable" or mode == "mutable32"):
            emit_prediction[multioutput, dtype](model.predict(test, rng))
            var p = model.predict_proba(test, rng)
            comptime if multioutput:
                for j in range(outputs):
                    emit(rebind[List[Dense]](p)[j])
            else:
                emit(rebind[Dense](p))
        else:
            emit_prediction[multioutput, dtype](model.predict(test))
            var p = model.predict_proba(test)
            var lp = model.predict_log_proba(test)
            comptime if multioutput:
                for j in range(outputs):
                    emit(rebind[List[Dense]](p)[j])
                    emit(rebind[List[Dense]](lp)[j])
            else:
                emit(rebind[Dense](p))
                emit(rebind[Dense](lp))
    if outputs == 1 and kind != 1:
        if (mode == "mutable" or mode == "mutable32"):
            var optional_weights = Optional[NDArray[dtype]]()
            if weighted:
                optional_weights = weights.copy()
            print(model.score(X, targets, rng, optional_weights))
        elif weighted:
            print(model.score(X, targets, weights.copy()))
        else:
            print(model.score(X, targets))


def emit_prediction[multioutput: Bool, dtype: DType](y: DummyClassifier[multioutput, dtype].Prediction) raises:
    comptime if multioutput:
        emit_labels(rebind[List[List[Label]]](y))
    else:
        emit_labels(rebind[List[Label]](y))
