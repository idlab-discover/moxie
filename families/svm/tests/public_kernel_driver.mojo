"""Validation-only bridge exercising the assembled public estimators."""
from std.sys import argv
import numojo as nm
from moxie.svm import SVC, NuSVC, SVR, NuSVR, OneClassSVM, Gamma, ClassWeight


def emit[dtype: DType](name: String, a: nm.NDArray[dtype]) raises:
    if a.ndim == 1:
        print("ARRAY", name, a.shape[0])
    else:
        print("ARRAY", name, a.shape[0], a.shape[1])
    for i in range(a.size):
        print(a.unsafe_get(i))


def emit_list(name: String, a: List[Float64]) raises:
    print("LIST", name, len(a))
    for i in range(len(a)):
        print(a[i])


def main() raises:
    var args = argv()
    var contents: String
    with open(args[1], "r") as file:
        contents = file.read()
    var t = contents.split()
    var n = Int(t[0])
    var p = Int(t[1])
    var m = Int(t[2])
    var kind = Int(t[3])
    var kernel = String(t[4])
    var gamma = Gamma(String(t[5]))
    if gamma.mode == "value":
        gamma = Gamma(Float64(t[6]))
    var degree = Int(t[7])
    var coef0 = Float64(t[8])
    var c = Float64(t[9])
    var nu = Float64(t[10])
    var epsilon = Float64(t[11])
    var tol = Float64(t[12])
    var shrinking = Int(t[13]) != 0
    var probability = Int(t[14]) != 0
    var limit = Int(t[15])
    var seed = Int(t[16])
    var mode = String(t[17])
    var decision_shape = String(t[18])
    var break_ties = Int(t[19]) != 0
    var single = Int(t[20]) != 0
    var k = Int(t[21])
    var x = nm.zeros[DType.float64](nm.Shape(n, p))
    var y = nm.zeros[DType.float64](nm.Shape(n))
    var labels = List[Float64]()
    var sw = nm.zeros[DType.float64](nm.Shape(n))
    var query = nm.zeros[DType.float64](nm.Shape(m, p))
    var cursor = 22
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
    for i in range(m * p):
        query.unsafe_set(i, Float64(t[cursor]))
        cursor += 1
    var costs = ClassWeight[Float64](mode)
    if mode == "explicit":
        for _ in range(k):
            costs.labels.append(Float64(t[cursor]))
            costs.values.append(Float64(t[cursor + 1]))
            cursor += 2
    if kind == 0:
        var model = SVC(kernel=kernel.copy(), gamma=gamma.copy(), degree=degree, coef0=coef0, tol=tol, shrinking=shrinking, max_iter=limit, C=c, probability=probability, random_state=seed, class_weight=costs.copy(), decision_function_shape=decision_shape.copy(), break_ties=break_ties)
        if single:
            _ = model.fit(x.astype[DType.float32](), labels, sw.copy())
        else:
            _ = model.fit(x, labels, sw.copy())
        emit("support_", model.support_)
        emit("support_vectors_", model.support_vectors_)
        emit("n_support_", model.n_support_)
        emit("dual_coef_", model.dual_coef_)
        emit("intercept_", model.intercept_)
        print("SCALAR", "fit_status_", model.fit_status_)
        print("SCALAR", "n_features_in_", model.n_features_in_)
        emit("n_iter_", model.n_iter_)
        emit_list("classes_", model.classes_)
        emit("class_weight_", model.class_weight_)
        emit("decision", model.decision_function(query))
        emit_list("prediction", model.predict(query))
        print("SCALAR", "score", model.score(x, labels, sw.copy()))
        if probability:
            emit("probability", model.predict_proba(query))
            emit("log_probability", model.predict_log_proba(query))
            emit("probA_", model.probA_)
            emit("probB_", model.probB_)
        if kernel == "linear":
            emit("coef_", model.coef_())
    elif kind == 1:
        var model = NuSVC(kernel=kernel.copy(), gamma=gamma.copy(), degree=degree, coef0=coef0, tol=tol, shrinking=shrinking, max_iter=limit, nu=nu, probability=probability, random_state=seed, class_weight=costs.copy(), decision_function_shape=decision_shape.copy(), break_ties=break_ties)
        if single:
            _ = model.fit(x.astype[DType.float32](), labels, sw.copy())
        else:
            _ = model.fit(x, labels, sw.copy())
        emit("support_", model.support_)
        emit("support_vectors_", model.support_vectors_)
        emit("n_support_", model.n_support_)
        emit("dual_coef_", model.dual_coef_)
        emit("intercept_", model.intercept_)
        print("SCALAR", "fit_status_", model.fit_status_)
        print("SCALAR", "n_features_in_", model.n_features_in_)
        emit("n_iter_", model.n_iter_)
        emit_list("classes_", model.classes_)
        emit("class_weight_", model.class_weight_)
        emit("decision", model.decision_function(query))
        emit_list("prediction", model.predict(query))
        print("SCALAR", "score", model.score(x, labels, sw.copy()))
        if probability:
            emit("probability", model.predict_proba(query))
            emit("log_probability", model.predict_log_proba(query))
            emit("probA_", model.probA_)
            emit("probB_", model.probB_)
        if kernel == "linear":
            emit("coef_", model.coef_())
    elif kind == 2:
        var model = OneClassSVM(kernel=kernel.copy(), gamma=gamma.copy(), degree=degree, coef0=coef0, tol=tol, shrinking=shrinking, max_iter=limit, nu=nu)
        if single:
            _ = model.fit(x.astype[DType.float32](), sample_weight=sw.copy())
        else:
            _ = model.fit(x, sample_weight=sw.copy())
        emit("support_", model.support_)
        emit("support_vectors_", model.support_vectors_)
        emit("n_support_", model.n_support_)
        emit("dual_coef_", model.dual_coef_)
        emit("intercept_", model.intercept_)
        print("SCALAR", "fit_status_", model.fit_status_)
        print("SCALAR", "n_features_in_", model.n_features_in_)
        print("ARRAY", "n_iter_", 1)
        print(model.n_iter_)
        emit("prediction", model.predict(query))
        emit("decision", model.decision_function(query))
        emit("score_samples", model.score_samples(query))
        emit("offset_", model.offset_)
        if kernel == "linear":
            emit("coef_", model.coef_())
    elif kind == 3:
        var model = SVR(kernel=kernel.copy(), gamma=gamma.copy(), degree=degree, coef0=coef0, tol=tol, shrinking=shrinking, max_iter=limit, C=c, epsilon=epsilon)
        if single:
            _ = model.fit(x.astype[DType.float32](), y, sw.copy())
        else:
            _ = model.fit(x, y, sw.copy())
        emit("support_", model.support_)
        emit("support_vectors_", model.support_vectors_)
        emit("n_support_", model.n_support_)
        emit("dual_coef_", model.dual_coef_)
        emit("intercept_", model.intercept_)
        print("SCALAR", "fit_status_", model.fit_status_)
        print("SCALAR", "n_features_in_", model.n_features_in_)
        print("ARRAY", "n_iter_", 1)
        print(model.n_iter_)
        emit("prediction", model.predict(query))
        print("SCALAR", "score", model.score(x, y, sw.copy()))
        if kernel == "linear":
            emit("coef_", model.coef_())
    elif kind == 4:
        var model = NuSVR(kernel=kernel.copy(), gamma=gamma.copy(), degree=degree, coef0=coef0, tol=tol, shrinking=shrinking, max_iter=limit, C=c, nu=nu)
        if single:
            _ = model.fit(x.astype[DType.float32](), y, sw.copy())
        else:
            _ = model.fit(x, y, sw.copy())
        emit("support_", model.support_)
        emit("support_vectors_", model.support_vectors_)
        emit("n_support_", model.n_support_)
        emit("dual_coef_", model.dual_coef_)
        emit("intercept_", model.intercept_)
        print("SCALAR", "fit_status_", model.fit_status_)
        print("SCALAR", "n_features_in_", model.n_features_in_)
        print("ARRAY", "n_iter_", 1)
        print(model.n_iter_)
        emit("prediction", model.predict(query))
        print("SCALAR", "score", model.score(x, y, sw.copy()))
        if kernel == "linear":
            emit("coef_", model.coef_())
