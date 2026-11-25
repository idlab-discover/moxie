"""Validation-only bridge exercising the assembled public estimators."""
from std.sys import argv
import numojo as nm
from moxie.compose import ColumnTable
from moxie.svm import SVC, NuSVC, SVR, NuSVR, OneClassSVM, Gamma, ClassWeight, Probability
from moxie.warnings import FitWarning
from moxie.metadata_routing import SampleWeightRequest
from moxie.serialization import ModelSchema, load_model, save_model
from moxie.svm.kernel_serialization import SVCSchema, NuSVCSchema, SVRSchema, NuSVRSchema, OneClassSVMSchema


def restore[Schema: ModelSchema](model: Schema.Model, path: String) raises -> Schema.Model:
    save_model[Schema](model, path)
    return load_model[Schema](path)


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


def emit_warnings(records: List[FitWarning]):
    for record in records:
        print("WARNING", record.category, record.message)


def emit_request(method: String, request: SampleWeightRequest):
    var value = request.value()
    if value.isa[String]():
        print("REQUEST", method, value[String])
    elif value.isa[Bool]():
        print("REQUEST", method, "True" if value[Bool] else "False")
    else:
        print("REQUEST", method, "None")


def strided_float32(X: nm.NDArray[DType.float64]) raises -> nm.NDArray[DType.float32]:
    var width = X.shape[1]
    var storage = nm.zeros[DType.float32](nm.Shape(X.shape[0], width + 1))
    for i in range(X.shape[0]):
        for j in range(width):
            storage.unsafe_set(i * (width + 1) + j + 1, Float32(X.unsafe_get(i * width + j)))
    return storage.view_with_layout(X.shape, storage.strides, 1)


def emit_feature_names(names: Optional[List[String]]) raises:
    if not names:
        raise Error("Expected named fit to record feature names")
    for name in names.value():
        print("FEATURE", name)


def mixed_numeric_table(
    X: nm.NDArray[DType.float64], names: List[String],
) raises -> ColumnTable[]:
    var rows = X.shape[0]
    var width = X.shape[1]
    var matrix = nm.zeros[DType.float64](nm.Shape(rows, width - 1))
    var first = nm.zeros[DType.float32](nm.Shape(rows))
    var remaining_names = List[String](capacity=width - 1)
    for j in range(1, width):
        remaining_names.append(names[j].copy())
    for i in range(rows):
        first.unsafe_set(i, Float32(X.unsafe_get(i * width)))
        for j in range(1, width):
            matrix.unsafe_set(i * (width - 1) + j - 1, X.unsafe_get(i * width + j))
    var table = ColumnTable(matrix, remaining_names)
    table.add_numeric(names[0], first, position=0)
    return table^


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
    var probability_mode = Int(t[14])
    var probability = Probability("deprecated")
    if probability_mode >= 0:
        probability = Probability(probability_mode == 1)
    var limit = Int(t[15])
    var seed = Int(t[16])
    var mode = String(t[17])
    var decision_shape = String(t[18])
    var break_ties = Int(t[19]) != 0
    var input_kind = Int(t[20])
    var single = input_kind == 1
    var k = Int(t[21])
    var feature_names = List[String](capacity=p)
    for i in range(p):
        feature_names.append(String("column_", i))
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
    var training = mixed_numeric_table(x, feature_names) if input_kind == 2 else ColumnTable(x, feature_names)
    var testing = mixed_numeric_table(query, feature_names) if input_kind == 2 else ColumnTable(query, feature_names)
    if kind == 0:
        var model = SVC(kernel=kernel.copy(), gamma=gamma.copy(), degree=degree, coef0=coef0, tol=tol, shrinking=shrinking, max_iter=limit, C=c, probability=probability, random_state=seed, class_weight=costs.copy(), decision_function_shape=decision_shape.copy(), break_ties=break_ties)
        _ = model.set_fit_request(sample_weight="training_weight").set_score_request(
            sample_weight=False
        )
        if single:
            _ = model.fit(ColumnTable(strided_float32(x), feature_names), y, sample_weight=sw.copy())
        else:
            _ = model.fit(training, labels, sample_weight=sw.copy())
        if len(args) == 3:
            model = restore[SVCSchema[]](model, String(args[2]))
        var routing = model.get_metadata_routing()
        emit_request("fit", routing.fit)
        emit_request("score", routing.score)
        emit("support_", model.support_)
        emit("support_vectors_", model.support_vectors_)
        emit("n_support_", model.n_support_)
        emit("dual_coef_", model.dual_coef_)
        emit("intercept_", model.intercept_)
        emit_feature_names(model.feature_names_in_)
        emit_warnings(model.fit_warnings)
        print("SCALAR", "fit_status_", model.fit_status_)
        print("SCALAR", "n_features_in_", model.n_features_in_)
        emit("n_iter_", model.n_iter_)
        emit_list("classes_", model.classes_)
        emit("class_weight_", model.class_weight_)
        emit("decision", model.decision_function(testing))
        emit_list("prediction", model.predict(testing))
        print("SCALAR", "score", model.score(training, labels, sample_weight=sw.copy()))
        if probability_mode == 1:
            emit("probability", model.predict_proba(testing))
            emit("log_probability", model.predict_log_proba(testing))
            emit("probA_", model.probA_)
            emit("probB_", model.probB_)
        if kernel == "linear":
            emit("coef_", model.coef_())
    elif kind == 1:
        var model = NuSVC(kernel=kernel.copy(), gamma=gamma.copy(), degree=degree, coef0=coef0, tol=tol, shrinking=shrinking, max_iter=limit, nu=nu, probability=probability, random_state=seed, class_weight=costs.copy(), decision_function_shape=decision_shape.copy(), break_ties=break_ties)
        _ = model.set_fit_request(sample_weight="training_weight").set_score_request(
            sample_weight=False
        )
        if single:
            _ = model.fit(ColumnTable(strided_float32(x), feature_names), y, sample_weight=sw.copy())
        else:
            _ = model.fit(training, labels, sample_weight=sw.copy())
        if len(args) == 3:
            model = restore[NuSVCSchema[]](model, String(args[2]))
        var routing = model.get_metadata_routing()
        emit_request("fit", routing.fit)
        emit_request("score", routing.score)
        emit("support_", model.support_)
        emit("support_vectors_", model.support_vectors_)
        emit("n_support_", model.n_support_)
        emit("dual_coef_", model.dual_coef_)
        emit("intercept_", model.intercept_)
        emit_feature_names(model.feature_names_in_)
        emit_warnings(model.fit_warnings)
        print("SCALAR", "fit_status_", model.fit_status_)
        print("SCALAR", "n_features_in_", model.n_features_in_)
        emit("n_iter_", model.n_iter_)
        emit_list("classes_", model.classes_)
        emit("class_weight_", model.class_weight_)
        emit("decision", model.decision_function(testing))
        emit_list("prediction", model.predict(testing))
        print("SCALAR", "score", model.score(training, labels, sample_weight=sw.copy()))
        if probability_mode == 1:
            emit("probability", model.predict_proba(testing))
            emit("log_probability", model.predict_log_proba(testing))
            emit("probA_", model.probA_)
            emit("probB_", model.probB_)
        if kernel == "linear":
            emit("coef_", model.coef_())
    elif kind == 2:
        var model = OneClassSVM(kernel=kernel.copy(), gamma=gamma.copy(), degree=degree, coef0=coef0, tol=tol, shrinking=shrinking, max_iter=limit, nu=nu)
        _ = model.set_fit_request(sample_weight="training_weight")
        if single:
            _ = model.fit(ColumnTable(strided_float32(x), feature_names), sample_weight=sw.copy())
        else:
            _ = model.fit(training, sample_weight=sw.copy())
        if len(args) == 3:
            model = restore[OneClassSVMSchema](model, String(args[2]))
        emit_request("fit", model.get_metadata_routing())
        emit("support_", model.support_)
        emit("support_vectors_", model.support_vectors_)
        emit("n_support_", model.n_support_)
        emit("dual_coef_", model.dual_coef_)
        emit("intercept_", model.intercept_)
        emit_feature_names(model.feature_names_in_)
        emit_warnings(model.fit_warnings)
        print("SCALAR", "fit_status_", model.fit_status_)
        print("SCALAR", "n_features_in_", model.n_features_in_)
        print("ARRAY", "n_iter_", 1)
        print(model.n_iter_)
        emit("prediction", model.predict(testing))
        emit("decision", model.decision_function(testing))
        emit("score_samples", model.score_samples(testing))
        emit("offset_", model.offset_)
        if kernel == "linear":
            emit("coef_", model.coef_())
    elif kind == 3:
        var model = SVR(kernel=kernel.copy(), gamma=gamma.copy(), degree=degree, coef0=coef0, tol=tol, shrinking=shrinking, max_iter=limit, C=c, epsilon=epsilon)
        _ = model.set_fit_request(sample_weight="training_weight").set_score_request(
            sample_weight=False
        )
        if single:
            _ = model.fit(ColumnTable(strided_float32(x), feature_names), y, sample_weight=sw.copy())
        else:
            _ = model.fit(training, y, sample_weight=sw.copy())
        if len(args) == 3:
            model = restore[SVRSchema](model, String(args[2]))
        var routing = model.get_metadata_routing()
        emit_request("fit", routing.fit)
        emit_request("score", routing.score)
        emit("support_", model.support_)
        emit("support_vectors_", model.support_vectors_)
        emit("n_support_", model.n_support_)
        emit("dual_coef_", model.dual_coef_)
        emit("intercept_", model.intercept_)
        emit_feature_names(model.feature_names_in_)
        emit_warnings(model.fit_warnings)
        print("SCALAR", "fit_status_", model.fit_status_)
        print("SCALAR", "n_features_in_", model.n_features_in_)
        print("ARRAY", "n_iter_", 1)
        print(model.n_iter_)
        emit("prediction", model.predict(testing))
        emit("decision", model.decision_function(testing))
        print("SCALAR", "score", model.score(training, y, sample_weight=sw.copy()))
        if kernel == "linear":
            emit("coef_", model.coef_())
    elif kind == 4:
        var model = NuSVR(kernel=kernel.copy(), gamma=gamma.copy(), degree=degree, coef0=coef0, tol=tol, shrinking=shrinking, max_iter=limit, C=c, nu=nu)
        _ = model.set_fit_request(sample_weight="training_weight").set_score_request(
            sample_weight=False
        )
        if single:
            _ = model.fit(ColumnTable(strided_float32(x), feature_names), y, sample_weight=sw.copy())
        else:
            _ = model.fit(training, y, sample_weight=sw.copy())
        if len(args) == 3:
            model = restore[NuSVRSchema](model, String(args[2]))
        var routing = model.get_metadata_routing()
        emit_request("fit", routing.fit)
        emit_request("score", routing.score)
        emit("support_", model.support_)
        emit("support_vectors_", model.support_vectors_)
        emit("n_support_", model.n_support_)
        emit("dual_coef_", model.dual_coef_)
        emit("intercept_", model.intercept_)
        emit_feature_names(model.feature_names_in_)
        emit_warnings(model.fit_warnings)
        print("SCALAR", "fit_status_", model.fit_status_)
        print("SCALAR", "n_features_in_", model.n_features_in_)
        print("ARRAY", "n_iter_", 1)
        print(model.n_iter_)
        emit("prediction", model.predict(testing))
        emit("decision", model.decision_function(testing))
        print("SCALAR", "score", model.score(training, y, sample_weight=sw.copy()))
        if kernel == "linear":
            emit("coef_", model.coef_())
