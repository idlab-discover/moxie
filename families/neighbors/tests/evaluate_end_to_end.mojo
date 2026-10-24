"""File-driven native bridge for end-to-end estimator parity; Python is only the test harness.

Usage: evaluate_end_to_end <f32|f64> <input file> <estimator> <options...>
The input file holds whitespace-separated `rows cols values...` matrices:
training X, training targets y (n, 1), queries Q and query targets yq (m, 1).
"""
import numojo as nm
from std.sys import argv
from moxie.neighbors import (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
    RadiusNeighborsClassifier, RadiusNeighborsRegressor, LocalOutlierFactor, KernelDensity, NearestCentroid)
from moxie.neighbors._contracts import TopKResult, RadiusResult

struct Reader(Movable):
    var tokens: List[String]
    var cursor: Int
    def __init__(out self, path: String) raises:
        self.tokens = List[String]()
        self.cursor = 0
        with open(path,"r") as file:
            var contents = file.read()
            for token in contents.split(): self.tokens.append(String(token))
    def number(mut self) raises -> Float64:
        var value = Float64(self.tokens[self.cursor])
        self.cursor += 1
        return value
    def matrix[dtype: DType](mut self) raises -> nm.NDArray[dtype]:
        var rows = Int(self.number())
        var cols = Int(self.number())
        var result = nm.zeros[dtype](nm.Shape(rows,cols))
        for i in range(rows*cols): result.unsafe_set(i,self.number().cast[dtype]())
        return result^
    def vector[dtype: DType](mut self) raises -> nm.NDArray[dtype]:
        var rows = Int(self.number())
        _ = self.number()
        var result = nm.zeros[dtype](nm.Shape(rows))
        for i in range(rows): result.unsafe_set(i,self.number().cast[dtype]())
        return result^

def emit[dtype: DType](key: String, values: nm.NDArray[dtype]) raises:
    for i in range(values.size): print(key, i, values.unsafe_get(i))

def emit_topk[dtype: DType](key: String, result: TopKResult[dtype]) raises:
    for i in range(len(result.indices)): print(key+"I", i, result.indices[i])
    emit(key+"D", result.distances.value())

def emit_radius[dtype: DType](key: String, result: RadiusResult[dtype]) raises:
    for i in range(len(result.offsets)): print(key+"O", i, result.offsets[i])
    for i in range(len(result.indices)): print(key+"I", i, result.indices[i])
    emit(key+"D", result.distances.value())

def run[dtype: DType](args: List[String]) raises:
    var input = Reader(args[2])
    var X = input.matrix[dtype]()
    var y = input.vector[dtype]()
    var Q = input.matrix[dtype]()
    var yq = input.vector[dtype]()
    var kind = args[3]
    var algorithm = args[4]
    var metric = args[5]
    var p = Float64(args[6])
    var weights = args[7]
    var k = Int(args[8])
    var radius = Float64(args[9])
    if kind == "nn":
        var model = NearestNeighbors[dtype](k, radius, algorithm=algorithm, metric=metric, p=p, leaf_size=3)
        _ = model.fit(X)
        emit_topk("K", model.kneighbors(Q.copy()))
        emit_topk("S", model.kneighbors())
        emit_radius("R", model.radius_neighbors(Q.copy(), sort_results=True))
        print("M 0", model.effective_metric_)
    elif kind == "knnc":
        var model = KNeighborsClassifier[dtype](k, weights=weights, algorithm=algorithm, metric=metric, p=p, leaf_size=3)
        _ = model.fit(X, y)
        emit("C", model.predict(Q))
        emit("P", model.predict_proba(Q)[0])
        emit("C0", model.classes_[0])
        print("A 0", model.score(Q, yq))
    elif kind == "knnr":
        var model = KNeighborsRegressor[dtype](k, weights=weights, algorithm=algorithm, metric=metric, p=p, leaf_size=3)
        _ = model.fit(X, y)
        emit("V", model.predict(Q))
        print("A 0", model.score(Q, yq))
    elif kind == "rnc":
        var model = RadiusNeighborsClassifier[dtype](radius, weights=weights, algorithm=algorithm, metric=metric, p=p, leaf_size=3, outlier_most_frequent=True)
        _ = model.fit(X, y)
        emit("C", model.predict(Q))
        emit("P", model.predict_proba(Q)[0])
        print("A 0", model.score(Q, yq))
    elif kind == "rnr":
        var model = RadiusNeighborsRegressor[dtype](radius, weights=weights, algorithm=algorithm, metric=metric, p=p, leaf_size=3)
        _ = model.fit(X, y)
        emit("V", model.predict(Q))
    elif kind == "lof":
        var contamination = Optional[Float64](None)
        if weights != "auto": contamination = Float64(weights)
        var outliers = LocalOutlierFactor[dtype](k, algorithm=algorithm, metric=metric, p=p, leaf_size=3, contamination=contamination)
        emit("L", outliers.fit_predict(X))
        emit("F", outliers.negative_outlier_factor_)
        print("O 0", outliers.offset_)
        print("N 0", outliers.n_neighbors_)
        var novelty = LocalOutlierFactor[dtype](k, algorithm=algorithm, metric=metric, p=p, leaf_size=3, contamination=contamination, novelty=True)
        _ = novelty.fit(X)
        emit("SS", novelty.score_samples(Q))
        emit("DF", novelty.decision_function(Q))
        emit("PR", novelty.predict(Q))
    elif kind == "kde":
        # weights carries the kernel, radius the bandwidth.
        var model = KernelDensity[dtype](bandwidth=radius, algorithm=algorithm, kernel=weights, metric=metric)
        _ = model.fit(X)
        emit("SS", model.score_samples(Q))
        print("T 0", model.score(Q))
    elif kind == "nc":
        var shrink = Optional[Float64](None)
        if radius > 0: shrink = radius
        var model = NearestCentroid[dtype](metric, shrink_threshold=shrink, priors=weights)
        _ = model.fit(X, y)
        emit("C", model.predict(Q))
        emit("CE", model.centroids_)
        if metric == "euclidean":
            emit("P", model.predict_proba(Q))
            emit("DF", model.decision_function(Q))
        print("A 0", model.score(Q, yq))
    else:
        raise Error("unknown estimator " + kind)

def main() raises:
    var args = argv()
    var owned = List[String]()
    for arg in args: owned.append(String(arg))
    if len(owned) != 10:
        raise Error("usage: evaluate_end_to_end <f32|f64> <file> <estimator> <algorithm> <metric> <p> <weights> <k> <radius>")
    if owned[1] == "f32":
        run[DType.float32](owned)
    elif owned[1] == "f64":
        run[DType.float64](owned)
    else:
        raise Error("dtype must be f32 or f64")
