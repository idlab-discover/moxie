import numojo as nm
from std.sys import argv
from moxie.neighbors import LocalOutlierFactor, NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor, NeighborWeights
from callable_metric import WeightedL1
from moxie.base import clone

@fieldwise_init
struct SmoothWeights(NeighborWeights):
    def apply(self, distances: nm.NDArray[DType.float64], offsets: List[Int]) raises -> nm.NDArray[DType.float64]:
        var result = distances.copy()
        for i in range(result.shape[0]): result.unsafe_set(i,1/(1+result.unsafe_get(i)))
        return result^

def emit[dtype: DType](prefix: String, x: nm.NDArray[dtype]) raises:
    var cols = 1 if x.ndim == 1 else x.shape[1]
    for i in range(x.shape[0]):
        for j in range(cols): print(prefix, i, j, x.unsafe_get(i*cols+j))

def evaluate[dtype: DType](radius_mode: Bool, mode: String, multi: Bool, self_query: Bool, algorithm: String) raises:
    var x = nm.zeros[dtype](nm.Shape(7,2))
    for i in range(2,7):
        x.unsafe_set(i*2,Float64(i).cast[dtype]())
        x.unsafe_set(i*2+1,Float64((i*i)%5).cast[dtype]())
    var q = nm.zeros[dtype](nm.Shape(3,2))
    q.unsafe_set(2,2.5)
    q.unsafe_set(3,1.5)
    q.unsafe_set(4,30)
    var cols = 2 if multi else 1
    var labels = nm.zeros[dtype](nm.Shape(7,cols) if multi else nm.Shape(7))
    var y = nm.zeros[dtype](nm.Shape(7,cols) if multi else nm.Shape(7))
    for i in range(7):
        labels.unsafe_set(i*cols,Float64(2+(i*2)%3*3).cast[dtype]())
        y.unsafe_set(i*cols,(Float64(i*i)/3).cast[dtype]())
        if multi:
            labels.unsafe_set(i*cols+1,Float64(-2+(i%2)*5).cast[dtype]())
            y.unsafe_set(i*cols+1,(Float64(i)/7).cast[dtype]())
    for col in range(cols):
        labels.unsafe_set(cols+col, labels.unsafe_get(col))
        y.unsafe_set(cols+col, y.unsafe_get(col))
    var metric = WeightedL1[dtype](nm.ones[DType.float64](nm.Shape(2)))
    metric.weights.unsafe_set(0, 1.75)
    metric.weights.unsafe_set(1, 0.625)
    var nn = NearestNeighbors[dtype, WeightedL1[dtype]](3, algorithm=algorithm, metric_params=metric.copy())
    nn.fit(x)
    var lof = LocalOutlierFactor[dtype,WeightedL1[dtype]](3,algorithm=algorithm,metric_params=metric.copy(),novelty=True)
    lof.fit(x)
    emit("L",lof.negative_outlier_factor_)
    emit("N",lof.score_samples(q))
    var query = Optional[nm.NDArray[dtype]](None)
    if not self_query: query = q.view()
    var top = nn.kneighbors(query)
    for i in range(len(top.indices)):
        print("K", i // top.k, top.indices[i], top.distances.value().unsafe_get(i))
    var rad = nn.radius_neighbors(query, radius=Float64(3.5), sort_results=True)
    for i in range(len(rad.offsets)-1):
        print("O", i, rad.offsets[i+1]-rad.offsets[i])
        for j in range(rad.offsets[i], rad.offsets[i+1]):
            print("D", i, rad.indices[j], rad.distances.value().unsafe_get(j))
    var weight_name = String("distance") if mode == "distance" else String("uniform")
    if radius_mode:
        var classifier = RadiusNeighborsClassifier[dtype, WeightedL1[dtype]](3.5,weights=weight_name,metric_params=metric.copy(),algorithm=algorithm,leaf_size=1,outlier_most_frequent=True)
        var regressor = RadiusNeighborsRegressor[dtype, WeightedL1[dtype]](3.5,weights=weight_name,metric_params=metric.copy(),algorithm=algorithm,leaf_size=1)
        classifier.fit(x,labels)
        regressor.fit(x,y)
        classifier = clone(classifier)
        regressor = clone(regressor)
        classifier.fit(x,labels)
        regressor.fit(x,y)
        var pred = classifier.predict(query)
        var probs = classifier.predict_proba(query)
        var values = regressor.predict(query)
        if mode == "callable":
            pred = classifier.predict_with_weights(query,SmoothWeights())
            probs = classifier.predict_proba_with_weights(query,SmoothWeights())
            values = regressor.predict_with_weights(query,SmoothWeights())
        emit("C",pred)
        emit("R",values)
        for col in range(len(probs)): emit("P"+String(col),probs[col])
    else:
        var classifier = KNeighborsClassifier[dtype, WeightedL1[dtype]](3,weights=weight_name,metric_params=metric.copy(),algorithm=algorithm,leaf_size=1)
        var regressor = KNeighborsRegressor[dtype, WeightedL1[dtype]](3,weights=weight_name,metric_params=metric.copy(),algorithm=algorithm,leaf_size=1)
        classifier.fit(x,labels)
        regressor.fit(x,y)
        classifier = clone(classifier)
        regressor = clone(regressor)
        classifier.fit(x,labels)
        regressor.fit(x,y)
        var pred = classifier.predict(query)
        var probs = classifier.predict_proba(query)
        var values = regressor.predict(query)
        if mode == "callable":
            pred = classifier.predict_with_weights(query,SmoothWeights())
            probs = classifier.predict_proba_with_weights(query,SmoothWeights())
            values = regressor.predict_with_weights(query,SmoothWeights())
        emit("C",pred)
        emit("R",values)
        for col in range(len(probs)): emit("P"+String(col),probs[col])

def main() raises:
    var args = argv()
    if args[1] == "f32": evaluate[DType.float32](args[2]=="radius",args[3],args[4]=="multi",args[5]=="self",args[6])
    else: evaluate[DType.float64](args[2]=="radius",args[3],args[4]=="multi",args[5]=="self",args[6])
