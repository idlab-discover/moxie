import numojo as nm
from std.sys import argv
from moxie.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor, NeighborWeights
from moxie.neighbors._distance import Metric

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

def evaluate[dtype: DType](radius_mode: Bool, mode: String, multi: Bool, self_query: Bool, precomputed: Bool) raises:
    var args = argv()
    var algorithm = String(args[7]) if len(args)>7 else String("auto")
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
    var metric_name = String("euclidean")
    if precomputed:
        var m = Metric[dtype]("euclidean")
        var dx = nm.zeros[dtype](nm.Shape(7,7))
        var dq = nm.zeros[dtype](nm.Shape(3,7))
        for i in range(7):
            for j in range(7): dx.unsafe_set(i*7+j,m.distance(x,i,x,j).cast[dtype]())
        for i in range(3):
            for j in range(7): dq.unsafe_set(i*7+j,m.distance(q,i,x,j).cast[dtype]())
        x = dx^
        q = dq^
        metric_name = "precomputed"
    var query = Optional[nm.NDArray[dtype]](None)
    if not self_query: query = q.view()
    var weight_name = String("distance") if mode == "distance" else String("uniform")
    if radius_mode:
        var classifier = RadiusNeighborsClassifier[dtype](3.5,weights=weight_name,metric=metric_name,algorithm=algorithm,leaf_size=1,outlier_most_frequent=True)
        var regressor = RadiusNeighborsRegressor[dtype](3.5,weights=weight_name,metric=metric_name,algorithm=algorithm,leaf_size=1)
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
        var classifier = KNeighborsClassifier[dtype](3,weights=weight_name,metric=metric_name,algorithm=algorithm,leaf_size=1)
        var regressor = KNeighborsRegressor[dtype](3,weights=weight_name,metric=metric_name,algorithm=algorithm,leaf_size=1)
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
    if args[1] == "f32": evaluate[DType.float32](args[2]=="radius",args[3],args[4]=="multi",args[5]=="self",args[6]=="precomputed")
    else: evaluate[DType.float64](args[2]=="radius",args[3],args[4]=="multi",args[5]=="self",args[6]=="precomputed")
