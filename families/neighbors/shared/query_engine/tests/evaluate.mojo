import numojo as nm
from std.sys import argv
from moxie.neighbors._distance import Metric
from moxie.neighbors._query_engine import BruteForce


def evaluate[dtype: DType](name: String, p: Float64, weighted: Bool) raises:
    var cols = 2 if name == "haversine" else 5
    var x = nm.zeros[dtype](nm.Shape(9,cols))
    var q = nm.zeros[dtype](nm.Shape(3,cols))
    var binary = name == "jaccard" or name == "dice" or name == "rogerstanimoto" or name == "russellrao" or name == "sokalmichener" or name == "sokalsneath" or name == "yule"
    for i in range(9):
        for j in range(cols):
            var v = Float64((i*7+j*3+1)%13)/7
            if binary: v = Float64(1) if (i+j)%3 != 0 else Float64(0)
            x.unsafe_set(i*cols+j,v.cast[dtype]())
    for i in range(3):
        for j in range(cols):
            var v = Float64((i*11+j*5+2)%17)/9
            if binary: v = Float64(1) if (i+j)%2 != 0 else Float64(0)
            q.unsafe_set(i*cols+j,v.cast[dtype]())
    if name == "nan_euclidean":
        q.unsafe_set(0,Float64("nan").cast[dtype]())
        for j in range(cols): x.unsafe_set(8*cols+j,Float64("nan").cast[dtype]())
    var weights = Optional[nm.NDArray[dtype]](None)
    if weighted:
        var w = nm.zeros[dtype](nm.Shape(cols))
        for j in range(cols): w.unsafe_set(j,Float64(j).cast[dtype]())
        weights = w^
    var v = nm.zeros[dtype](nm.Shape(cols))
    var vi = nm.zeros[dtype](nm.Shape(cols,cols))
    for j in range(cols):
        v.unsafe_set(j,Float64(j+1).cast[dtype]())
        vi.unsafe_set(j*cols+j,(Float64(1)/Float64(j+1)).cast[dtype]())
    var metric = Metric[dtype](name,p,weights,V=v^,VI=vi^)
    metric.validate_features(cols)
    for i in range(3):
        for j in range(9): print("D",i,j,metric.distance(q,i,x,j))
    var engine = BruteForce(metric,chunk_size=2,allow_nan=name == "nan_euclidean")
    engine.fit(x)
    var k = engine.kneighbors(q.view(),4)
    for i in range(len(k.indices)): print("K",k.indices[i],k.distances.value().unsafe_get(i))
    var r = engine.radius_neighbors(q.view(),1.2,sort_results=True)
    for i in range(3):
        for j in range(r.offsets[i],r.offsets[i+1]): print("R",i,r.indices[j],r.distances.value().unsafe_get(j))
    var s = engine.kneighbors(None,3)
    for i in range(len(s.indices)): print("S",s.indices[i],s.distances.value().unsafe_get(i))


def inferred_fixture[dtype: DType](fixture: Int) raises -> nm.NDArray[dtype]:
    """Deterministic fixtures mirrored in parity.py; covariance is well conditioned."""
    var rows = 3 if fixture == 0 else (12 if fixture == 1 else 25)
    var cols = 2 if fixture == 0 else (4 if fixture == 1 else 3)
    var x = nm.zeros[dtype](nm.Shape(rows,cols))
    for i in range(rows):
        for j in range(cols):
            var v: Float64
            if fixture == 0: v = Float64(1) if i == j + 1 else Float64(0)
            elif fixture == 1: v = Float64((i*7+j*5+i*j*j+1)%13)/7
            else:
                var a = Float64((i*5+1)%11)/3
                var b = a*0.5 + Float64((i*3)%7)/5
                v = a if j == 0 else (b if j == 1 else Float64((i*i)%13)/4 - b*0.25)
            x.unsafe_set(i*cols+j,v.cast[dtype]())
    return x^


def evaluate_inferred[dtype: DType](fixture: Int, name: String) raises:
    """Self queries with parameters inferred from the fitted data."""
    var x = inferred_fixture[dtype](fixture)
    var engine = BruteForce(Metric[dtype](name),chunk_size=2)
    engine.fit(x)
    var k = min(3, x.shape[0]-1)
    var s = engine.kneighbors(None,k)
    for i in range(len(s.indices)): print("S",s.indices[i],s.distances.value().unsafe_get(i))
    var r = engine.radius_neighbors(None,1.5,sort_results=True)
    for i in range(x.shape[0]):
        for j in range(r.offsets[i],r.offsets[i+1]): print("T",i,r.indices[j],r.distances.value().unsafe_get(j))


def main() raises:
    var args = argv()
    if args[2] == "mahalanobis_inferred" or args[2] == "seuclidean_inferred":
        var name = "mahalanobis" if args[2] == "mahalanobis_inferred" else "seuclidean"
        if args[1] == "f32": evaluate_inferred[DType.float32](Int(args[3]),name)
        else: evaluate_inferred[DType.float64](Int(args[3]),name)
        return
    if args[1] == "f32": evaluate[DType.float32](args[2],Float64(args[3]),args[4] == "1")
    else: evaluate[DType.float64](args[2],Float64(args[3]),args[4] == "1")
