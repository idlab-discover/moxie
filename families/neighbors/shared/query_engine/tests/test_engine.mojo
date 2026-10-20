from std.math import isnan
import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.neighbors._distance import Metric, at, metric_power
from moxie.neighbors._contracts import DenseMetric
from std.math import abs, sqrt
from moxie.neighbors._query_engine import BruteForce, distance_weights


def exercise[dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(4,2))
    x.unsafe_set(2,3)
    x.unsafe_set(3,4)
    x.unsafe_set(6,6)
    x.unsafe_set(7,8)
    var metric = Metric[dtype]("euclidean")
    var engine = BruteForce(metric,chunk_size=2)
    with assert_raises():
        _ = engine.kneighbors(None,1)
    engine.fit(x)
    var result = engine.kneighbors(None,2)
    assert_equal(result.indices[0],2)
    assert_equal(Float64(result.distances.value().unsafe_get(0)),0)
    assert_equal(Float64(result.distances.value().unsafe_get(1)),5)
    var explicit = engine.kneighbors(x.view(),1)
    assert_equal(explicit.indices[0],0)
    var r = engine.radius_neighbors(None,0)
    assert_equal(len(r.indices),2)
    assert_equal(r.indices[0],2)
    assert_equal(r.indices[1],0)
    var full = engine.radius_neighbors(x.view(),Float64("inf"),sort_results=True)
    assert_equal(len(full.indices),16)
    var indices_only = engine.kneighbors(None,1,False)
    assert_true(not indices_only.distances)
    var radius_only = engine.radius_neighbors(None,0,False)
    assert_true(not radius_only.distances)
    with assert_raises():
        _ = engine.kneighbors(None,4)
    with assert_raises():
        _ = engine.radius_neighbors(None,-1)
    with assert_raises():
        _ = engine.radius_neighbors(None,1,False,True)
    var clone = engine.clone()
    with assert_raises():
        _ = clone.kneighbors(None,1)
    x.unsafe_set(0,100)
    var independent = engine.kneighbors(None,1)
    assert_equal(independent.indices[0],2)
    var invalid = nm.zeros[dtype](nm.Shape(0,2))
    with assert_raises(): engine.fit(invalid)
    assert_equal(engine.n_samples_fit_,4)
    engine.fit(x)
    var refitted = engine.kneighbors(None,1)
    assert_true(refitted.indices[0] != 2)
    var pre = nm.zeros[dtype](nm.Shape(2,2))
    pre.unsafe_set(0,7)
    pre.unsafe_set(1,1)
    pre.unsafe_set(2,3)
    pre.unsafe_set(3,9)
    var pe = BruteForce(metric,precomputed=True)
    pe.fit(pre)
    var pr = pe.kneighbors(None,1)
    assert_equal(pr.indices[0],1)
    assert_equal(Float64(pr.distances.value().unsafe_get(1)),3)
    var ds = nm.zeros[dtype](nm.Shape(3))
    ds.unsafe_set(2,2)
    var weights = distance_weights(ds,[0,3])
    assert_equal(Float64(weights.unsafe_get(0)),1)
    assert_equal(Float64(weights.unsafe_get(2)),0)



@fieldwise_init
struct CallableMetric(DenseMetric):
    comptime dtype = DType.float64
    var invalid: Bool
    def validate_features(self, n_features: Int) raises:
        if n_features != 1: raise Error("ValueError: expected one feature")
    def distance(self, x: nm.NDArray[DType.float64], i: Int,
                 y: nm.NDArray[DType.float64], j: Int) raises -> Float64:
        if self.invalid: return -1
        return abs(at(x,i,0)-at(y,j,0))

def additional_contracts() raises:
    var backing = nm.zeros[DType.float64](nm.Shape(8,2))
    for i in range(8): backing.unsafe_set(i*2+1,Float64(i))
    var x = backing.view_with_layout(nm.Shape(4,1),nm.NDArrayStrides(4,1),1)
    var engine = BruteForce(CallableMetric(False),chunk_size=1)
    engine.fit(x)
    var got = engine.kneighbors(None,1)
    assert_equal(got.indices[0],1)
    assert_equal(Float64(got.distances.value().unsafe_get(0)),2)
    var invalid = BruteForce(CallableMetric(True))
    invalid.fit(x)
    with assert_raises(): _ = invalid.kneighbors(None,1)
    var q = nm.zeros[DType.float64](nm.Shape(1,1))
    q.unsafe_set(0,1)
    var metric = Metric[DType.float64]("minkowski",p=0.5)
    var fractional = BruteForce(metric)
    fractional.fit(x)
    var r = fractional.radius_neighbors(q.view(),1)
    assert_equal(len(r.indices),2)
    assert_true(abs(metric_power(Float64(2),0.5)-sqrt(Float64(2))) < 1e-15)
    assert_equal(metric_power(Float64(2),3),Float64(8))
    var bad_weights = nm.zeros[DType.float64](nm.Shape(2))
    bad_weights.unsafe_set(1,-1)
    var offsets: List[Int] = [0,2]
    with assert_raises(): _ = distance_weights(bad_weights,offsets)

def inferred_mahalanobis[dtype: DType]() raises:
    var tol = 1e-12 if dtype == DType.float64 else 1e-5
    var x = nm.zeros[dtype](nm.Shape(3,2))
    x.unsafe_set(2,1)
    x.unsafe_set(5,1)
    var engine = BruteForce(Metric[dtype]("mahalanobis"))
    engine.fit(x)
    # sklearn: VI = inv(cov(X.T)).T = [[4,2],[2,4]]; every off-diagonal distance is 2.
    var got = engine.kneighbors(None,2)
    for i in range(6): assert_true(abs(Float64(got.distances.value().unsafe_get(i))-2) < tol)
    assert_equal(got.indices[0],1)
    var r = engine.radius_neighbors(None,2.5,sort_results=True)
    assert_equal(len(r.indices),6)
    assert_equal(len(engine.radius_neighbors(None,1.5).indices),0)
    # Explicit queries never infer VI, even for identical data (sklearn `X is Y`).
    with assert_raises(contains="'VI' parameter is required"):
        _ = engine.kneighbors(x.view(),1)
    with assert_raises(contains="'VI' parameter is required"):
        _ = engine.radius_neighbors(x.view(),1)
    _ = engine.kneighbors(None,1)
    with assert_raises(contains="'VI' parameter is required"):
        _ = engine.kneighbors(x.view(),1)
    # Refit derives VI from the new data: scaling column 0 by 2 keeps distances.
    var scaled = x.copy()
    scaled.unsafe_set(2,2)
    engine.fit(scaled)
    var refit = engine.kneighbors(None,1)
    assert_true(abs(Float64(refit.distances.value().unsafe_get(0))-2) < tol)
    var clone = engine.clone()
    with assert_raises(): _ = clone.kneighbors(None,1)
    # Singular covariance: fit succeeds, the self query raises (sklearn LinAlgError).
    var singular = nm.zeros[dtype](nm.Shape(3,2))
    for i in range(3):
        singular.unsafe_set(2*i,Scalar[dtype](i))
        singular.unsafe_set(2*i+1,Scalar[dtype](i))
    var se = BruteForce(Metric[dtype]("mahalanobis"))
    se.fit(singular)
    with assert_raises(contains="cannot infer Mahalanobis VI"):
        _ = se.kneighbors(None,1)
    # One feature: numpy.cov yields a 0-d array and inversion fails in sklearn.
    var single = nm.zeros[dtype](nm.Shape(3,1))
    single.unsafe_set(1,1)
    single.unsafe_set(2,3)
    var fe = BruteForce(Metric[dtype]("mahalanobis"))
    fe.fit(single)
    with assert_raises(contains="at least two features"):
        _ = fe.kneighbors(None,1)
    # One fitted row: no non-self pairs, so a self radius query is empty.
    var one = BruteForce(Metric[dtype]("mahalanobis"))
    one.fit(x[0:1, :])
    assert_equal(len(one.radius_neighbors(None,10).indices),0)
    # Covariance-form V is not a brute parameter.
    var v = nm.ones[dtype](nm.Shape(2))
    var ve = BruteForce(Metric[dtype]("mahalanobis",V=v^))
    with assert_raises(contains="requires VI, not V"): ve.fit(x)


def inferred_seuclidean[dtype: DType]() raises:
    var tol = 1e-12 if dtype == DType.float64 else 1e-5
    var x = nm.zeros[dtype](nm.Shape(3,2))
    x.unsafe_set(2,1)
    x.unsafe_set(5,1)
    var engine = BruteForce(Metric[dtype]("seuclidean"))
    engine.fit(x)
    var result = engine.kneighbors(None,1)
    for i in range(3):
        assert_true(abs(Float64(result.distances.value().unsafe_get(i))-sqrt(Float64(3))) < tol)
    assert_equal(len(engine.radius_neighbors(None,2).indices),4)
    with assert_raises(contains="'V' parameter is required"):
        _ = engine.kneighbors(x.view(),1)
    with assert_raises(contains="'V' parameter is required"):
        _ = engine.radius_neighbors(x.view(),2)
    x.unsafe_set(2,4)
    engine.fit(x)
    result = engine.kneighbors(None,1)
    assert_true(abs(Float64(result.distances.value().unsafe_get(0))-sqrt(Float64(3))) < tol)
    var constant = nm.ones[dtype](nm.Shape(3,2))
    engine.fit(constant)
    var undefined = engine.kneighbors(None,1)
    assert_true(isnan(Float64(undefined.distances.value().unsafe_get(0))))
    assert_equal(len(engine.radius_neighbors(None,10).indices),0)
    var clone = engine.clone()
    with assert_raises(): _ = clone.kneighbors(None,1)
    var single = nm.zeros[dtype](nm.Shape(3,1))
    single.unsafe_set(1,1)
    single.unsafe_set(2,2)
    engine.fit(single)
    result = engine.kneighbors(None,1)
    assert_equal(Float64(result.distances.value().unsafe_get(0)),1)
    engine.fit(single[0:1,:])
    assert_equal(len(engine.radius_neighbors(None,2).indices),0)


def metric_edges() raises:
    var x = nm.zeros[DType.float64](nm.Shape(3,2))
    var bray = BruteForce(Metric[DType.float64]("braycurtis"))
    bray.fit(x)
    assert_equal(Float64(bray.kneighbors(None,1).distances.value().unsafe_get(0)),0)
    x.unsafe_set(0,1)
    x.unsafe_set(2,-1)
    var metric = Metric[DType.float64]("braycurtis")
    assert_equal(metric.distance(x,0,x,1),1)
    x.unsafe_set(0,-99)
    x.unsafe_set(1,2)
    x.unsafe_set(2,3)
    x.unsafe_set(3,4)
    var missing = Metric[DType.float64]("nan_euclidean",missing_values=-99)
    assert_true(abs(missing.distance(x,0,x,1)-sqrt(Float64(8))) < 1e-14)
    x.unsafe_set(1,-99)
    assert_true(isnan(missing.distance(x,0,x,1)))
    for name in [String("dice"),String("sokalsneath"),String("correlation")]:
        var engine = BruteForce(Metric[DType.float64](name))
        var zeros = nm.zeros[DType.float64](nm.Shape(3,2))
        engine.fit(zeros)
        assert_true(isnan(Float64(engine.kneighbors(None,1).distances.value().unsafe_get(0))))
        assert_equal(len(engine.radius_neighbors(None,1).indices),0)


def main() raises:
    inferred_seuclidean[DType.float32]()
    inferred_seuclidean[DType.float64]()
    metric_edges()
    additional_contracts()
    inferred_mahalanobis[DType.float32]()
    inferred_mahalanobis[DType.float64]()
    exercise[DType.float32]()
    exercise[DType.float64]()
    print("PASS dense query lifecycle, ties, radius, precomputed, weights and inferred Mahalanobis")
