import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from std.math import nan
from moxie.base import clone
from moxie.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor, BallTree, LocalOutlierFactor
from moxie.neighbors._contracts import DenseMetric
from callable_metric import WeightedL1

@fieldwise_init
struct InvalidMetric(DenseMetric):
    comptime dtype = DType.float64
    var mode: Int
    def validate_features(self, n_features: Int) raises:
        pass
    def distance(self, x: nm.NDArray[Self.dtype], i: Int,
                 y: nm.NDArray[Self.dtype], j: Int) raises -> Float64:
        if self.mode == 0: raise Error("custom metric failure")
        if self.mode == 1: return -1
        return nan[DType.float64]()

@fieldwise_init
struct NonzeroSelfDistance(DenseMetric):
    comptime dtype = DType.float64
    def validate_features(self, n_features: Int) raises:
        pass
    def distance(self, x: nm.NDArray[Self.dtype], i: Int,
                 y: nm.NDArray[Self.dtype], j: Int) raises -> Float64:
        return 7

def exercise[dtype: DType]() raises:
    comptime M = WeightedL1[dtype]
    var weights = nm.ones[DType.float64](nm.Shape(2))
    weights.unsafe_set(0, 3)
    var metric = M(weights.copy())
    var x = nm.zeros[dtype](nm.Shape(4, 2))
    x.unsafe_set(4, 1)
    x.unsafe_set(7, 2)
    var y = nm.zeros[dtype](nm.Shape(4))
    y.unsafe_set(1, 2)
    y.unsafe_set(2, 4)
    y.unsafe_set(3, 6)
    var model = NearestNeighbors[dtype, M](2, algorithm="brute", metric_params=metric.copy())
    with assert_raises(): _ = model.kneighbors()
    model.fit(x)
    var strided = x.view_with_layout(nm.Shape(4, 2), nm.NDArrayStrides(2, -1), 1)
    var strided_model = NearestNeighbors[dtype, M](2, algorithm="brute", metric_params=metric.copy())
    strided_model.fit(strided)
    assert_equal(strided_model.kneighbors().indices[1], 2)
    var strided_query = strided_model.kneighbors(strided.view())
    assert_equal(Float64(strided_query.distances.value().unsafe_get(0)), 0)
    metric.weights.unsafe_set(0, 20)
    var params = model.get_params()
    params.metric_params.value().weights.unsafe_set(0, 10)
    assert_equal(model.effective_metric_, "callable")
    assert_equal(model.effective_metric_params_.weights.unsafe_get(0), 3)
    var result = model.kneighbors()
    assert_equal(result.indices[0], 1)
    assert_equal(result.indices[1], 3)
    assert_equal(Float64(result.distances.value().unsafe_get(1)), 2)
    var radius = model.radius_neighbors(radius=Float64(0))
    assert_equal(len(radius.indices), 2)
    assert_equal(radius.indices[0], 1)
    assert_equal(radius.indices[1], 0)
    assert_equal(len(model.radius_neighbors(x.view(), radius=Float64(0)).indices), 6)
    assert_true(not model.kneighbors(return_distance=False).distances)
    with assert_raises(): _ = model.kneighbors(n_neighbors=4)
    with assert_raises(): _ = model.radius_neighbors(return_distance=False, sort_results=True)
    var bad_x = nm.zeros[dtype](nm.Shape(2, 3))
    with assert_raises(): model.fit(bad_x)
    assert_equal(model.n_samples_fit_, 4)
    with assert_raises(): _ = model.kneighbors(bad_x.view())
    model.effective_metric_params_.weights.unsafe_set(0, 0.1)
    assert_equal(model.kneighbors().indices[1], 3)
    var fresh = clone(model)
    assert_true(not fresh.__sklearn_is_fitted__())
    with assert_raises(): _ = fresh.kneighbors()
    fresh.fit(x)
    assert_equal(fresh.effective_metric_params_.weights.unsafe_get(0), 3)
    # Changes are configuration-only until refit, and survive cloning.
    params.metric_params = M(nm.full[DType.float64](nm.Shape(2), 0.25))
    model.set_params(params)
    assert_equal(model.kneighbors().indices[1], 3)
    fresh = clone(model)
    fresh.fit(x)
    assert_equal(fresh.kneighbors().indices[1], 2)
    model.fit(x)
    assert_equal(model.kneighbors().indices[1], 2)
    for algorithm in [String("auto"), String("kd_tree"), String("ball_tree")]:
        params.algorithm = algorithm
        model.set_params(params)
        if algorithm == "kd_tree":
            with assert_raises(): model.fit(x)
        else:
            model.fit(x)
        assert_equal(model.kneighbors().indices[1], 2)
    with assert_raises(): _ = NearestNeighbors[dtype, M](algorithm="brute")
    var automatic = NearestNeighbors[dtype,M](1,metric_params=metric.copy())
    automatic.fit(x)
    assert_equal(automatic._fit_method,"ball_tree")
    var tree = BallTree[dtype, M](x,1,metric_params=metric.copy())
    var brute = NearestNeighbors[dtype, M](2,algorithm="brute",metric_params=metric.copy())
    brute.fit(x)
    var direct = tree.query(x,2)
    var expected = brute.kneighbors(x.view())
    for i in range(len(direct.indices)):
        assert_equal(direct.distances.value().unsafe_get(i),expected.distances.value().unsafe_get(i))
    var lof = LocalOutlierFactor[dtype, M](2,algorithm="ball_tree",metric_params=metric.copy(),novelty=True)
    lof.fit(x)
    var lof_clone = clone(lof)
    assert_true(not lof_clone.__sklearn_is_fitted__())
    _ = lof.score_samples(x)
    var c = KNeighborsClassifier[dtype, M](2, algorithm="brute", metric_params=metric.copy())
    var r = KNeighborsRegressor[dtype, M](2, algorithm="brute", metric_params=metric.copy())
    var rc = RadiusNeighborsClassifier[dtype, M](2, algorithm="brute", metric_params=metric.copy())
    var rr = RadiusNeighborsRegressor[dtype, M](2, algorithm="brute", metric_params=metric.copy())
    c.fit(x,y)
    r.fit(x,y)
    rc.fit(x,y)
    rr.fit(x,y)
    var cc = clone(c)
    var cr = clone(r)
    var crc = clone(rc)
    var crr = clone(rr)
    assert_true(not cc.__sklearn_is_fitted__())
    assert_true(not cr.__sklearn_is_fitted__())
    assert_true(not crc.__sklearn_is_fitted__())
    assert_true(not crr.__sklearn_is_fitted__())
    cc.fit(x,y)
    cr.fit(x,y)
    crc.fit(x,y)
    crr.fit(x,y)
    assert_equal(cc.effective_metric_params_.weights.unsafe_get(0), 20)
    assert_equal(cr.effective_metric_params_.weights.unsafe_get(0), 20)
    assert_equal(crc.effective_metric_params_.weights.unsafe_get(0), 20)
    assert_equal(crr.effective_metric_params_.weights.unsafe_get(0), 20)
    assert_equal(Float64(cr.predict(x.view()).unsafe_get(0)), 1)
    assert_equal(Float64(cc.predict(x.view()).unsafe_get(0)), 0)
    assert_equal(crr.predict(x.view()).unsafe_get(0), (Float64(8)/3).cast[dtype]())
    assert_equal(Float64(crc.predict(x.view()).unsafe_get(0)), 0)
    var cc_params = cc.get_params()
    cc_params.neighbors.metric_params = M(nm.full[DType.float64](nm.Shape(2), 0.25))
    cc.set_params(cc_params)
    assert_equal(cc.effective_metric_params_.weights.unsafe_get(0), 20)
    cc.fit(x,y)
    assert_equal(cc.effective_metric_params_.weights.unsafe_get(0), 0.25)
    with assert_raises(): cc.fit(bad_x,y)
    assert_equal(cc.n_samples_fit_, 4)
    var cr_params = cr.get_params()
    cr_params.neighbors.metric_params = M(nm.full[DType.float64](nm.Shape(2), 0.25))
    cr.set_params(cr_params)
    assert_equal(cr.effective_metric_params_.weights.unsafe_get(0), 20)
    cr.fit(x,y)
    assert_equal(cr.effective_metric_params_.weights.unsafe_get(0), 0.25)
    with assert_raises(): cr.fit(bad_x,y)
    assert_equal(cr.n_samples_fit_, 4)
    var crc_params = crc.get_params()
    crc_params.neighbors.metric_params = M(nm.full[DType.float64](nm.Shape(2), 0.25))
    crc.set_params(crc_params)
    assert_equal(crc.effective_metric_params_.weights.unsafe_get(0), 20)
    crc.fit(x,y)
    assert_equal(crc.effective_metric_params_.weights.unsafe_get(0), 0.25)
    with assert_raises(): crc.fit(bad_x,y)
    assert_equal(crc.n_samples_fit_, 4)
    var crr_params = crr.get_params()
    crr_params.neighbors.metric_params = M(nm.full[DType.float64](nm.Shape(2), 0.25))
    crr.set_params(crr_params)
    assert_equal(crr.effective_metric_params_.weights.unsafe_get(0), 20)
    crr.fit(x,y)
    assert_equal(crr.effective_metric_params_.weights.unsafe_get(0), 0.25)
    with assert_raises(): crr.fit(bad_x,y)
    assert_equal(crr.n_samples_fit_, 4)

def main() raises:
    exercise[DType.float32]()
    exercise[DType.float64]()
    var x = nm.zeros[DType.float64](nm.Shape(2, 1))
    for mode in range(3):
        var model = NearestNeighbors[DType.float64, InvalidMetric](1, algorithm="brute", metric_params=InvalidMetric(mode))
        model.fit(x)
        with assert_raises(): _ = model.kneighbors(x.view())
        with assert_raises(): _ = model.radius_neighbors(x.view())
    var nonzero = NearestNeighbors[DType.float64, NonzeroSelfDistance](1, algorithm="brute", metric_params=NonzeroSelfDistance())
    nonzero.fit(x)
    assert_equal(nonzero.kneighbors().indices[0], 1)
    assert_equal(nonzero.kneighbors(x.view()).indices[0], 0)
    assert_equal(Float64(nonzero.kneighbors().distances.value().unsafe_get(0)), 7)
    print("PASS native callable metric contracts")
