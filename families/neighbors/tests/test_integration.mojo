import numojo as nm
from std.testing import assert_equal, assert_true
from moxie.base import clone
from moxie.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor, NearestNeighbors, KernelDensity, LocalOutlierFactor
from moxie.neighbors._distance import Metric
from moxie.neighbors._query_engine import NeighborWeights
from moxie.neighbors._base import KNeighborsClassifierAPI, KNeighborsRegressorAPI, RadiusNeighborsClassifierAPI, RadiusNeighborsRegressorAPI, NeighborDensity, NoveltyDetector, KNeighborsQuery

@fieldwise_init
struct Smooth(NeighborWeights):
    var shift: Float64
    def apply(self, distances: nm.NDArray[DType.float64], offsets: List[Int]) raises -> nm.NDArray[DType.float64]:
        var result = distances.copy()
        for i in range(result.shape[0]): result.unsafe_set(i,1/(self.shift+result.unsafe_get(i)))
        return result^

def classifier_predict[T: KNeighborsClassifierAPI](model: T, x: T.Input) raises -> T.Prediction:
    return model.predict(x)

def fit_and_predict[T: KNeighborsClassifierAPI, params_origin: ImmOrigin](mut model: T, x: T.Input, y: T.Target, ref[params_origin] params: T.Params) raises -> T.Prediction:
    return model.set_params(params).fit(x,y).predict(x)

def density_samples[T: NeighborDensity](model: T, x: T.Input) raises -> T.SampleScores:
    return model.score_samples(x)

def main() raises:
    comptime assert conforms_to(KNeighborsClassifier, KNeighborsClassifierAPI)
    comptime assert conforms_to(KNeighborsRegressor, KNeighborsRegressorAPI)
    comptime assert conforms_to(RadiusNeighborsClassifier, RadiusNeighborsClassifierAPI)
    comptime assert conforms_to(RadiusNeighborsRegressor, RadiusNeighborsRegressorAPI)
    comptime assert conforms_to(NearestNeighbors, KNeighborsQuery)
    comptime assert conforms_to(LocalOutlierFactor, NoveltyDetector)
    comptime assert conforms_to(KernelDensity, NeighborDensity)
    var x = nm.zeros[DType.float64](nm.Shape(3,1))
    x.unsafe_set(1,2)
    x.unsafe_set(2,5)
    var y = nm.zeros[DType.float64](nm.Shape(3))
    y.unsafe_set(1,2)
    y.unsafe_set(2,5)
    var c = KNeighborsClassifier[DType.float64,Metric[DType.float64],nm.NDArray[DType.float64],Smooth](2,weighting=Smooth(1))
    var configuration = c.get_params()
    var chained = c.set_params(configuration).fit(x,y).predict(x)
    assert_equal(chained.unsafe_get(1),2)
    var explicit = c.predict_with_weights(x.copy(),Smooth(1))
    var stored = c.predict(x)
    for i in range(3): assert_equal(stored.unsafe_get(i),explicit.unsafe_get(i))
    assert_equal(classifier_predict(c,x).unsafe_get(1),2)
    assert_equal(fit_and_predict(c,x,y,configuration).unsafe_get(1),2)
    var copied = clone(c)
    assert_true(not copied.__sklearn_is_fitted__())
    assert_equal(copied.get_params().weighting.value().shift,1)
    var r = KNeighborsRegressor[DType.float64,Metric[DType.float64],Smooth](2,weighting=Smooth(1))
    r.fit(x,y)
    explicit = r.predict_with_weights(x.copy(),Smooth(1))
    stored = r.predict(x)
    for i in range(3): assert_equal(stored.unsafe_get(i),explicit.unsafe_get(i))
    var params = r.get_params()
    params.weighting = Smooth(10)
    r.set_params(params)
    explicit = r.predict_with_weights(x.copy(),Smooth(10))
    stored = r.predict(x)
    for i in range(3): assert_equal(stored.unsafe_get(i),explicit.unsafe_get(i))
    var rc = RadiusNeighborsClassifier[DType.float64,Metric[DType.float64],nm.NDArray[DType.float64],Smooth](3,weighting=Smooth(1))
    rc.fit(x,y)
    assert_equal(rc.score(x,y),1)
    var rr = RadiusNeighborsRegressor[DType.float64,Metric[DType.float64],Smooth](3,weighting=Smooth(1))
    rr.fit(x,y)
    explicit = rr.predict_with_weights(x.copy(),Smooth(1))
    stored = rr.predict(x)
    for i in range(3): assert_equal(stored.unsafe_get(i),explicit.unsafe_get(i))
    var kde = KernelDensity()
    kde.fit(x)
    assert_equal(density_samples(kde,x).unsafe_get(0),kde.score_samples(x).unsafe_get(0))
    print("PASS concrete capability dispatch and stored callable weights")
