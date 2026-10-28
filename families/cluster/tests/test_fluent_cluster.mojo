"""Live Moxie fluent mutation contracts preserve estimator identity."""
import numojo as nm
from std.testing import assert_equal, assert_true, assert_false
from moxie.base import clone
from moxie.cluster import (
    Clusterer, PredictiveClusterer, IncrementalClusterer,
    AffinityPropagation, KMeans, MiniBatchKMeans, BisectingKMeans, DBSCAN, MeanShift,
)

def fit_chain[E: Clusterer & Deinitable](mut model: E, x: E.Input, y: Optional[E.Target]) raises:
    _ = model.set_params(model.get_params()).fit(x, y).fit(x, y)
    assert_true(model.__sklearn_is_fitted__())
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())

def predict_chain[E: PredictiveClusterer](mut model: E, x: E.Input, y: Optional[E.Target]) raises -> E.Prediction:
    return model.set_params(model.get_params()).fit(x, y).predict(x)

def update_chain[E: IncrementalClusterer](mut model: E, x: E.Input, y: Optional[E.Target]) raises:
    _ = model.partial_fit(x, y).partial_fit(x, y)
    assert_true(model.__sklearn_is_fitted__())

def check[dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(6,1))
    for i in range(3,6): x[nm.Item(i,0)] = 4
    var ignored = Optional[nm.NDArray[dtype]](None)
    var affinity = AffinityPropagation[dtype](random_state=0)
    fit_chain(affinity,x,ignored)
    var kmeans = KMeans[dtype](2,random_state=0)
    fit_chain(kmeans,x,ignored)
    var predicted = predict_chain(kmeans,x,ignored)
    assert_equal(predicted.item(0),kmeans.labels_.item(0))
    var minibatch = MiniBatchKMeans[dtype](2,random_state=0)
    fit_chain(minibatch,x,ignored)
    update_chain(minibatch,x,ignored)
    var bisecting = BisectingKMeans[dtype](2,random_state=0)
    fit_chain(bisecting,x,ignored)
    var density = DBSCAN[dtype](1,min_samples=2)
    fit_chain(density,x,ignored)
    _ = density.set_params(min_samples=7).fit(x)
    assert_equal(density.labels_.item(0),Int64(-1))
    var shift = MeanShift[dtype](bandwidth=1.0)
    fit_chain(shift,x,ignored)
    predicted = predict_chain(shift,x,ignored)
    assert_equal(predicted.item(0),shift.labels_.item(0))
    _ = shift.set_params(max_iter=0).fit(x)
    assert_equal(shift.params.max_iter,0)
    assert_equal(shift.n_iter_,0)

def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: fluent cluster fit, set_params and partial_fit")
