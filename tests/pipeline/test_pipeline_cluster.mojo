"""Typed fit_predict with predictive and non-predictive clusterers."""
from moxie.base import ClusterMixin, Predictor, TransformerMixin, clone
from moxie.cluster import DBSCAN, KMeans
from moxie.pipeline import UnsupervisedPipeline, make_pipeline
from moxie.preprocessing import StandardScaler
from numojo import NDArray, Shape, zeros
from parity_io import ResultLine
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from unsupervised_fixtures import query_data, training_data


def fit_predict[C: ClusterMixin](
    mut pipe: C, X: C.Input, y: Optional[C.Target] = None,
) raises -> C.Labels:
    return pipe.fit_predict(X, y)


def check_dbscan() raises:
    comptime Estimator = UnsupervisedPipeline[StandardScaler[DType.float32], DBSCAN[DType.float32]]
    comptime assert not conforms_to(Estimator, Predictor)
    comptime assert not conforms_to(Estimator, TransformerMixin)
    var pipe = UnsupervisedPipeline(StandardScaler[DType.float32](), DBSCAN[DType.float32](eps=0.55, min_samples=2))
    var X = training_data[DType.float32]()
    assert_false(pipe.__sklearn_is_fitted__())
    with assert_raises():
        _ = fit_predict(pipe, zeros[DType.float32](Shape(0, 2)))
    assert_false(pipe.__sklearn_is_fitted__())
    var labels = fit_predict(pipe, X)
    assert_true(pipe.__sklearn_is_fitted__())
    assert_equal(labels.shape[0], X.shape[0])
    var line = ResultLine("dbscan")
    line.add("labels", labels)
    labels.unsafe_set(0, -99)
    assert_equal(pipe.estimator.labels_.unsafe_get(0), 0)

    var fresh = clone(pipe)
    assert_false(fresh.__sklearn_is_fitted__())
    var params = fresh.get_params()
    params.estimator.eps = 0.15
    _ = fresh.set_params(params).fit(X)
    line.add("refit", fresh.estimator.labels_)
    line.add("original", pipe.estimator.labels_)
    line.emit()


def check_kmeans() raises:
    var centers = zeros[DType.float64](Shape(2, 2))
    centers.unsafe_set(0, -1)
    centers.unsafe_set(1, -1)
    centers.unsafe_set(2, 1)
    centers.unsafe_set(3, 1)
    var pipe = make_pipeline(
        StandardScaler(), make_pipeline(StandardScaler(), KMeans(n_clusters=2, init=centers, n_init=1)),
    )
    var X = training_data[DType.float64]()
    var query = query_data[DType.float64]()
    assert_equal(pipe.estimator.estimator_name, "kmeans")
    with assert_raises(contains="NotFittedError"):
        _ = pipe.predict(query)
    var target = Optional[NDArray[DType.float64]](zeros[DType.float64](Shape(X.shape[0])))
    var labels = fit_predict(pipe, X, target)
    with assert_raises():
        _ = pipe.predict(zeros[DType.float64](Shape(2, 1)))
    var line = ResultLine("kmeans")
    line.add("labels", labels)
    line.add("predictions", pipe.predict(query))
    line.add("distances", pipe.transform(query))
    var fresh = clone(pipe)
    assert_false(fresh.__sklearn_is_fitted__())
    line.add("fit_transform", fresh.fit_transform(X))
    line.emit()


def main() raises:
    check_dbscan()
    check_kmeans()
