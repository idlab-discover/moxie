"""Exercise outlier training labels through the generic Pipeline capability."""
from moxie.base import ClusterMixin, FitPredictor, clone
from moxie.neighbors import LocalOutlierFactor
from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler
from moxie.svm import OneClassSVM
from numojo import NDArray, Shape, ones, zeros
from parity_io import ResultLine
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from unsupervised_fixtures import query_data, training_data


def training_labels[E: FitPredictor](
    mut estimator: E, X: E.Input, y: Optional[E.Target] = None,
) raises -> E.Labels:
    return estimator.fit_predict(X, y)


def test_lof[dtype: DType](case_name: String) raises:
    var X = training_data[dtype]()
    var pipe = make_pipeline(
        StandardScaler[dtype](),
        LocalOutlierFactor[dtype](n_neighbors=2),
    )
    comptime assert not conforms_to(type_of(pipe), ClusterMixin)
    var labels: NDArray[dtype] = training_labels(pipe, X)
    assert_equal(labels.ndim, 1)
    assert_equal(labels.shape[0], X.shape[0])
    assert_true(pipe.__sklearn_is_fitted__())
    with assert_raises(contains="novelty=True"):
        _ = pipe.predict(X)

    var fresh = clone(pipe)
    assert_false(fresh.__sklearn_is_fitted__())
    with assert_raises():
        _ = training_labels(fresh, zeros[dtype](Shape(0, 2)))
    assert_false(fresh.__sklearn_is_fitted__())
    var params = fresh.get_params()
    params.estimator.neighbors.n_neighbors = 4
    params.estimator.contamination = 0.35
    _ = fresh.set_params(params)
    var y = zeros[dtype](Shape(X.shape[0]))
    var refitted = training_labels(fresh, X, y.copy())
    assert_equal(labels.unsafe_get(8), Scalar[dtype](-1))
    assert_equal(refitted.unsafe_get(8), Scalar[dtype](1))
    var line = ResultLine(case_name)
    line.add("labels", labels)
    line.add("refitted_labels", refitted)
    line.add("refit_same_instance", training_labels(fresh, X))
    line.add("original_scores", pipe.estimator.negative_outlier_factor_)
    line.emit()

    var novelty = clone(pipe)
    params.estimator.novelty = True
    _ = novelty.set_params(params)
    with assert_raises(contains="novelty=False"):
        _ = training_labels(novelty, X)
    assert_true(novelty.transformer.__sklearn_is_fitted__())
    assert_false(novelty.__sklearn_is_fitted__())


def test_one_class_svm() raises:
    var X = training_data[DType.float64]()
    var query = query_data[DType.float64]()
    var pipe = make_pipeline(
        StandardScaler(),
        make_pipeline(StandardScaler(), OneClassSVM(nu=0.6)),
    )
    comptime assert not conforms_to(type_of(pipe), ClusterMixin)
    with assert_raises(contains="NotFittedError"):
        _ = pipe.predict(query)
    var labels: NDArray[DType.int64] = training_labels(pipe, X)
    assert_equal(labels.ndim, 1)
    assert_equal(labels.shape[0], X.shape[0])
    assert_true(pipe.__sklearn_is_fitted__())

    var line = ResultLine("one_class_svm")
    line.add("labels", labels)
    line.add("prediction", pipe.predict(query))
    line.emit()

    var final = OneClassSVM(nu=0.6)
    var weights = ones[DType.float64](Shape(X.shape[0]))
    var keyword = final.fit_predict(X, sample_weight=weights.copy())
    var positional = final.fit_predict(X, None, weights^)
    for i in range(X.shape[0]):
        assert_equal(keyword.unsafe_get(i), positional.unsafe_get(i))


def main() raises:
    test_lof[DType.float32]("lof32")
    test_lof[DType.float64]("lof64")
    test_one_class_svm()
