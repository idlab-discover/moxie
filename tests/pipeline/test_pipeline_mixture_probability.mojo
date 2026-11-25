"""Check typed probability forwarding through a scaled mixture and its refits."""
from std.testing import assert_equal, assert_raises
from moxie.base import ClassifierMixin, LogProbabilityPredictor, clone
from moxie.mixture import GaussianMixture
from moxie.pipeline import UnsupervisedPipeline, make_pipeline
from moxie.preprocessing import StandardScaler
from numojo import NDArray, Shape, ones, zeros
from parity_io import ResultLine
from test_pipeline_probability import through_trait
from unsupervised_fixtures import query_data, training_data


def check_mixture[dtype: DType](name: String) raises:
    comptime Scaler = StandardScaler[dtype]
    comptime Mixture = GaussianMixture[dtype]
    comptime Pipe = UnsupervisedPipeline[Scaler, Mixture]
    comptime assert not conforms_to(Pipe, LogProbabilityPredictor)
    comptime assert not conforms_to(Pipe, ClassifierMixin)

    var X = training_data[dtype]()
    var query = query_data[dtype]()
    query.itemset([2, 1], 0)
    var weights = ones[dtype](Shape(2))
    weights.itemset(0, 0.5)
    weights.itemset(1, 0.5)
    var means = zeros[dtype](Shape(2, 2))
    means.itemset([0, 0], -0.7)
    means.itemset([1, 0], 0.7)
    var pipe = make_pipeline(Scaler(), Mixture(
        n_components=2, covariance_type="diag", reg_covar=0.15, tol=1e-6,
        random_state=0, weights_init=weights^, means_init=means^,
        precisions_init=ones[dtype](Shape(2, 2)),
    ))
    with assert_raises(contains="NotFittedError"):
        _ = pipe.predict_proba(query)
    _ = pipe.fit(X)
    var probabilities = through_trait(pipe, query)
    comptime assert type_of(probabilities) == NDArray[dtype]
    assert_equal(probabilities.shape[0], query.shape[0])
    assert_equal(probabilities.shape[1], 2)
    with assert_raises(contains="ValueError"):
        _ = pipe.predict_proba(zeros[dtype](Shape(3, 1)))

    var line = ResultLine(name)
    line.add("fitted", probabilities)
    var nested = make_pipeline(Scaler(), clone(pipe))
    _ = nested.fit(X)
    line.add("nested", through_trait(nested, query))

    var fresh = clone(pipe)
    with assert_raises(contains="NotFittedError"):
        _ = fresh.predict_proba(query)
    var params = fresh.get_params()
    params.estimator.reg_covar = 0.35
    line.add("clone_refit", fresh.set_params(params).fit(X).predict_proba(query))
    line.add("original", pipe.predict_proba(query))
    line.add("refit", pipe.set_params(params).fit(X).predict_proba(query))
    line.emit()


def main() raises:
    check_mixture[DType.float32]("float32")
    check_mixture[DType.float64]("float64")
