"""Covariance finals expose their target-free composition capabilities."""
from moxie.base import ClusterMixin, DensityMixin, FitPredictor, UnsupervisedScorer, clone
from moxie.covariance import EllipticEnvelope, LedoitWolf
from moxie.pipeline import UnsupervisedPipeline, make_pipeline
from moxie.preprocessing import StandardScaler
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from unsupervised_fixtures import query_data, training_data


def fit_labels[E: FitPredictor](
    mut estimator: E, X: E.Input, y: Optional[E.Target],
) raises -> E.Labels:
    return estimator.fit_predict(X, y)


def score[E: UnsupervisedScorer](
    estimator: E, X: E.Input, y: Optional[E.Target],
) raises -> Float64:
    return estimator.score(X, y)


def main() raises:
    comptime Scaler = StandardScaler[DType.float32]
    comptime Detector = EllipticEnvelope[DType.float32]
    comptime Covariance = LedoitWolf[DType.float32]
    comptime assert not conforms_to(UnsupervisedPipeline[Scaler, Detector], ClusterMixin)
    comptime assert not conforms_to(UnsupervisedPipeline[Scaler, Covariance], DensityMixin)

    var X = training_data[DType.float32]()
    var query = query_data[DType.float32]()
    var outliers = make_pipeline(Scaler(), Detector(contamination=0.2, random_state=0))
    with assert_raises(contains="NotFittedError"):
        _ = outliers.decision_function(query)
    var labels = fit_labels(outliers, X, None)
    assert_equal(labels.size, X.shape[0])
    assert_equal(outliers.predict(query).size, query.shape[0])
    assert_equal(outliers.score_samples(query).size, query.shape[0])

    var fresh = clone(outliers)
    assert_false(fresh.__sklearn_is_fitted__())
    var params = fresh.get_params()
    assert_equal(params.transformer_name, "standardscaler")
    assert_equal(params.estimator_name, "ellipticenvelope")
    params.estimator.contamination = 0.3
    _ = fresh.set_params(params).fit(X, labels^)
    assert_true(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.get_params().estimator.contamination, 0.3)
    assert_equal(outliers.get_params().estimator.contamination, 0.2)

    var covariance = make_pipeline(Scaler(), Covariance())
    with assert_raises(contains="NotFittedError"):
        _ = score(covariance, query, None)
    _ = covariance.fit(X)
    var scaled = covariance.transformer.transform(query)
    assert_equal(score(covariance, query, None), covariance.estimator.score(scaled))
