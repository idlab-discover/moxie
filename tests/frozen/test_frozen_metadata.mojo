"""Sample-weight requests through a frozen production estimator."""

from std.testing import assert_equal, assert_false, assert_raises, assert_true

from moxie.base import clone
from moxie.frozen import FrozenEstimator
from moxie.isotonic import IsotonicRegression
from moxie.metadata_routing import FitScoreMetadataRequestProvider
from moxie.preprocessing import StandardScaler
from numojo import NDArray, array


def consume_requests[M: FitScoreMetadataRequestProvider](model: M):
    var requests = model.get_metadata_routing()
    print("fit_alias", requests.fit.requested_name().or_else(""))
    print("score_alias", requests.score.requested_name().or_else(""))


def main() raises:
    comptime Model = IsotonicRegression[]
    comptime Array = NDArray[DType.float64]
    comptime assert not conforms_to(
        FrozenEstimator[StandardScaler[]], FitScoreMetadataRequestProvider
    )

    var model = Model()
    var default_frozen = FrozenEstimator(Model())
    var defaults = default_frozen.get_metadata_routing()
    assert_true(defaults.fit.value().isa[NoneType]())
    assert_true(defaults.score.value().isa[NoneType]())

    var X = array[DType.float64]([0.0, 1.0, 2.0], [3])
    var y = X.copy()
    var changed = array[DType.float64]([0.0, 9.0, 2.0], [3])
    var weights = array[DType.float64]([1.0, 0.0, 1.0], [3])
    _ = model.set_fit_request(sample_weight="train_weights").set_score_request(
        sample_weight="score_weights"
    )
    var frozen = FrozenEstimator(model^)
    var snapshot = frozen.get_metadata_routing()
    snapshot.score.set(False)
    assert_false(snapshot.score.value()[Bool])
    assert_equal(
        frozen.get_metadata_routing().score.requested_name().or_else(""),
        "score_weights",
    )
    var metadata = Dict[String, Array]()
    metadata["sample_weight"] = weights.copy()
    with assert_raises(contains="UnsetMetadataPassedError"):
        _ = defaults.fit.route(metadata)
    assert_false(Bool(snapshot.score.route(metadata)))
    metadata["train_weights"] = array[DType.float64]([1.0, 1.0, 1.0], [3])
    metadata["score_weights"] = weights.copy()
    var train_weights = snapshot.fit.route(metadata)
    assert_true(Bool(train_weights))
    with assert_raises(contains="NotFittedError"):
        _ = frozen.fit(X, y, sample_weight=train_weights.value())

    var fitted = Model()
    _ = fitted.set_fit_request(sample_weight="train_weights").set_score_request(
        sample_weight="score_weights"
    ).fit(X, y)
    _ = frozen.set_params(estimator=fitted^)
    var nested = FrozenEstimator(clone(frozen))
    var score_weights = nested.get_metadata_routing().score.route(metadata)
    assert_true(Bool(score_weights))
    _ = frozen.fit(X, changed, sample_weight=train_weights.value())
    var prediction = nested.predict(X).tolist()[1]
    assert_equal(prediction, 1.0)
    assert_true(nested.score(X, changed) < 1.0)
    var weighted_score = nested.score(X, changed, sample_weight=score_weights^)
    assert_equal(weighted_score, 1.0)
    _ = frozen.set_params(estimator=Model())
    assert_true(frozen.get_metadata_routing().fit.value().isa[NoneType]())
    consume_requests(nested)
    print("prediction", prediction)
    print("weighted_score", weighted_score)
