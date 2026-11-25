"""Exercise the generic multioutput fit router with dense supervised SVMs."""

from std.testing import assert_equal, assert_raises

import numojo as nm

from moxie.base import Predictor, WeightedSupervisedEstimator
from moxie.metadata_routing import FitScoreMetadataRequestProvider
from moxie.multioutput import MultiOutputClassifier, MultiOutputRegressor
from moxie.svm import Dual, LinearSVC, LinearSVR, NuSVC, NuSVR, SVC, SVR
from moxie.svm._contracts import Dense


def check_requests[E: FitScoreMetadataRequestProvider](model: E) raises:
    var requests = model.get_metadata_routing()
    assert_equal(requests.fit.requested_name().or_else(""), "weights")
    assert_equal(requests.score.requested_name().or_else(""), "sample_weight")
    requests.fit.set(False)
    assert_equal(model.get_metadata_routing().fit.requested_name().or_else(""), "weights")


def classify[
    E: WeightedSupervisedEstimator & FitScoreMetadataRequestProvider & Predictor & Deinitable
](
    name: String, var child: E, X: Dense, y: nm.NDArray[DType.int64],
    metadata: Dict[String, Dense],
) raises:
    check_requests(child)
    var model = MultiOutputClassifier(child^)
    _ = model.fit(X, y, metadata=metadata)
    var prediction = model.predict(X)
    for i in range(prediction.size):
        print(name, prediction.unsafe_get(i))


def regress[
    E: WeightedSupervisedEstimator & FitScoreMetadataRequestProvider & Predictor & Deinitable
](
    name: String, var child: E, X: Dense, y: Dense, metadata: Dict[String, Dense],
) raises:
    check_requests(child)
    var model = MultiOutputRegressor(child^)
    _ = model.fit(X, y, metadata=metadata)
    var prediction = model.predict(X)
    for i in range(prediction.size):
        print(name, prediction.unsafe_get(i))


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(8, 2))
    var classification = nm.zeros[DType.int64](nm.Shape(8, 2))
    var regression = nm.zeros[DType.float64](nm.Shape(8, 2))
    var weights = nm.ones[DType.float64](nm.Shape(8))
    for i in range(8):
        X.unsafe_set(2 * i, Float64(i) - 3.5)
        X.unsafe_set(2 * i + 1, Float64(i % 3))
        classification.unsafe_set(2 * i, Int64(i >= 4))
        classification.unsafe_set(2 * i + 1, Int64(i % 2))
        regression.unsafe_set(2 * i, Float64(i % 3) - 0.5 * Float64(i))
        regression.unsafe_set(2 * i + 1, Float64(i % 2) + 0.25 * Float64(i))
    weights.unsafe_set(2, 8.0)
    weights.unsafe_set(5, 2.0)
    var metadata = Dict[String, Dense]()
    metadata["weights"] = weights.copy()

    var unset = MultiOutputRegressor(SVR(kernel="linear"))
    var standard = Dict[String, Dense]()
    standard["sample_weight"] = weights^
    with assert_raises(contains="UnsetMetadataPassedError"):
        _ = unset.fit(X, regression, metadata=standard)

    var svc = SVC[Int](kernel="linear", tol=1e-9)
    _ = svc.set_fit_request(sample_weight="weights").set_score_request(sample_weight=True)
    classify("SVC", svc^, X, classification, metadata)
    var nu_svc = NuSVC[Int](kernel="linear", nu=0.3, tol=1e-9)
    _ = nu_svc.set_fit_request(sample_weight="weights").set_score_request(sample_weight=True)
    classify("NuSVC", nu_svc^, X, classification, metadata)
    var linear_svc = LinearSVC[Int](dual=Dual(False), tol=1e-9, random_state=0)
    _ = linear_svc.set_fit_request(sample_weight="weights").set_score_request(sample_weight=True)
    classify("LinearSVC", linear_svc^, X, classification, metadata)

    var svr = SVR(kernel="linear", tol=1e-9)
    _ = svr.set_fit_request(sample_weight="weights").set_score_request(sample_weight=True)
    regress("SVR", svr^, X, regression, metadata)
    var nu_svr = NuSVR(kernel="rbf", tol=1e-9)
    _ = nu_svr.set_fit_request(sample_weight="weights").set_score_request(sample_weight=True)
    regress("NuSVR", nu_svr^, X, regression, metadata)
    var linear_svr = LinearSVR(
        loss="squared_epsilon_insensitive", dual=Dual(False), tol=1e-9, random_state=0,
    )
    _ = linear_svr.set_fit_request(sample_weight="weights").set_score_request(sample_weight=True)
    regress("LinearSVR", linear_svr^, X, regression, metadata)
