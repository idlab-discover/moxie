"""Native observations consumed by compare_reference.py."""
from moxie.base import clone
from moxie.frozen import FrozenEstimator
from dummy_regressor import DummyRegressor
from test_frozen_capabilities import ResponseProbe, TransformProbe, DensityProbe


def main() raises:
    var X: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    for scenario in range(5):
        var base = Float64(scenario - 2) * 8.0
        var y: List[Float64] = [base, base + 1, base + 3, base + 8]
        var model = DummyRegressor()
        _ = model.fit(X, y)
        var frozen = FrozenEstimator(model^)
        print(scenario, "before", frozen.predict(X))
        print(scenario, "score", frozen.score(X, y))
        var changed: List[Float64] = [100.0, 101.0, 103.0, 108.0]
        for _ in range(3):
            _ = frozen.fit(X, changed)
        _ = frozen.fit(List[Float64](), changed)
        _ = frozen.fit("ignored", 42)
        var cloned = clone(frozen)
        print(scenario, "after", frozen.predict(X))
        print(scenario, "clone", cloned.predict(X))
        print(scenario, "empty", frozen.predict(List[Float64]()))
        print(scenario, "changed_score", frozen.score(X, changed))
        var params = frozen.get_params()
        var replacement = DummyRegressor()
        _ = replacement.fit(X, changed)
        _ = frozen.set_params(estimator=replacement^)
        print(scenario, "replaced", frozen.predict(X))
        _ = frozen.set_params(params)
        print(scenario, "restored", frozen.predict(X))

    # Dispatch fixtures are deliberately synthetic, not classifier algorithms.
    var response = FrozenEstimator(ResponseProbe())
    _ = response.fit(1, 2, sample_weight=-100)
    print(5, "predict", clone(response).predict(2))
    print(5, "proba", response.predict_proba(3))
    print(5, "log_proba", response.predict_log_proba(3))
    print(5, "decision", response.decision_function(3))
    print(5, "score", response.score(3, 4))
    print(5, "weighted_score", response.score(3, 4, Optional[Int](5)))
    var transform = FrozenEstimator(TransformProbe())
    _ = transform.fit(100)
    print(6, "transform", transform.transform(3))
    print(6, "inverse", transform.inverse_transform(10))
    var density = FrozenEstimator(DensityProbe())
    _ = density.fit(100)
    print(7, "score", density.score(5))
    print(7, "target_score", density.score(5, Optional[String]("abc")))
    print(7, "samples", density.score_samples(5))
