"""Fit once; freeze; call fit repeatedly without changing predictions."""
from moxie.base import clone
from moxie.frozen import FrozenEstimator
from dummy_regressor import DummyRegressor


def main() raises:
    var X: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var y: List[Float64] = [2.0, 3.0, 5.0, 10.0]
    var model = DummyRegressor()
    model.fit(X, y)
    var frozen = FrozenEstimator(model^)
    print("Before:", frozen.predict(X))
    var changed_y: List[Float64] = [100.0, 100.0, 100.0, 100.0]
    frozen.fit(X, changed_y)
    frozen.fit(X, changed_y)
    var cloned = clone(frozen)
    print("After repeated fit:", frozen.predict(X))
    print("Clone:", cloned.predict(X))
    print("Learned constant:", frozen.estimator().constant_.value())
