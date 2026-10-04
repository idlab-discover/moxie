"""Frozen ownership, parameter and estimator behavior contracts."""
from std.testing import assert_equal, assert_false, assert_true, assert_raises
from moxie.base import (
    clone,
    RegressorMixin,
    ClassifierMixin,
    Predictor,
    FitTransformer,
    ClusterMixin,
    TransformerMixin,
    IncrementalRegressor,
)
from moxie.frozen import FrozenEstimator
from moxie.utils.validation import check_is_fitted
from dummy_regressor import DummyRegressor, DummyParams


def generic_prediction[
    E: RegressorMixin
](model: E, X: E.Input) raises -> E.Prediction:
    return model.predict(X)


def fitted_clone() raises -> FrozenEstimator[DummyRegressor]:
    var model = DummyRegressor()
    var X: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var y: List[Float64] = [2.0, 3.0, 5.0, 10.0]
    _ = model.fit(X, y)
    var frozen = FrozenEstimator(model^)
    return clone(frozen)


def main() raises:
    comptime F = FrozenEstimator[DummyRegressor]
    comptime assert conforms_to(F, RegressorMixin)
    comptime assert conforms_to(F, Predictor)
    comptime assert not conforms_to(F, ClassifierMixin)
    comptime assert not conforms_to(F, TransformerMixin)
    comptime assert not conforms_to(F, FitTransformer)
    comptime assert not conforms_to(F, ClusterMixin)
    comptime assert not conforms_to(F, IncrementalRegressor)

    var X: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var y: List[Float64] = [2.0, 3.0, 5.0, 10.0]
    var unfitted = F(DummyRegressor())
    assert_false(unfitted.__sklearn_is_fitted__())
    assert_false(clone(unfitted).__sklearn_is_fitted__())
    with assert_raises():
        _ = unfitted.fit(X, y)
    with assert_raises():
        check_is_fitted(unfitted)
    with assert_raises():
        _ = unfitted.predict(X)
    assert_false(unfitted.__sklearn_is_fitted__())

    var frozen = fitted_clone()  # original model and wrapper are already gone
    check_is_fitted(frozen)
    var expected: List[Float64] = [5.0, 5.0, 5.0, 5.0]
    assert_equal(generic_prediction(frozen, X), expected)
    assert_equal(frozen.score(X, y), 0.0)
    assert_equal(frozen.estimator().constant_.value(), 5.0)
    var normal_clone = clone(frozen.estimator())
    assert_false(normal_clone.__sklearn_is_fitted__())

    var changed: List[Float64] = [100.0, 200.0, 300.0, 400.0]
    var source = DummyRegressor()
    _ = source.fit(X, y)
    var from_copy = F(source.copy())
    _ = source.fit(X, changed)
    assert_equal(from_copy.predict(X), expected)
    with assert_raises():
        _ = source.fit(X, List[Float64]())
    assert_equal(source.constant_.value(), 250.0)
    var unsupported = DummyRegressor(strategy="median")
    with assert_raises():
        _ = unsupported.fit(X, y)
    assert_false(unsupported.__sklearn_is_fitted__())
    for _ in range(3):
        _ = frozen.fit(X, changed)
    var empty = List[Float64]()
    _ = frozen.fit(empty, changed)  # no shape validation in the reference
    _ = frozen.fit("ignored", 42)  # no input type validation either
    assert_equal(frozen.predict(X), expected)
    assert_equal(len(frozen.predict(empty)), 0)

    var copied = frozen.copy()
    var cloned = clone(frozen)
    var deep = frozen.get_params()
    var shallow = frozen.get_params(False)
    assert_true(
        Pointer(to=frozen.estimator()) == Pointer(to=cloned.estimator())
    )
    assert_true(Pointer(to=deep.estimator()) == Pointer(to=shallow.estimator()))
    _ = frozen.set_params()
    assert_equal(frozen.predict(X), expected)
    _ = frozen.set_params(estimator=DummyRegressor())
    assert_false(frozen.__sklearn_is_fitted__())
    assert_equal(cloned.predict(X), expected)
    assert_equal(copied.predict(X), expected)
    _ = frozen.set_params(deep)
    assert_true(
        Pointer(to=frozen.estimator()) == Pointer(to=cloned.estimator())
    )

    var replacement = DummyRegressor()
    _ = replacement.fit(X, changed)
    _ = frozen.set_params(estimator=replacement^)
    assert_equal(frozen.estimator().constant_.value(), 250.0)
    assert_equal(cloned.predict(X), expected)
    var reconstructed = F(shallow)
    assert_equal(reconstructed.predict(X), expected)
    with assert_raises():
        _ = frozen.score(X, empty)
    with assert_raises():
        _ = frozen.score(X, y, Optional[List[Float64]](changed.copy()))

    # A nested frozen estimator retains the same semantics.
    var nested = FrozenEstimator(cloned^)
    _ = nested.fit(X, changed)
    assert_equal(clone(nested).predict(X), expected)
    print("Frozen estimator contracts passed")
