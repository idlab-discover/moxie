"""Borrowed fluent returns preserve estimator identity through shared traits."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from numojo import NDArray, Shape
from moxie.base import (
    BaseEstimator, SupervisedEstimator, Predictor, RegressorMixin, WeightedSupervisedEstimator,
    UnsupervisedEstimator, TransformerMixin, IncrementalTransformer,
    IncrementalRegressor, IncrementalClassifier, clone,
)
from moxie.dummy import DummyRegressor
from moxie.preprocessing import StandardScaler
from moxie.pipeline import Pipeline, FeatureUnion, UnionPair, UnionSentinel
from moxie.compose import ColumnTransformer, ColumnEntry, NoTransformer, TransformedTargetRegressor
from moxie.frozen import FrozenEstimator
from test_frozen_updates import RegressorProbe, ClassifierProbe


struct ScalarEstimator(TrivialRegisterPassable, BaseEstimator):
    """Register-passable self and Params exercise the conversion-thunk boundary."""
    comptime Params = Int
    var offset: Int

    def __init__(out self, offset: Int = 0):
        self.offset = offset

    def get_params(self, deep: Bool = True) raises -> Int:
        return self.offset

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Int
    ) raises -> ref[self] Self:
        self.offset = params
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.offset)

    def __sklearn_is_fitted__(self) -> Bool:
        return False


def configure[E: BaseEstimator, origin: MutOrigin](
    ref[origin] model: E, params: E.Params
) raises -> ref[model] E:
    return model.set_params(params)


def supervised[E: SupervisedEstimator & Predictor](
    mut model: E, params: E.Params, X: E.Input, y: E.Target
) raises -> E.Prediction:
    return configure(model, params).fit(X, y).predict(X)


def weighted[E: WeightedSupervisedEstimator & RegressorMixin](
    mut model: E, params: E.Params, X: E.Input, y: E.Target, weights: E.SampleWeight
) raises -> E.Prediction:
    return model.set_params(params).fit(X, y, sample_weight=weights).predict(X)


def unsupervised[E: UnsupervisedEstimator & TransformerMixin](
    mut model: E, params: E.Params, X: E.Input
) raises -> E.Transformed where conforms_to(E.Target, Deinitable):
    return model.set_params(params).fit(X).transform(X)


def incremental[E: IncrementalTransformer](mut model: E, X: E.Input) raises -> E.Transformed where conforms_to(E.Target, Deinitable):
    return model.partial_fit(X).partial_fit(X).transform(X)


def regression_update[E: IncrementalRegressor](
    mut model: E, X: E.Input, y: E.Target
) raises -> E.Prediction:
    return model.partial_fit(X, y).partial_fit(X, y).predict(X)


def classification_update[E: IncrementalClassifier](
    mut model: E, X: E.Input, y: E.Target, var classes: E.Classes
) raises -> E.Prediction where conforms_to(E.Classes, Deinitable):
    return model.partial_fit(X, y, classes^).partial_fit(X, y).predict(X)


def check[dtype: DType]() raises:
    var X = NDArray[dtype](Shape(2, 1))
    X.itemset(0, 0)
    X.itemset(1, 2)
    var y = NDArray[dtype](Shape(2))
    y.itemset(0, 2)
    y.itemset(1, 6)
    var weights = NDArray[dtype](Shape(2))
    weights.itemset(0, 1)
    weights.itemset(1, 3)
    var model = DummyRegressor[dtype]()
    var predicted = model.set_params(strategy=String("mean")).fit(X, y).predict(X)
    assert_equal(predicted.item(0), 4)
    assert_true(model.__sklearn_is_fitted__())
    assert_equal(model.constant_.value().item(0), 4)
    predicted = weighted(model, model.get_params(), X, y, weights)
    assert_equal(predicted.item(0), 5)
    assert_equal(model.predict(X).item(0), 5)
    var params = model.get_params()
    params.strategy = "median"
    predicted = supervised(model, params, X, y)
    assert_equal(model.strategy, "median")
    assert_equal(predicted.item(0), 4)
    y.itemset(1, 10)
    _ = model.fit(X, y).fit(X, y)
    assert_equal(model.predict(X).item(0), 6)
    var fresh = clone(model)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.strategy, "median")
    _ = fresh.set_params(strategy=String("mean")).fit(X, y, sample_weight=weights)
    assert_equal(fresh.predict(X).item(0), 8)
    assert_equal(model.predict(X).item(0), 6)
    with assert_raises():
        _ = fresh.set_params(strategy=String("invalid")).fit(X, y).predict(X)
    assert_equal(fresh.strategy, "invalid")
    # A failed chain keeps prior mutations; no transactional rollback is added.
    assert_equal(model.strategy, "median")
    var unfitted = DummyRegressor[dtype]()
    var bad_y = NDArray[dtype](Shape(0))
    with assert_raises():
        _ = unfitted.fit(X, bad_y).predict(X)
    assert_false(unfitted.__sklearn_is_fitted__())

    var scaler = StandardScaler[dtype]()
    var transformed = unsupervised(scaler, scaler.get_params(), X)
    assert_equal(transformed.item(0), -1)
    assert_true(scaler.__sklearn_is_fitted__())
    transformed = incremental(scaler, X)
    assert_equal(transformed.item(1), 1)
    assert_equal(scaler.n_samples_seen_.value(), 6)
    _ = scaler.fit(X)
    assert_equal(scaler.n_samples_seen_.value(), 2)
    var names: List[String] = ["feature"]
    transformed = scaler.fit_named(X, names).transform(X)
    assert_equal(scaler.feature_names_in_.value()[0], "feature")

    var pipe = Pipeline(StandardScaler[dtype](), DummyRegressor[dtype]())
    predicted = supervised(pipe, pipe.get_params(), X, y)
    assert_equal(predicted.item(0), 6)
    assert_true(pipe.transformer.__sklearn_is_fitted__())
    assert_true(pipe.estimator.__sklearn_is_fitted__())
    var bad_params = pipe.get_params()
    bad_params.estimator_name = bad_params.transformer_name
    with assert_raises():
        _ = pipe.set_params(bad_params).fit(X, y).predict(X)
    assert_true(pipe.__sklearn_is_fitted__())
    assert_equal(pipe.predict(X).item(0), 6)
    var pipe_clone = clone(pipe)
    assert_false(pipe_clone.__sklearn_is_fitted__())
    var columns = ColumnTransformer(ColumnEntry[NoTransformer[dtype], dtype]("selected", "passthrough", [0]))
    transformed = columns.set_params(columns.get_params()).fit(X).transform(X)
    assert_equal(transformed.item(1), 2)
    assert_true(columns.__sklearn_is_fitted__())
    var union = FeatureUnion(UnionPair(UnionSentinel[dtype]("a"), UnionSentinel[dtype]("b")))
    transformed = unsupervised(union, union.get_params(), X)
    assert_equal(transformed.shape[1], 2)
    assert_true(union.__sklearn_is_fitted__())
    var target = TransformedTargetRegressor[dtype]()
    predicted = target.set_params(target.get_params()).fit(X, y).predict(X)
    assert_true(target.__sklearn_is_fitted__())
    assert_equal(predicted.shape[0], 2)
    var frozen = FrozenEstimator(model^)
    predicted = frozen.set_params().fit(X, y).predict(X)
    assert_equal(predicted.item(0), 6)
    assert_true(frozen.__sklearn_is_fitted__())


def main() raises:
    var scalar = ScalarEstimator()
    _ = configure(scalar, 4).set_params(7)
    assert_equal(scalar.offset, 7)
    check[DType.float32]()
    check[DType.float64]()
    var regression = FrozenEstimator(RegressorProbe())
    assert_equal(regression_update(regression, 2, 3), 10)
    assert_equal(regression.estimator().state, 10)
    var classifier = FrozenEstimator(ClassifierProbe())
    assert_equal(classification_update(classifier, 2, 3, 2), 10)
    assert_equal(classifier.estimator().state, 10)
    with assert_raises():
        _ = classifier.partial_fit(2, 3, 4).predict(2)
    assert_equal(classifier.estimator().state, 10)
    print("Fluent concrete, generic, incremental and composition contracts passed")
