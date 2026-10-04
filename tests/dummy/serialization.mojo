import numojo as nm
from numojo import Shape, Item, NDArray
from std.testing import assert_equal, assert_true, assert_raises
from moxie.dummy import DummyRegressor, DummyClassifier, Label
from moxie.dummy_serialization import DummyRegressorSchema, DummyClassifierSchema
from moxie.serialization import dumps, loads, save_model, load_model
from moxie.base import clone
from moxie.serialization_snappy import SnappyCodec


def regressors[dtype: DType]() raises:
    var x = nm.ones[dtype](Shape(4, 2))
    var y = nm.zeros[dtype](Shape(4, 2))
    for i in range(4):
        y[Item(i, 0)] = Scalar[dtype](i)
        y[Item(i, 1)] = Scalar[dtype](i * 2)
    var constants = nm.ones[dtype](Shape(2))
    var strategies: List[String] = ["mean", "median", "quantile", "constant"]
    comptime Schema = DummyRegressorSchema[dtype]
    for strategy in strategies:
        var model = DummyRegressor[dtype](strategy=strategy, quantile=0.7, constant=constants.copy())
        var fresh = Schema.encode(model)
        var before_fit = Schema.decode(fresh)
        with assert_raises():
            _ = before_fit.predict(x)
        var weights = nm.ones[dtype](Shape(4))
        weights[Item(0)] = 2
        _ = model.fit(x, y, sample_weight=weights)
        # Constructor parameters may change while learned state remains fitted.
        _ = model.set_params(strategy="median")
        save_model[Schema](model, "builds/regressor.moxie")
        var restored = load_model[Schema]("builds/regressor.moxie")
        save_model[Schema, SnappyCodec](model, "builds/regressor-snappy.moxie")
        var compressed = load_model[Schema, SnappyCodec]("builds/regressor-snappy.moxie")
        var compressed_state = Schema.encode(compressed)
        var plain_state = Schema.encode(restored)
        assert_equal(dumps(compressed_state), dumps(plain_state))
        var expected = model.predict(x)
        var got = restored.predict(x)
        for i in range(4):
            for j in range(2):
                assert_equal(got.item(i, j), expected.item(i, j))
        assert_equal(restored.score(x, y), model.score(x, y))
        assert_equal(restored.strategy, "median")
        assert_equal(restored.quantile.value(), 0.7)
        assert_equal(restored.n_features_in_.value(), 2)
        var copy = clone(restored)
        assert_true(not copy.__sklearn_is_fitted__())
        restored.constant_.value()[Item(0, 0)] = 99
        assert_true(model.constant_.value().item(0, 0) != 99)
        _ = restored.fit(x, y)
        var archive = Schema.encode(model)
        archive.schema_version = 2
        with assert_raises():
            _ = Schema.decode(archive)


def classifiers[multi: Bool, dtype: DType]() raises:
    comptime Model = DummyClassifier[multi, dtype]
    comptime Schema = DummyClassifierSchema[multi, dtype]
    var x = nm.ones[dtype](Shape(4, 2))
    var single: List[Label] = [Label("猫\x00"), Label("é"), Label("猫\x00"), Label("z")]
    var rows = List[List[Label]]()
    for i in range(4):
        rows.append([single[i].copy(), Label(9007199254740993 + i % 2)])
    var y: Model.Target
    comptime if multi:
        y = rebind[Model.Target](rows).copy()
    else:
        y = rebind[Model.Target](single).copy()
    var strategies: List[String] = ["prior", "most_frequent", "uniform", "stratified", "constant"]
    for strategy in strategies:
        var constants: List[Label] = [Label("猫\x00")]
        comptime if multi:
            constants.append(Label(9007199254740993))
        var model = Model(strategy=strategy, random_state=17, constant=constants.copy())
        var fresh = Schema.encode(model)
        var before_fit = Schema.decode(fresh)
        assert_true(not before_fit.__sklearn_is_fitted__())
        _ = model.fit(x, y)
        _ = model.set_params(strategy="uniform")
        save_model[Schema](model, "builds/classifier.moxie")
        var restored = load_model[Schema]("builds/classifier.moxie")
        save_model[Schema, SnappyCodec](model, "builds/classifier-snappy.moxie")
        var compressed = load_model[Schema, SnappyCodec]("builds/classifier-snappy.moxie")
        var compressed_state = Schema.encode(compressed)
        var plain_state = Schema.encode(restored)
        assert_equal(dumps(compressed_state), dumps(plain_state))
        assert_equal(restored._strategy, strategy)
        assert_equal(restored.strategy, "uniform")
        assert_equal(restored.random_state.value(), 17)
        var expected = model.predict(x)
        var got = restored.predict(x)
        for i in range(4):
            comptime if multi:
                for j in range(2):
                    assert_true(rebind[List[List[Label]]](got)[i][j] == rebind[List[List[Label]]](expected)[i][j])
            else:
                assert_true(rebind[List[Label]](got)[i] == rebind[List[Label]](expected)[i])
        var ep = model.predict_proba(x)
        var gp = restored.predict_proba(x)
        comptime if multi:
            for j in range(2):
                for i in range(4):
                    for k in range(rebind[List[NDArray[DType.float64]]](ep)[j].shape[1]):
                        assert_equal(rebind[List[NDArray[DType.float64]]](gp)[j].item(i,k), rebind[List[NDArray[DType.float64]]](ep)[j].item(i,k))
        else:
            for i in range(4):
                for k in range(rebind[NDArray[DType.float64]](ep).shape[1]):
                    assert_equal(rebind[NDArray[DType.float64]](gp).item(i,k), rebind[NDArray[DType.float64]](ep).item(i,k))
        comptime if multi:
            with assert_raises():
                _ = model.score(x, y)
            with assert_raises():
                _ = restored.score(x, y)
        else:
            assert_equal(restored.score(x, y), model.score(x, y))
        var cloned = clone(restored)
        assert_true(not cloned.__sklearn_is_fitted__())
        var archive = Schema.encode(model)
        var bytes = dumps(archive)
        var read = loads(bytes)
        var again = Schema.decode(read)
        assert_equal(again.n_outputs_.value(), 2 if multi else 1)
        _ = restored.fit(x, y)


def edge_states() raises:
    var x = nm.ones[DType.float64](Shape(3, 1))
    var labels: List[Label] = [Label(Float64(-0.0)), Label(Float64(2.5)), Label(Float64(2.5))]
    var model = DummyClassifier(strategy="prior")
    var weights = nm.ones[DType.float64](Shape(3))
    weights[Item(0)] = 3
    _ = model.fit(x, labels, sample_weight=weights)
    var archive = DummyClassifierSchema.encode(model)
    var restored = DummyClassifierSchema.decode(archive)
    assert_true(not restored.random_state)
    assert_equal(restored.classes_[0].kind, 1)
    assert_equal(restored.class_prior_.item(0), model.class_prior_.item(0))
    with assert_raises():
        _ = DummyClassifierSchema[False, DType.float32].decode(archive)
    with assert_raises():
        _ = DummyClassifierSchema[True].decode(archive)
    archive.payloads[0].shape[0] += 1
    with assert_raises():
        _ = DummyClassifierSchema.decode(archive)
    var y = nm.ones[DType.float64](Shape(3))
    var regressor = DummyRegressor()
    _ = regressor.fit(Optional[NDArray[DType.float64]](), y)
    var state = DummyRegressorSchema.encode(regressor)
    var again = DummyRegressorSchema.decode(state)
    assert_true(not again.n_features_in_)
    assert_equal(again.predict(x).item(0), 1.0)
    state.manifest.append(0)
    with assert_raises():
        _ = DummyRegressorSchema.decode(state)


def main() raises:
    edge_states()
    regressors[DType.float32]()
    regressors[DType.float64]()
    classifiers[False, DType.float32]()
    classifiers[False, DType.float64]()
    classifiers[True, DType.float32]()
    classifiers[True, DType.float64]()
    print("dummy serialization round trips passed")
