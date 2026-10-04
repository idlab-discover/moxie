"""Native dtype, strided-view, and ownership contracts for NuMojo inputs."""
from std.testing import assert_equal, assert_false, assert_raises
from std.math import nan
from moxie.base import clone
from moxie.dummy import DummyClassifier, DummyRegressor, Label
from numojo import NDArray, Shape, Item, NDArrayStrides
import numojo as nm


def check_dtype[dtype: DType]() raises:
    comptime Array = NDArray[dtype]
    var X = nm.zeros[dtype](Shape(4, 2))
    var backing = nm.zeros[dtype](Shape(8))
    var weight_backing = nm.zeros[dtype](Shape(8))
    for i in range(8):
        backing[Item(i)] = Scalar[dtype](i)
        weight_backing[Item(i)] = Scalar[dtype](i + 1)
    var y = backing.view_with_layout(Shape(4), NDArrayStrides(-2), 7)
    var weights = weight_backing.view_with_layout(Shape(4), NDArrayStrides(-2), 7)
    var reg = DummyRegressor[dtype]()
    with assert_raises():
        _ = reg.predict(X)
    _ = reg.fit(X, y)
    var result = reg.predict(X, return_std=True)
    comptime assert type_of(result.prediction) == Array
    comptime assert type_of(result.std.value()) == NDArray[DType.float64]
    assert_equal(result.prediction.item(0), Scalar[dtype](4))
    assert_equal(result.std.value().item(0), Float64(0))
    _ = reg.fit(X, y, sample_weight=weights)
    assert_equal(reg.predict(X).item(0), Scalar[dtype](5))
    assert_equal(reg.score(X, y, weights.copy()), 0.0)
    assert_equal(reg.score(Optional[Array](), y, weights.copy()), 0.0)
    var fresh = clone(reg)
    comptime assert type_of(fresh) == DummyRegressor[dtype]
    assert_false(fresh.__sklearn_is_fitted__())
    backing[Item(7)] = 99
    assert_equal(reg.predict(X).item(0), Scalar[dtype](5))
    backing[Item(7)] = 7
    _ = reg.set_params(strategy="median")
    _ = reg.fit(X, y)
    assert_equal(reg.predict(X).item(0), Scalar[dtype](4))
    _ = reg.set_params(strategy="quantile", quantile=Optional[Optional[Float64]](0.0))
    _ = reg.fit(X, y, sample_weight=weights)
    assert_equal(reg.predict(X).item(0), Scalar[dtype](1))
    _ = reg.set_params(quantile=Optional[Optional[Float64]](1.0))
    _ = reg.fit(X, y, sample_weight=weights)
    assert_equal(reg.predict(X).item(0), Scalar[dtype](7))
    var matrix_backing = nm.zeros[dtype](Shape(16))
    for i in range(16):
        matrix_backing[Item(i)] = Scalar[dtype](i)
    var matrix = matrix_backing.view_with_layout(Shape(4, 2), NDArrayStrides(-4, 2), 12)
    _ = reg.set_params(strategy="mean")
    _ = reg.fit(X, matrix, sample_weight=weights)
    var multi = reg.predict(X)
    assert_equal(multi.shape, Shape(4, 2))
    assert_equal(multi.item(0, 0), Scalar[dtype](8))
    assert_equal(multi.item(0, 1), Scalar[dtype](10))
    var constants = matrix_backing.view_with_layout(Shape(2), NDArrayStrides(-2), 14)
    var constant_model = DummyRegressor[dtype](strategy="constant", constant=constants.copy())
    matrix_backing[Item(14)] = 100
    _ = constant_model.fit(X, matrix)
    assert_equal(constant_model.predict(X).item(0, 0), Scalar[dtype](14))
    var params = constant_model.get_params()
    params.constant.value()[Item(0)] = 99
    assert_equal(constant_model.constant.value()[Item(0)], Scalar[dtype](14))
    var scalar = DummyRegressor[dtype](strategy="constant", constant=Scalar[dtype](3))
    _ = scalar.fit(X, y)
    assert_equal(scalar.predict(X).item(0), Scalar[dtype](3))
    _ = scalar.set_params(constant=Scalar[dtype](2))
    _ = scalar.fit(X, y)
    assert_equal(scalar.predict(X).item(0), Scalar[dtype](2))
    var invalid = y.copy()
    invalid[Item(0)] = nan[dtype]()
    with assert_raises():
        _ = reg.fit(X, invalid)
    var zero_weights = nm.zeros[dtype](Shape(4))
    with assert_raises():
        _ = reg.fit(X, y, sample_weight=zero_weights)
    assert_equal(scalar.predict(nm.zeros[dtype](Shape(0, 3))).shape, Shape(0))
    var labels: List[Label] = [Label(1), Label(1), Label(2), Label(2)]
    var clf = DummyClassifier[dtype=dtype]()
    _ = clf.fit(X, labels, sample_weight=weights)
    var probabilities = clf.predict_proba(X)
    comptime assert type_of(probabilities) == NDArray[DType.float64]
    assert_equal(probabilities.item(0, 0), 0.7)
    assert_equal(clf.score(Optional[Array](), labels, weights.copy()), 0.7)
    var cloned = clone(clf)
    comptime assert type_of(cloned) == DummyClassifier[dtype=dtype]
    assert_false(cloned.__sklearn_is_fitted__())
    var classifier_multi = DummyClassifier[multioutput=True, dtype=dtype](random_state=42, strategy="stratified")
    var rows: List[List[Label]] = [[Label(1), Label(0)], [Label(1), Label(1)], [Label(2), Label(0)], [Label(2), Label(1)]]
    _ = classifier_multi.fit(X, rows, sample_weight=weights)
    assert_equal(len(classifier_multi.predict_proba(X)), 2)
    print("PASS NuMojo dtype, strides, output types and ownership:", dtype)


def main() raises:
    check_dtype[DType.float32]()
    check_dtype[DType.float64]()
