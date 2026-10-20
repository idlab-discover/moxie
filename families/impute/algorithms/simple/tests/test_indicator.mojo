import numojo as nm
from std.testing import assert_equal, assert_raises, assert_false
from moxie import clone
from moxie.impute import MissingIndicator
from moxie.impute._plumbing import MissingValue


def check[dtype: DType]() raises:
    var X = nm.zeros[dtype](nm.Shape(2, 3))
    X.unsafe_set(0, -1)
    X.unsafe_set(2, -1)
    X.unsafe_set(5, -1)
    var model = MissingIndicator[dtype](missing_values=MissingValue[dtype](-1))
    assert_equal(model.get_params().sparse[String], "auto")
    with assert_raises():
        _ = model.transform(X)
    var mask = model.fit_transform(X)
    assert_equal(mask.shape[1], 2)
    assert_equal(model.features_, [0, 2])
    assert_equal(model.n_features_in_, 3)
    assert_equal(mask.unsafe_get(0), True)
    assert_equal(mask.unsafe_get(2), False)
    assert_equal(model.get_feature_names_out(), [String("missingindicator_x0"), String("missingindicator_x2")])
    var cloned = clone(model)
    assert_false(cloned.__sklearn_is_fitted__())
    assert_equal(cloned.get_params().sparse[String], "auto")
    var explicit_auto = MissingIndicator[dtype](missing_values=MissingValue[dtype](-1), sparse="auto")
    var auto_mask = explicit_auto.fit_transform(X)
    var dense = MissingIndicator[dtype](missing_values=MissingValue[dtype](-1), sparse=False)
    var dense_mask = dense.fit_transform(X)
    for i in range(mask.size):
        assert_equal(auto_mask.unsafe_get(i), mask.unsafe_get(i))
        assert_equal(dense_mask.unsafe_get(i), mask.unsafe_get(i))
    _ = dense.set_params(sparse="auto")
    assert_false(dense.__sklearn_is_fitted__())
    assert_equal(dense.get_params().sparse[String], "auto")
    _ = dense.fit(X)
    _ = dense.set_params(sparse=False)
    assert_false(dense.__sklearn_is_fitted__())
    assert_false(dense.get_params().sparse[Bool])
    _ = dense.fit(X)
    _ = dense.set_params(sparse="invalid")
    with assert_raises():
        _ = dense.fit(X)
    _ = dense.set_params(sparse=True)
    with assert_raises():
        _ = dense.fit(X)
    X.unsafe_set(1, -1)
    with assert_raises():
        _ = model.transform(X)
    var params = model.get_params()
    params.error_on_new = False
    _ = model.set_params(params)
    assert_false(model.__sklearn_is_fitted__())
    _ = model.fit(X)
    assert_equal(model.features_, [0, 1, 2])
    params.sparse = True
    _ = model.set_params(params)
    with assert_raises():
        _ = model.fit(X)
    params.sparse = False
    params.features = "all"
    _ = model.set_params(params)
    _ = model.fit(X)
    assert_equal(model.transform(X).shape[1], 3)
    var clean = nm.zeros[dtype](nm.Shape(2, 3))
    var empty = MissingIndicator[dtype]()
    assert_equal(empty.fit_transform(clean).shape[1], 0)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    check[DType.int8]()
    check[DType.int16]()
    check[DType.int32]()
    check[DType.int64]()
    var integer = nm.zeros[DType.uint64](nm.Shape(1, 2))
    integer.unsafe_set(0, UInt64(1152921504606846976))
    integer.unsafe_set(1, UInt64(1152921504606846977))
    var exact = MissingIndicator[DType.uint64](missing_values=MissingValue[DType.uint64](UInt64(1152921504606846977)))
    assert_equal(exact.fit_transform(integer).shape[1], 1)
    assert_equal(exact.features_, [1])
    var no_nan = MissingIndicator[DType.uint64]()
    assert_equal(no_nan.fit_transform(integer).shape[1], 0)
    print("PASS: dense indicator contracts")
