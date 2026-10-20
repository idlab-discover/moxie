import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie import clone
from moxie.impute import SimpleImputer
from moxie.impute._simple import ColumnStrategy
from moxie.impute._plumbing import MissingValue

struct Minimum(ColumnStrategy):
    def __init__(out self):
        pass
    def __call__(self, values: nm.NDArray[DType.float64]) raises -> Float64:
        if values.size == 0:
            return 17
        var result = values.unsafe_get(0)
        for i in range(values.size):
            result = min(result, values.unsafe_get(i))
        return result


def check[dtype: DType]() raises:
    var X = nm.zeros[dtype](nm.Shape(3, 3))
    var data: List[Int] = [1, -1, -1, 3, 2, -1, -1, 6, -1]
    for i in range(9):
        X.unsafe_set(i, SIMD[dtype, 1](data[i]))
    for strategy in [String("mean"), String("median"), String("most_frequent"), String("constant")]:
        var model = SimpleImputer[dtype](strategy=strategy, missing_values=MissingValue[dtype](-1), add_indicator=True)
        with assert_raises():
            _ = model.transform(X)
        var out = model.fit_transform(X)
        assert_equal(out.shape[1], 5)
        assert_equal(out.unsafe_get(4), 1)
        assert_true(model.statistics_.unsafe_get(2) != model.statistics_.unsafe_get(2))
        if strategy == "mean" or strategy == "median":
            assert_equal(out.unsafe_get(1), 4)
            assert_equal(out.unsafe_get(10), 2)
        if strategy == "most_frequent":
            assert_equal(out.unsafe_get(1), 2)
            assert_equal(out.unsafe_get(10), 1)
        var inverse = model.inverse_transform(out)
        for i in range(9):
            assert_equal(inverse.unsafe_get(i), Float64(X.unsafe_get(i)))
        var copied = clone(model)
        assert_false(copied.__sklearn_is_fitted__())
        var params = model.get_params()
        params.keep_empty_features = True
        params.fill_value = SIMD[dtype, 1](9)
        _ = model.set_params(params)
        assert_false(model.__sklearn_is_fitted__())
        var kept = model.fit_transform(X)
        assert_equal(kept.shape[1], 6)
        assert_equal(kept.unsafe_get(2), SIMD[dtype, 1](9 if strategy == "constant" else 0))
    var custom = SimpleImputer[dtype, Minimum](strategy="callable", callback=Minimum(), missing_values=MissingValue[dtype](-1))
    var output = custom.fit_transform(X)
    assert_equal(output.shape[1], 3)
    assert_equal(output.unsafe_get(2), 17)
    assert_equal(output.unsafe_get(6), 1)
    var no_indicator = SimpleImputer[dtype](missing_values=MissingValue[dtype](-1))
    _ = no_indicator.fit(X)
    with assert_raises():
        _ = no_indicator.inverse_transform(X)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: dense simple numeric strategies")
