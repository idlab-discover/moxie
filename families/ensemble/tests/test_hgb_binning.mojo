"""Contracts for the private histogram gradient boosting bin mapper."""
from std.math import nan
from std.testing import assert_equal, assert_raises, assert_true
from numojo import NDArray, NDArrayStrides, Shape
from moxie.ensemble._hist_gradient_boosting.binning import _BinMapper


def test_layout_and_lifecycle() raises:
    var data = NDArray[DType.float32](Shape(3, 2))
    for i in range(data.size):
        data.unsafe_set(i, Float32(i))
    var reversed = data.view_with_layout(Shape(3, 2), NDArrayStrides(-2, 1), 4)
    var mapper = _BinMapper(n_bins=5)
    with assert_raises(contains="NotFittedError"):
        _ = mapper.transform(reversed)
    with assert_raises(contains="NotFittedError"):
        _ = mapper.make_known_categories_bitsets()

    var binned = mapper.fit(reversed).transform(reversed)
    assert_equal(binned.ndim, 2)
    assert_equal(binned.shape[0], 3)
    assert_equal(binned.shape[1], 2)
    assert_true(binned.is_f_contiguous())
    assert_equal(mapper.bin_thresholds_[0][0], 1.0)
    assert_equal(mapper.bin_thresholds_[1][1], 4.0)
    for feature in range(2):
        for row in range(3):
            assert_equal(binned.unsafe_get(feature * 3 + row), UInt8(2 - row))

    var other = NDArray[DType.float64](Shape(2, 1))
    other.unsafe_set(0, 10)
    other.unsafe_set(1, 20)
    with assert_raises(contains="fitted feature count"):
        _ = mapper.transform(other)
    _ = mapper.fit(other)
    assert_equal(len(mapper.bin_thresholds_), 1)
    assert_equal(mapper.bin_thresholds_[0][0], 15.0)
    mapper.n_bins = 2
    with assert_raises(contains="n_bins must be"):
        _ = mapper.fit(other)
    assert_equal(mapper.transform(other).unsafe_get(1), UInt8(1))
    assert_equal(mapper.missing_values_bin_idx_, UInt8(4))


def test_degenerate_columns_and_validation() raises:
    var data = NDArray[DType.float64](Shape(3, 1))
    var weights = NDArray[DType.float64](Shape(3))
    for i in range(3):
        data.unsafe_set(i, nan[DType.float64]())
        weights.unsafe_set(i, 0)
    var mapper = _BinMapper()
    var binned = mapper.fit(data).transform(data)
    assert_equal(mapper.n_bins_non_missing_[0], UInt32(1))
    assert_equal(len(mapper.bin_thresholds_[0]), 0)
    for i in range(3):
        assert_equal(binned.unsafe_get(i), UInt8(255))

    data.unsafe_set(0, 5)
    _ = mapper.fit(data, sample_weight=weights.view())
    assert_equal(len(mapper.bin_thresholds_[0]), 0)
    assert_equal(mapper.transform(data).unsafe_get(0), UInt8(0))

    # Weight access follows a reversed view, including its offset.
    data.unsafe_set(1, 10)
    data.unsafe_set(2, 25)
    weights.unsafe_set(0, 1)
    weights.unsafe_set(1, 1)
    var reversed_weights = weights.view_with_layout(Shape(3), NDArrayStrides(-1), 2)
    _ = mapper.fit(data, sample_weight=reversed_weights^)
    assert_equal(mapper.bin_thresholds_[0][0], 17.5)

    var domains = List[Optional[List[Float64]]]()
    with assert_raises(contains="one entry per feature"):
        _ = mapper.fit(data, known_categories=domains.copy())
    var unsorted: List[Float64] = [12.0, 11.0]
    domains.append(unsorted^)
    with assert_raises(contains="sorted distinct integers"):
        _ = mapper.fit(data, known_categories=domains^)

    weights.unsafe_set(0, -1)
    with assert_raises(contains="nonnegative"):
        _ = mapper.fit(data, sample_weight=weights.view())
    with assert_raises(contains="nonempty rank-2"):
        _ = mapper.fit(weights)
    with assert_raises(contains="sample_weight.shape"):
        _ = mapper.fit(data, sample_weight=NDArray[DType.float64](Shape(1)))


def main() raises:
    test_layout_and_lifecycle()
    test_degenerate_columns_and_validation()
    print("PASS: HGB bin mapper native contracts")
