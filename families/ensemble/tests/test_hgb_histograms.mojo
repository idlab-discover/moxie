from std.math import nan
from std.testing import TestSuite, assert_equal, assert_raises

from moxie.ensemble._hist_gradient_boosting.binning import _BinMapper
from moxie.ensemble._hist_gradient_boosting.histogram import compute_histograms
from numojo import NDArray, NDArrayStrides, Shape


def test_mapper_output_and_strided_views() raises:
    var X = NDArray[DType.float64](Shape(4, 2))
    var values: List[Float64] = [0, 3, 1, 0, nan[DType.float64](), 3, 0, 1]
    for i in range(len(values)):
        X.unsafe_set(i, values[i])
    var mapper = _BinMapper(n_bins=4)
    var binned = mapper.fit(X).transform(X)
    var reversed = binned.view_with_layout(
        Shape(4, 2), NDArrayStrides(-1, binned.strides[1]), 3
    )
    var rows = [0, 1, 2, 3]
    var gradients: List[Float32] = [4, 3, 2, 1]
    var hessians: List[Float32] = [0.5, 0.25, 1, 2]
    var result = compute_histograms(
        reversed, 4, Span(rows), Span(gradients), Span(hessians)
    )
    assert_equal(result.feature(0)[0].count, 2)
    assert_equal(result.feature(0)[0].sum_gradients, 5.0)
    assert_equal(result.feature(0)[0].sum_hessians, 2.5)
    assert_equal(result.feature(0)[3].count, 1)
    assert_equal(result.feature(0)[3].sum_gradients, 3.0)
    assert_equal(result.feature(1)[2].count, 2)
    assert_equal(result.feature(1)[2].sum_gradients, 4.0)
    assert_equal(result.feature(1)[2].sum_hessians, 2.25)


def test_empty_partitions_and_feature_selection() raises:
    var X = NDArray[DType.uint8](Shape(1, 2))
    X.unsafe_set(0, 0)
    X.unsafe_set(1, 2)
    var rows = [0]
    var empty_rows = List[Int]()
    var gradients: List[Float32] = [1]
    var constant = List[Float32]()
    var selected: List[Int] = [1]
    var parent = compute_histograms(X, 3, Span(rows), Span(gradients), Span(constant))
    var sibling = compute_histograms(
        X, 3, Span(empty_rows), Span(gradients), Span(constant), allowed_features=selected^
    )
    assert_equal(len(sibling.feature(0)), 0)
    assert_equal(sibling.feature(1)[2].count, 0)
    var other = parent^.subtract_sibling(sibling)
    assert_equal(len(other.feature(0)), 0)
    assert_equal(other.feature(1)[2].count, 1)
    assert_equal(other.feature(1)[2].sum_gradients, 1.0)
    assert_equal(other.feature(1)[2].sum_hessians, 0.0)


def test_full_partition_uses_supplied_order() raises:
    var X = NDArray[DType.uint8](Shape(3, 1))
    for row in range(3):
        X.unsafe_set(row, 0)
    var ordered = [0, 1, 2]
    var permuted = [0, 2, 1]
    var gradients: List[Float32] = [1e20, -1e20, 1]
    var constant = List[Float32]()
    var root = compute_histograms(X, 3, Span(ordered), Span(gradients), Span(constant))
    var reordered = compute_histograms(
        X, 3, Span(permuted), Span(gradients), Span(constant)
    )
    assert_equal(root.feature(0)[0].sum_gradients, 1.0)
    assert_equal(reordered.feature(0)[0].sum_gradients, 0.0)
    assert_equal(reordered.feature(0)[0].count, 3)


def test_invalid_input_and_subtraction() raises:
    var X = NDArray[DType.uint8](Shape(1, 2))
    X.unsafe_set(0, 0)
    X.unsafe_set(1, 3)
    var rows = [0]
    var bad_rows = [-1]
    var empty_rows = List[Int]()
    var gradients: List[Float32] = [1]
    var constant = List[Float32]()
    var bad_hessians: List[Float32] = [1, 1]
    with assert_raises():
        _ = compute_histograms(X, 4, Span(bad_rows), Span(gradients), Span(constant))
    with assert_raises():
        _ = compute_histograms(X, 3, Span(rows), Span(gradients), Span(constant))
    with assert_raises():
        _ = compute_histograms(X, 257, Span(rows), Span(gradients), Span(constant))
    with assert_raises():
        _ = compute_histograms(X, 4, Span(rows), Span(constant), Span(constant))
    with assert_raises():
        _ = compute_histograms(X, 4, Span(rows), Span(gradients), Span(bad_hessians))
    with assert_raises():
        var selected: List[Int] = [1, 1]
        _ = compute_histograms(
            X, 4, Span(rows), Span(gradients), Span(constant), allowed_features=selected^
        )
    with assert_raises():
        var selected: List[Int] = [2]
        _ = compute_histograms(
            X, 4, Span(rows), Span(gradients), Span(constant), allowed_features=selected^
        )
    var sibling = compute_histograms(X, 4, Span(rows), Span(gradients), Span(constant))
    with assert_raises():
        _ = sibling.feature(-1)
    with assert_raises():
        var empty = compute_histograms(
            X, 4, Span(empty_rows), Span(gradients), Span(constant)
        )
        _ = empty^.subtract_sibling(sibling)
    with assert_raises():
        var selected: List[Int] = [0]
        var subset = compute_histograms(
            X, 4, Span(rows), Span(gradients), Span(constant), allowed_features=selected^
        )
        _ = subset^.subtract_sibling(sibling)
    with assert_raises():
        var variable = compute_histograms(
            X, 4, Span(rows), Span(gradients), Span(gradients)
        )
        _ = variable^.subtract_sibling(sibling)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
