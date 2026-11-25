from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from moxie.ensemble._hist_gradient_boosting.binning import _BinMapper
from moxie.ensemble._hist_gradient_boosting.histogram import (
    _HistogramBin,
    compute_histograms,
)
from moxie.ensemble._hist_gradient_boosting.splitting import (
    _SplitOptions,
    _ValueBounds,
    compute_node_value,
    find_node_split,
    partition_rows,
)
from numojo import NDArray, NDArrayStrides, Shape


def test_split_mapper_and_subtracted_histograms() raises:
    var X = NDArray[DType.float64](Shape(6, 2))
    var values: List[Float64] = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    for i in range(len(values)):
        X.unsafe_set(i, values[i])
    var mapper = _BinMapper(n_bins=4)
    var binned = mapper.fit(X).transform(X)
    var reversed = binned.view_with_layout(
        Shape(6, 2), NDArrayStrides(-1, binned.strides[1]), 5
    )
    var rows = [0, 1, 2, 3, 4, 5]
    var sibling_rows = [4, 5]
    var kept_rows = [3, 0, 2, 1]
    var gradients: List[Float32] = [-2, 2, -2, 2, -2, 2]
    var hessians = List[Float32]()
    var parent = compute_histograms(
        reversed, 4, Span(rows), Span(gradients), Span(hessians)
    )
    var sibling = compute_histograms(
        reversed, 4, Span(sibling_rows), Span(gradients), Span(hessians)
    )
    var kept = parent^.subtract_sibling(sibling)

    var total = _HistogramBin(0.0, 4.0, 4)
    var bounds = _ValueBounds(-1, 1)
    var options = _SplitOptions(min_samples_leaf=2)
    var value = compute_node_value(total, bounds, options.l2_regularization)
    var selected = [1, 0]
    var missing: List[Bool] = [False, False]
    var monotonic: List[Int] = [-1, 1]
    var result = find_node_split(
        kept,
        total,
        value,
        mapper.n_bins_non_missing_,
        missing,
        mapper.is_categorical_,
        monotonic,
        allowed_features=Span(selected),
        bounds=bounds,
        options=options,
    )
    assert_true(Bool(result))
    ref split = result.value()
    assert_equal(split.feature, 1)
    assert_equal(split.bin_idx, UInt8(0))
    assert_equal(split.value_left, -1.0)
    assert_equal(split.value_right, 1.0)
    assert_equal(split.gain, 8.0)

    var scratch = List[Int](length=4, fill=0)
    var boundary = partition_rows(
        reversed,
        split,
        mapper.missing_values_bin_idx_,
        Span(kept_rows),
        Span(scratch),
    )
    assert_equal(boundary, 2)
    assert_equal(kept_rows, [3, 1, 0, 2])

    var invalid_rows = [3, 1, 1, 0]
    with assert_raises(contains="left partition count"):
        _ = partition_rows(
            reversed,
            split,
            mapper.missing_values_bin_idx_,
            Span(invalid_rows),
            Span(scratch),
        )
    assert_equal(invalid_rows, [3, 1, 1, 0])


def test_zero_candidates_and_invalid_metadata() raises:
    var X = NDArray[DType.uint8](Shape(2, 1))
    X.unsafe_set(0, 0)
    X.unsafe_set(1, 1)
    var rows = [0, 1]
    var gradients: List[Float32] = [1, -1]
    var hessians = List[Float32]()
    var histograms = compute_histograms(
        X, 3, Span(rows), Span(gradients), Span(hessians)
    )
    var total = _HistogramBin(0.0, 2.0, 2)
    var counts: List[UInt32] = [2]
    var no: List[Bool] = [False]
    var yes: List[Bool] = [True]
    var monotonic: List[Int] = [1]
    var features = [0]
    var none = List[Int]()
    var result = find_node_split(
        histograms,
        total,
        0,
        counts,
        no,
        no,
        monotonic,
        allowed_features=Span(none),
    )
    assert_true(not result)
    with assert_raises(contains="categorical features cannot be monotonic"):
        _ = find_node_split(
            histograms,
            total,
            0,
            counts,
            no,
            yes,
            monotonic,
            allowed_features=Span(features),
        )
    var no_counts = List[UInt32]()
    with assert_raises(contains="metadata must match"):
        _ = find_node_split(
            histograms,
            total,
            0,
            no_counts,
            no,
            no,
            monotonic,
            allowed_features=Span(features),
        )
    with assert_raises(contains="mask must match"):
        _ = find_node_split(
            histograms,
            total,
            0,
            counts,
            no,
            no,
            monotonic,
            allowed_features=Span(features),
            feature_mask=[True, False],
        )
    var excluded = compute_histograms(
        X,
        3,
        Span(rows),
        Span(gradients),
        Span(hessians),
        allowed_features=none.copy(),
    )
    with assert_raises(contains="absent from histograms"):
        _ = find_node_split(
            excluded,
            total,
            0,
            counts,
            no,
            no,
            monotonic,
            allowed_features=Span(features),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
