from std.testing import assert_equal, assert_true, assert_false, assert_raises
import numojo as nm
from moxie.impute._plumbing import (
    MissingValue, missing_mask, FeatureLayout, select_features,
    append_indicator, ImputationStep, validate_indices,
)


def numeric[dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(2, 3))
    # Missing in columns 0 and 2; column 2 entirely empty.
    x.unsafe_set(0, -1)
    x.unsafe_set(1, 2)
    x.unsafe_set(2, -1)
    x.unsafe_set(3, 4)
    x.unsafe_set(4, 5)
    x.unsafe_set(5, -1)
    var zero_mask = missing_mask(x, MissingValue[dtype](0))
    assert_false(zero_mask.unsafe_get(1))
    var marker = MissingValue[dtype](-1)
    var mask = missing_mask(x, marker)
    assert_true(mask.unsafe_get(0))
    assert_false(mask.unsafe_get(1))
    var layout = FeatureLayout()
    with assert_raises():
        _ = layout.indicator(mask)
    with assert_raises():
        layout.set_retained([0])
    layout.fit(mask, error_on_new=False)
    assert_equal(layout.retained_features, [0, 1])
    assert_equal(layout.indicator_features, [0, 2])
    assert_equal(layout.empty_features, [2])
    var filled = select_features(x, layout.retained_features)
    filled.unsafe_set(0, 7)
    var out = append_indicator(filled, mask, layout)
    assert_equal(out.shape[0], 2)
    assert_equal(out.shape[1], 4)
    assert_equal(out.unsafe_get(0), 7)
    assert_equal(out.unsafe_get(2), 1)
    assert_equal(out.unsafe_get(3), 1)
    assert_equal(out.unsafe_get(6), 0)
    assert_equal(out.unsafe_get(7), 1)
    # Mask and selected output never borrow mutable caller data.
    x.unsafe_set(0, 9)
    assert_true(mask.unsafe_get(0))
    assert_equal(filled.unsafe_get(0), 7)
    filled.unsafe_set(0, 8)
    assert_equal(out.unsafe_get(0), 7)
    var copied = layout.copy()
    copied.set_retained([1])
    assert_equal(layout.retained_features, [0, 1])
    assert_equal(copied.retained_features, [1])
    with assert_raises():
        copied.set_retained([1, 1])
    assert_equal(copied.retained_features, [1])
    with assert_raises():
        layout.fit(mask, features="invalid")
    assert_equal(layout.indicator_features, [0, 2])
    var new_mask = mask.copy()
    new_mask.unsafe_set(1, True)
    layout.fit(mask)
    with assert_raises():
        _ = layout.indicator(new_mask)
    layout.fit(mask, features="all", keep_empty_features=True)
    assert_equal(layout.retained_features, [0, 1, 2])
    var all_indicators = layout.indicator(new_mask)
    assert_equal(all_indicators.shape[1], 3)
    var plain = nm.zeros[DType.bool](nm.Shape(2, 3))
    layout.fit(plain)
    assert_equal(len(layout.indicator_features), 0)
    var no_indicators = layout.indicator(plain)
    assert_equal(no_indicators.shape[1], 0)
    var empty = nm.zeros[DType.bool](nm.Shape(2, 3))
    for i in range(6):
        empty.unsafe_set(i, True)
    layout.fit(empty)
    var no_features = select_features(x, layout.retained_features)
    assert_equal(no_features.shape[1], 0)
    var only_indicators = append_indicator(no_features, empty, layout)
    assert_equal(only_indicators.shape[1], 3)
    for i in range(6):
        assert_equal(only_indicators.unsafe_get(i), 1)
    with assert_raises():
        _ = append_indicator(x, empty, layout)
    var wrong = nm.zeros[DType.bool](nm.Shape(1, 2))
    with assert_raises():
        _ = layout.indicator(wrong)
    var wrong_rows = nm.zeros[dtype](nm.Shape(1, 0))
    with assert_raises():
        _ = append_indicator(wrong_rows, empty, layout)
    var vector = nm.zeros[dtype](nm.Shape(3))
    with assert_raises():
        _ = missing_mask(vector, marker)
    var zero = nm.zeros[dtype](nm.Shape(0, 3))
    with assert_raises():
        _ = missing_mask(zero, marker)
    var zero_cols = nm.zeros[dtype](nm.Shape(2, 0))
    with assert_raises():
        _ = missing_mask(zero_cols, marker)
    x.unsafe_set(0, Float64("nan").cast[dtype]())
    var nanmask = missing_mask(x, MissingValue[dtype]())
    assert_true(nanmask.unsafe_get(0))
    var explicit_nan = missing_mask(x, MissingValue[dtype](Float64("nan").cast[dtype]()))
    assert_true(explicit_nan.unsafe_get(0))
    with assert_raises():
        _ = missing_mask(x, marker)
    x.unsafe_set(0, Float64("inf").cast[dtype]())
    with assert_raises():
        _ = missing_mask(x, MissingValue[dtype]())
    with assert_raises():
        _ = MissingValue[dtype](Float64("inf").cast[dtype]())
    # Nonzero offset, negative row stride, positive column stride.
    var backing = nm.zeros[dtype](nm.Shape(12))
    backing.unsafe_set(8, -1)
    var strided = backing.view_with_layout(nm.Shape(2, 2), nm.NDArrayStrides(-4, 2), 8)
    var sm = missing_mask(strided, marker)
    assert_true(sm.unsafe_get(0))
    assert_false(sm.unsafe_get(1))
    assert_false(sm.unsafe_get(2))
    var selected = select_features(strided, [0, 1])
    assert_equal(selected.unsafe_get(0), -1)
    assert_equal(selected.unsafe_get(2), 0)
    var bm = nm.zeros[DType.bool](nm.Shape(12))
    bm.unsafe_set(8, True)
    var mask_view = bm.view_with_layout(nm.Shape(2, 2), nm.NDArrayStrides(-4, 2), 8)
    layout.fit(mask_view, error_on_new=False)
    assert_equal(layout.indicator_features, [0])
    var joined = append_indicator(strided, mask_view, layout)
    assert_equal(joined.unsafe_get(2), 1)
    assert_equal(joined.unsafe_get(5), 0)


@fieldwise_init
struct ModelRecord(Copyable, Movable):
    var learned: List[Int]


def main() raises:
    numeric[DType.float32]()
    numeric[DType.float64]()
    var invalid_indices: List[List[Int]] = [[-1], [3], [1, 0], [1, 1]]
    for indices in invalid_indices:
        with assert_raises():
            validate_indices(indices, 3)
    var indices: List[Int] = [2, 0]
    var model = ModelRecord([42])
    var step = ImputationStep(1, indices, model^, 3)
    indices[0] = 1
    assert_equal(step.neighbor_feat_idx, [2, 0])
    var copy = step.copy()
    copy.estimator.learned[0] = 9
    assert_equal(step.estimator.learned[0], 42)
    assert_equal(copy.estimator.learned[0], 9)
    with assert_raises():
        _ = ImputationStep(3, [0], ModelRecord([1]), 3)
    with assert_raises():
        _ = ImputationStep(1, [1], ModelRecord([1]), 3)
    with assert_raises():
        _ = ImputationStep(1, [0, 0], ModelRecord([1]), 3)
    print("PASS: dense imputation plumbing contracts")
