import numojo as nm
from std.testing import assert_equal, assert_raises
from moxie.feature_selection._selection import SelectionState
from moxie.feature_selection._contracts import ScoreResult
from moxie._dense import row_squared_norms

def exercise[dtype: DType]() raises:
    var state = SelectionState()
    with assert_raises():
        _ = state.get_support()
    var mask: List[Bool] = [True, False, True]
    state.commit(mask)
    mask[0] = False
    var support = state.get_support(True)
    assert_equal(support[List[Int]][1], 2)
    var bools = state.get_support()
    bools[List[Bool]][0] = False
    assert_equal(state.selected_count(), 2)
    var x = nm.zeros[dtype](nm.Shape(2, 3))
    for i in range(6):
        x.unsafe_set(i, SIMD[dtype, 1](i + 1))
    var selected = state.transform(x)
    assert_equal(Float64(selected.unsafe_get(3)), 6.0)
    var restored = state.inverse_transform(selected)
    assert_equal(Float64(restored.unsafe_get(1)), 0.0)
    assert_equal(Float64(restored.unsafe_get(5)), 6.0)
    selected.unsafe_set(0, 99)
    assert_equal(Float64(x.unsafe_get(0)), 1.0)
    var reverse = x.view_with_layout(nm.Shape(2, 3), nm.NDArrayStrides(3, -1), 2)
    var reversed = state.transform(reverse)
    assert_equal(Float64(reversed.unsafe_get(0)), 3.0)
    assert_equal(Float64(reversed.unsafe_get(3)), 4.0)
    var scattered = state.inverse_transform(reversed)
    assert_equal(Float64(scattered.unsafe_get(3)), 6.0)
    var every_other = x.view_with_layout(nm.Shape(1, 3), nm.NDArrayStrides(6, 2), 0)
    var strided = state.transform(every_other)
    assert_equal(Float64(strided.unsafe_get(1)), 5.0)
    with assert_raises():
        _ = state.transform(nm.zeros[dtype](nm.Shape(3)))
    with assert_raises():
        _ = state.transform(nm.zeros[dtype](nm.Shape(0, 3)))
    with assert_raises():
        _ = state.inverse_transform(x)
    var norms = row_squared_norms(x)
    assert_equal(Float64(norms.unsafe_get(0)), 14.0)
    assert_equal(state.get_feature_names_out()[1], "x2")
    var names: List[String] = ["a", "b", "c"]
    state.commit(mask, names)
    assert_equal(state.get_feature_names_out()[0], "c")
    with assert_raises():
        _ = state.get_feature_names_out(["a", "z", "c"])
    with assert_raises():
        state.commit(mask, ["a"])
    assert_equal(state.selected_count(), 1)
    var copied = state.copy()
    state.reset()
    assert_equal(copied.selected_count(), 1)
    with assert_raises():
        _ = state.transform(x)
    copied.commit([False, False, False])
    var empty = copied.transform(x)
    assert_equal(empty.shape[1], 0)
    with assert_raises():
        _ = copied.inverse_transform(empty)
    with assert_raises():
        _ = copied.transform(nm.zeros[dtype](nm.Shape(2, 2)))
    var scores = ScoreResult[dtype](nm.zeros[dtype](nm.Shape(3)))
    assert_equal(scores.scores.shape[0], 3)
    var paired = ScoreResult[dtype](nm.zeros[dtype](nm.Shape(3)), nm.ones[dtype](nm.Shape(3)))
    assert_equal(paired.pvalues.value().shape[0], 3)
    with assert_raises():
        _ = ScoreResult[dtype](nm.zeros[dtype](nm.Shape(3)), nm.zeros[dtype](nm.Shape(2)))
    with assert_raises():
        copied.commit(List[Bool]())
    with assert_raises():
        _ = ScoreResult[dtype](nm.zeros[dtype](nm.Shape(1, 3)))

def main() raises:
    exercise[DType.float32]()
    exercise[DType.float64]()
    print("selection contracts passed")
