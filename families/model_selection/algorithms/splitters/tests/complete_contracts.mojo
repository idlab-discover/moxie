from moxie.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold, ShuffleSplit, GroupShuffleSplit, StratifiedShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold, check_cv, train_test_split, SplitSize, KFold, IndexArray, IndexSplit
from moxie._random import RandomState
from numojo import NDArray, Shape, Item, zeros
from numojo.core.layout.ndstrides import NDArrayStrides
from std.testing import assert_equal, assert_true, assert_raises
from edges import array, equal


def dense[dtype: DType]() raises:
    var X = NDArray[dtype](Shape(12, 2))
    var y = array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    for i in range(12):
        X[Item(i, 0)] = Scalar[dtype](i)
        X[Item(i, 1)] = Scalar[dtype](100 + i)
    var result = train_test_split(X, y, random_state=42, test_size=SplitSize(0.25), stratify=y.copy())
    assert_equal(result[0].shape[0], 9)
    assert_equal(result[1].shape[0], 3)
    for i in range(9):
        var row = Int(result[0][Item(i, 0)])
        assert_equal(Int(result[0][Item(i, 1)]), 100 + row)
        assert_equal(result[2][Item(i)], y[Item(row)])
    X[Item(0, 0)] = -99
    for i in range(9):
        assert_true(result[0][Item(i, 0)] >= 0)
    var sequential = train_test_split(y, shuffle=False, train_size=SplitSize(4), test_size=SplitSize(3))
    assert_equal(sequential[0].size, 4)
    assert_equal(sequential[1][Item(0)], 1)
    var reversed = y.view_with_layout(Shape(12), NDArrayStrides(strides=[-1]), 11)
    var reverse_result = train_test_split(reversed, shuffle=False)
    assert_equal(reverse_result[0][Item(0)], 2)
    with assert_raises():
        _ = train_test_split(X, array([0]), random_state=1)
    with assert_raises():
        _ = train_test_split(X, shuffle=False, stratify=y.copy())
    with assert_raises():
        _ = train_test_split(X, test_size=SplitSize(1.0), random_state=1)
    with assert_raises():
        _ = train_test_split(X, train_size=SplitSize(0.8), test_size=SplitSize(0.3), random_state=1)


def main() raises:
    assert_equal(GroupShuffleSplit().get_n_splits(), 5)
    var y = array([7, 7, 7, 7, 2, 2, 2, 2, -1, -1, -1, -1])
    var groups = array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    equal(StratifiedKFold(3, shuffle=True, random_state=42).split(12, y=y.copy()), StratifiedKFold(3, shuffle=True, random_state=42).split(12, y=y.copy()))
    var a = RandomState(42)
    var b = RandomState(42)
    equal(RepeatedStratifiedKFold(n_splits=3, n_repeats=2).split(12, a, y=y.copy()), RepeatedStratifiedKFold(n_splits=3, n_repeats=2).split(12, b, y=y.copy()))
    assert_equal(a.state_position(), b.state_position())
    var saved = a.state_position()
    _ = RepeatedKFold(n_splits=3, n_repeats=2).split(12, a)
    assert_true(saved != a.state_position())
    with assert_raises():
        _ = StratifiedKFold(3).split(4, y=array([0, 0, 1, 1]))
    with assert_raises():
        _ = GroupKFold(3).split(4, groups=array([0, 0, 1, 1]))
    with assert_raises():
        _ = StratifiedGroupKFold(3).split(12, y=y.copy())
    with assert_raises():
        _ = StratifiedShuffleSplit(2, random_state=0).split(4, y=array([0, 0, 0, 1]))
    with assert_raises():
        _ = ShuffleSplit(0)
    with assert_raises():
        _ = ShuffleSplit(1, random_state=0, test_size=SplitSize(0)).split(12)
    with assert_raises():
        _ = ShuffleSplit(1).split(12)
    with assert_raises():
        _ = GroupShuffleSplit(1, random_state=0).split(12)
    var checked = check_cv(3, y=y.copy(), classifier=True)
    equal(checked.split(12, y=y.copy()), StratifiedKFold(3).split(12, y=y.copy()))
    equal(check_cv(3).split(12), KFold(3).split(12))
    equal(check_cv(KFold(2)).split(12), KFold(2).split(12))
    var pairs = List[IndexSplit]()
    pairs.append(IndexSplit(array([0, 1]), array([2, 3])))
    var iterable = check_cv(pairs)
    pairs[0].train[Item(0)] = 3
    var cursor = iterable.split(4)
    assert_equal(cursor.next().value().train[Item(0)], 0)
    dense[DType.float32]()
    dense[DType.float64]()
    print("PASS full splitter contracts and dense slicing")
