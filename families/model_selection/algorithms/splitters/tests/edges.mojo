from moxie.model_selection import KFold, TimeSeriesSplit, LeaveOneOut, LeavePOut, LeaveOneGroupOut, LeavePGroupsOut, PredefinedSplit
from moxie.model_selection._indices import IndexArray, SplitCursor
from moxie._random import RandomState
from numojo import Shape, Item
from std.testing import assert_equal, assert_true, assert_false, assert_raises


def array(values: List[Int]) raises -> IndexArray:
    var result = IndexArray(Shape(len(values)))
    for i in range(len(values)):
        result[Item(i)] = Int64(values[i])
    return result^


def equal[C: SplitCursor & Deinitable](var left: C, var right: C) raises:
    while True:
        var a = left.next()
        var b = right.next()
        assert_equal(Bool(a), Bool(b))
        if not a:
            break
        assert_equal(a.value().train.size, b.value().train.size)
        assert_equal(a.value().test.size, b.value().test.size)
        for i in range(a.value().train.size):
            assert_equal(a.value().train[Item(i)], b.value().train[Item(i)])
        for i in range(a.value().test.size):
            assert_equal(a.value().test[Item(i)], b.value().test[Item(i)])


def main() raises:
    with assert_raises():
        _ = KFold(1)
    with assert_raises():
        _ = KFold(2, random_state=0)
    with assert_raises():
        _ = KFold(3).split(2)
    with assert_raises():
        _ = KFold(2).split(4, y=array([0, 1]))
    with assert_raises():
        _ = KFold(2).split(4, groups=array([0, 1]))
    with assert_raises():
        _ = KFold(2, shuffle=True).split(4)
    with assert_raises():
        _ = TimeSeriesSplit(3, gap=4, test_size=2).split(10)
    with assert_raises():
        _ = TimeSeriesSplit(3, test_size=0).split(10)
    with assert_raises():
        _ = LeaveOneOut().split(1)
    with assert_raises():
        _ = LeavePOut(4).split(4)
    with assert_raises():
        _ = LeaveOneGroupOut().split(4)
    with assert_raises():
        _ = LeaveOneGroupOut().split(4, groups=array([1, 1, 1, 1]))
    with assert_raises():
        _ = LeavePGroupsOut(2).split(4, groups=array([1, 2, 1, 2]))
    with assert_raises():
        _ = PredefinedSplit(IndexArray(Shape(2, 2)))

    var cv = KFold(3, shuffle=True, random_state=42)
    equal(cv.split(11), cv.split(11))
    var a = RandomState(42)
    var b = RandomState(42)
    equal(cv.split(11, a), cv.split(11, b))
    equal(cv.split(11, a), cv.split(11, b))
    var fresh = RandomState(42)
    var advanced_cursor = cv.split(11, a)
    var fresh_cursor = cv.split(11, fresh)
    var advanced = advanced_cursor.next()
    var initial = fresh_cursor.next()
    var differs = False
    for i in range(advanced.value().test.size):
        differs = differs or advanced.value().test[Item(i)] != initial.value().test[Item(i)]
    assert_true(differs)

    var labels = array([-1, 0, 1, 0])
    var predefined = PredefinedSplit(labels)
    labels[Item(1)] = 9
    assert_equal(Int(predefined.test_fold[Item(1)]), 0)
    var cursor = predefined.split()
    var first = cursor.next()
    var owned = first.value().copy()
    owned.test[Item(0)] = 99
    var independent = predefined.split()
    var again = independent.next()
    assert_equal(Int(again.value().test[Item(0)]), 1)
    _ = cursor.next()
    assert_false(Bool(cursor.next()))
    assert_false(Bool(cursor.next()))
    assert_equal(LeavePOut(2).get_n_splits(5), 10)
    assert_equal(LeavePGroupsOut(2).get_n_splits(groups=array([5, 5, -1, 8])), 3)
    print("PASS negative inputs, RNG replay/advancement, snapshot ownership, exhaustion")
