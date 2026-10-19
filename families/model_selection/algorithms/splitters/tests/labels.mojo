"""Native contracts for String/Float32/Float64 labels and groups (SplitLabels)."""
from moxie.model_selection import StratifiedKFold, StratifiedShuffleSplit, StratifiedGroupKFold, GroupKFold, LeaveOneGroupOut, RepeatedStratifiedKFold, SplitLabels, SplitSize, check_cv, train_test_split
from moxie.model_selection._indices import IndexArray, SplitCursor
from moxie._random import RandomState
from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_true, assert_false, assert_raises


def floats(values: List[Float64]) raises -> NDArray[DType.float64]:
    var result = NDArray[DType.float64](Shape(len(values)))
    for i in range(len(values)):
        result[Item(i)] = values[i]
    return result^


def ints(values: List[Int]) raises -> IndexArray:
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
        assert_equal(a.value().test.size, b.value().test.size)
        for i in range(a.value().test.size):
            assert_equal(a.value().test[Item(i)], b.value().test[Item(i)])


def test_handoff_reproducer() raises:
    # Integral floats are binary labels; sklearn accepts [1., 1., 2., 2.].
    var cursor = StratifiedKFold(2).split(4, y=floats([1.0, 1.0, 2.0, 2.0]))
    var first = cursor.next().value().copy()
    assert_equal(first.test.size, 2)
    # ...and rejects continuous targets with its own message.
    with assert_raises(contains="Supported target types are: ('binary', 'multiclass'). Got 'continuous' instead."):
        _ = StratifiedKFold(2).split(4, y=floats([1.1, 1.2, 2.1, 2.2]))


def test_strings_match_int64_codes() raises:
    # Sorted classes are apple=0, fig=1, pear=2; the Int64 path must agree exactly.
    var words: List[String] = ["pear", "apple", "fig", "apple", "pear", "fig", "pear", "apple", "fig"]
    var codes = ints([2, 0, 1, 0, 2, 1, 2, 0, 1])
    equal(StratifiedKFold(3, shuffle=True, random_state=4).split(9, y=words),
          StratifiedKFold(3, shuffle=True, random_state=4).split(9, y=codes.copy()))
    equal(StratifiedShuffleSplit(3, test_size=SplitSize(3), random_state=2).split(9, y=words),
          StratifiedShuffleSplit(3, test_size=SplitSize(3), random_state=2).split(9, y=codes.copy()))
    # A mutable application-owned stream gives the same folds as the equal seed.
    var rng = RandomState(4)
    equal(StratifiedKFold(3, shuffle=True).split(9, words, rng),
          StratifiedKFold(3, shuffle=True, random_state=4).split(9, y=words))


def test_typed_list_and_ownership() raises:
    # A bare literal needs two implicit conversions; Mojo allows one, so type the list.
    var labels: List[String] = ["b", "a", "b", "a"]
    var cursor = StratifiedKFold(2).split(4, y=labels)
    assert_true(Bool(cursor.next()))
    var source = floats([0.5, 0.5, 1.5, 1.5, 0.5, 1.5])
    var groups = SplitLabels(source)
    source[Item(0)] = 99.0  # SplitLabels owns its codes
    assert_equal(LeaveOneGroupOut().get_n_splits(groups=groups), 2)
    assert_equal(len(groups), 6)


def test_invalid_inputs() raises:
    with assert_raises(contains="finite"):
        _ = StratifiedKFold(2).split(4, y=floats([1.0, Float64("nan"), 2.0, 2.0]))
    with assert_raises(contains="finite"):
        _ = GroupKFold(2).split(4, groups=floats([1.0, Float64("inf"), 2.0, 3.0]))
    var four: List[String] = ["a", "a", "b", "b"]
    with assert_raises(contains="matching length"):
        _ = StratifiedKFold(2).split(5, y=four)
    with assert_raises(contains="matching length"):
        _ = GroupKFold(2).split(5, groups=four)
    # NDArray(Shape) storage is uninitialized; fill every matrix explicitly.
    var matrix = NDArray[DType.float64](Shape(4, 2))
    var single = NDArray[DType.float32](Shape(4, 1))
    var multilabel = NDArray[DType.float64](Shape(4, 2))
    for i in range(4):
        single[Item(i, 0)] = Float32(i % 2)
        for j in range(2):
            matrix[Item(i, j)] = Float64(i)
            multilabel[Item(i, j)] = Float64((i + j) % 2)
    with assert_raises(contains="groups must be rank one"):
        _ = GroupKFold(2).split(4, groups=matrix)
    with assert_raises(contains="rank-two Float32"):
        _ = StratifiedShuffleSplit(1, test_size=SplitSize(2), random_state=0).split(4, y=single)
    with assert_raises(contains="Supported target types"):
        _ = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=0).split(4, y=multilabel)
    with assert_raises(contains="stratify must have one entry per sample"):
        var two: List[String] = ["a", "b"]
        _ = train_test_split(floats([0, 1, 2, 3]), random_state=0, stratify=two)


def test_check_cv_dispatch() raises:
    var words: List[String] = ["x", "y", "x", "y", "x", "y"]
    assert_true(check_cv(3, words, classifier=True).stratified)
    assert_false(check_cv(3, words, classifier=False).stratified)
    # Continuous targets fall back to KFold, as sklearn's type_of_target dispatch does.
    assert_false(check_cv(3, floats([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), classifier=True).stratified)
    with assert_raises(contains="one entry per sample"):
        _ = check_cv(3, floats([0.1, 0.2, 0.3]), classifier=True).split(6, y=floats([0.1, 0.2, 0.3]))


def test_train_test_split_strings() raises:
    var X = floats([0, 1, 2, 3, 4, 5, 6, 7])
    var y: List[String] = ["cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog"]
    var parts = train_test_split(X, y, random_state=0, test_size=SplitSize(0.5), stratify=y)
    var X_train = parts[0].copy()
    var y_train = parts[2].copy()
    var y_test = parts[3].copy()
    assert_equal(len(y_train), 4)
    assert_equal(len(y_test), 4)
    var cats = 0
    for i in range(len(y_train)):
        # Rows and labels stay aligned: even X values are cats.
        var row = Int(X_train[Item(i)])
        assert_equal(y_train[i], y[row])
        if y_train[i] == "cat": cats += 1
    assert_equal(cats, 2)


def main() raises:
    test_handoff_reproducer()
    test_strings_match_int64_codes()
    test_typed_list_and_ownership()
    test_invalid_inputs()
    test_check_cv_dispatch()
    test_train_test_split_strings()
    print("PASS String/Float labels, groups, errors, ownership and train_test_split")
