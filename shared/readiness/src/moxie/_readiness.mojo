"""Small dense integration workload, not an estimator or reusable encoder."""
from moxie._random import RandomState
from moxie._dense import sum_axis2d, row_squared_norms
from numojo import NDArray, Shape, Item
import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises

def workload(n: Int, strategy: String, weighted: Bool, singleton: Bool, mut state: RandomState) raises:
    # Vocabulary is fixture data. Oracle independently obtains it with np.unique.
    var numbers = [Int(-7), 2, 11]
    var names = [String("amsterdam"), "paris", "東京"]
    var y_numbers = [Int(11), -7, 2, 11, -7, 2]
    var y_names = [String("東京"), "amsterdam", "paris", "東京", "amsterdam", "paris"]
    var weights = nm.zeros[DType.float64](Shape(6))
    for i in range(6):
        weights[Item(i)] = 1.0
        if weighted:
            weights[Item(i)] = Float64(i % 4) / 2.0
    for output in range(2):
        var classes = 3 - output
        if singleton:
            classes = 1
        var contributions = nm.zeros[DType.float64](Shape(6, classes))
        for i in range(6):
            var index = 0
            for j in range(3):
                if output == 0:
                    if y_numbers[i] == numbers[j]:
                        index = j
                else:
                    if y_names[i] == names[j]:
                        index = j
            index = index % classes
            contributions[Item(i, index)] = weights.item(i)
        var counts = sum_axis2d(contributions, 0)
        var p = nm.zeros[DType.float64](Shape(classes))
        var total = Float64(0)
        for j in range(classes):
            total += counts.item(j)
        for j in range(classes):
            p[Item(j)] = counts.item(j) / total
            if output == 0:
                print("CLASS", output, numbers[j])
            else:
                print("CLASS", output, names[j])
            print("COUNT", output, counts.item(j))
            print("PRIOR", output, p.item(j))
        var draws = nm.zeros[DType.int64](Shape(n, classes))
        if strategy == "stratified":
            draws = state.multinomial(1, p, n)
        for i in range(n):
            var selected = 0
            if strategy == "uniform":
                selected = state.randint(classes)
            elif strategy == "stratified":
                for j in range(classes):
                    if draws.item(i,j) == 1:
                        selected = j
            else:
                raise Error("ValueError: probe strategy must be uniform or stratified")
            print("INDEX", output, selected)
            if output == 0:
                print("LABEL", output, numbers[selected])
            else:
                print("LABEL", output, names[selected])
            if strategy == "stratified":
                for j in range(classes):
                    print("DRAW", output, draws.item(i,j))
    print("POSITION", state.state_position())
    for j in range(624):
        print("WORD", state.state_word(j))

def dense_contract[dtype: DType]() raises:
    var a = nm.zeros[dtype](Shape(2, 3))
    for i in range(2):
        for j in range(3):
            a[Item(i,j)] = Scalar[dtype](i * 3 + j)
    var copied = a.copy()
    var view = a.view()
    view[Item(0,0)] = 10
    assert_equal(a.item(0,0), 10)
    assert_equal(copied.item(0,0), 0)
    var norms = row_squared_norms(copied)
    assert_equal(norms.item(0), 5)
    assert_equal(norms.item(1), 50)
    var sums = sum_axis2d(copied, -2)
    assert_equal(sums.item(0), 3)
    assert_equal(sums.item(2), 7)
    var empty = nm.zeros[dtype](Shape(0, 3))
    var reduced = sum_axis2d(empty, 0)
    assert_equal(reduced.size, 3)
    assert_equal(reduced.item(0), 0)
    with assert_raises():
        _ = sum_axis2d(copied, 2)
    var vector = nm.zeros[dtype](Shape(3))
    with assert_raises():
        _ = row_squared_norms(vector)

def contracts() raises:
    dense_contract[DType.float32]()
    dense_contract[DType.float64]()
    var state = RandomState(42)
    _ = state.standard_normal()
    var snapshot = state.copy()
    assert_equal(state.standard_normal(), snapshot.standard_normal())
    _ = state.next_uint32()
    assert_true(state.state_position() != snapshot.state_position())
    var repeated = RandomState(42)
    var independent = RandomState(42)
    assert_equal(repeated.random_sample(), independent.random_sample())
    with assert_raises():
        _ = RandomState(-1)
    var bad = nm.zeros[DType.float64](Shape(2))
    with assert_raises():
        _ = state.categorical(bad)
    with assert_raises():
        _ = state.randint(0)
    with assert_raises():
        _ = state.multinomial(1, bad, -1)
