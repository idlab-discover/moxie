from moxie._random import RandomState
from moxie.model_selection import SplitLabels, SplitSize, train_test_split
from numojo import Item, NDArray, NDArrayStrides, Shape
from std.testing import assert_equal, assert_raises


def dataset[dtype: DType]() raises -> List[NDArray[dtype]]:
    var arrays = List[NDArray[dtype]]()
    for index in range(4):
        var columns = index + 1
        var data = NDArray[dtype](Shape(12) if index == 0 else Shape(12, columns))
        for row in range(12):
            if index == 0:
                data[Item(row)] = row.cast[dtype]()
            else:
                for column in range(columns):
                    data[Item(row, column)] = Int(100 * index + 10 * row + column).cast[dtype]()
        if index == 0:
            arrays.append(data.view_with_layout(Shape(12), NDArrayStrides(-1), 11))
        else:
            arrays.append(data^)
    return arrays^


def emit[dtype: DType](name: String, parts: List[NDArray[dtype]]) raises:
    assert_equal(len(parts), 8)
    for index in range(len(parts)):
        ref part = parts[index]
        var columns = index // 2 + 1
        assert_equal(part.ndim, 1 if columns == 1 else 2)
        if part.ndim == 2:
            assert_equal(part.shape[1], columns)
        print(name, index, part.ndim, part.shape[0], columns)
        for row in range(part.shape[0]):
            if part.ndim == 1:
                print(Int(part[Item(row)]))
            else:
                for column in range(columns):
                    print(Int(part[Item(row, column)]))


def check_errors() raises:
    comptime Dense = NDArray[DType.float32]
    var arrays = dataset[DType.float32]()
    var rng = RandomState(42)
    with assert_raises(contains="requires at least one array"):
        _ = train_test_split(List[Dense](), rng)
    with assert_raises(contains="requires at least one array"):
        _ = train_test_split(List[Dense]())
    var original = arrays.pop()
    arrays.append(Dense(Shape(12, 1, 1)))
    with assert_raises(contains="rank-one or rank-two"):
        _ = train_test_split(arrays, rng)
    _ = arrays.pop()
    arrays.append(Dense(Shape(11)))
    with assert_raises(contains="inconsistent sample counts"):
        _ = train_test_split(arrays, rng)
    _ = arrays.pop()
    arrays.append(original^)
    var short_groups: List[String] = ["a", "b"]
    with assert_raises(contains="one entry per sample"):
        _ = train_test_split(arrays, rng, stratify=SplitLabels(short_groups))
    var untouched_rng = RandomState(42)
    assert_equal(rng.next_uint32(), untouched_rng.next_uint32())
    var parts = train_test_split(arrays, shuffle=False)
    var original_value = parts[0][Item(0)]
    arrays[0][Item(0)] = -999
    assert_equal(parts[0][Item(0)], original_value)
    parts[0][Item(1)] = -999
    assert_equal(arrays[0][Item(1)], 10)


def main() raises:
    check_errors()
    var single = dataset[DType.float32]()
    var wide = dataset[DType.float64]()
    var indices = dataset[DType.int64]()
    var groups: List[String] = ["z", "h", "z", "q", "h", "q", "h", "z", "q", "h", "q", "z"]
    var rng = RandomState(13)
    emit("ordered", train_test_split(single, shuffle=False, train_size=SplitSize(7), test_size=SplitSize(3)))
    emit("seeded", train_test_split(wide, random_state=42))
    emit("stratified", train_test_split(wide, random_state=42, test_size=SplitSize(0.4), stratify=SplitLabels(groups)))
    emit("stream_first", train_test_split(indices, rng))
    emit("stream_second", train_test_split(indices, rng))
