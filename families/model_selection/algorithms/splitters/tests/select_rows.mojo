from moxie.model_selection import IndexArray, select_rows
from numojo import Item, NDArray, NDArrayStrides, Shape, arange, array
from std.testing import assert_equal, assert_raises


def check_array[dtype: DType]() raises:
    var source = arange[dtype](12).reshape(Shape(3, 4))
    var view = source.view_with_layout(Shape(3, 2), NDArrayStrides(-4, 2), 8)
    var indices = array[DType.int64]([2, 0, 2], [3])
    var result = select_rows(view, indices)
    assert_equal(result.ndim, 2)
    assert_equal(result.shape[0], 3)
    assert_equal(result.shape[1], 2)
    for i in range(3):
        assert_equal(result.item(i, 0), Scalar[dtype](8 if i == 1 else 0))
        assert_equal(result.item(i, 1), Scalar[dtype](10 if i == 1 else 2))
    result[Item(0, 0)] = 99
    assert_equal(source.item(0, 0), 0)
    assert_equal(result.item(2, 0), 0)

    var empty = select_rows(view, IndexArray(Shape(0)))
    assert_equal(empty.shape[0], 0)
    assert_equal(empty.shape[1], 2)


def check_pairwise_block[dtype: DType]() raises:
    var source = arange[dtype](16).reshape(Shape(4, 4))
    var gram = source.view_with_layout(Shape(3, 3), NDArrayStrides(-4, -1), 15)
    var rows = array[DType.int64]([2, 0, 2], [3])
    var train = array[DType.int64]([1, 0], [2])
    var block = select_rows(gram, rows, columns=train)
    assert_equal(block.shape[0], 3)
    assert_equal(block.shape[1], 2)
    assert_equal(block.item(0, 0), 6)
    assert_equal(block.item(0, 1), 7)
    assert_equal(block.item(1, 0), 14)
    assert_equal(block.item(1, 1), 15)
    assert_equal(block.item(2, 0), 6)
    assert_equal(block.item(2, 1), 7)
    block[Item(0, 0)] = 99
    assert_equal(gram.item(2, 1), 6)

    var empty = IndexArray(Shape(0))
    assert_equal(select_rows(gram, empty, columns=train).shape[0], 0)
    assert_equal(select_rows(gram, rows, columns=empty).shape[1], 0)
    with assert_raises():
        _ = select_rows(gram, rows, columns=array[DType.int64]([3], [1]))
    with assert_raises():
        _ = select_rows(gram, array[DType.int64]([-1], [1]), columns=train)
    with assert_raises():
        _ = select_rows(gram, rows, columns=train.reshape(Shape(1, 2)))
    with assert_raises():
        _ = select_rows(NDArray[dtype](Shape(3, 4)), rows, columns=train)


def main() raises:
    check_array[DType.float32]()
    check_array[DType.float64]()
    check_pairwise_block[DType.float32]()
    check_pairwise_block[DType.float64]()
    var indices = array[DType.int64]([2, 0, 2], [3])
    var vector = array[DType.int64]([4, 7, 9], [3])
    var result = select_rows(vector, indices)
    assert_equal(result.ndim, 1)
    assert_equal(result.item(0), 9)
    assert_equal(result.item(1), 4)
    assert_equal(result.item(2), 9)

    var labels: List[String] = ["low", "mid", "high"]
    var selected = select_rows(labels, indices)
    assert_equal(selected[0], "high")
    assert_equal(selected[1], "low")
    assert_equal(selected[2], "high")
    selected[0] = "other"
    assert_equal(labels[2], "high")
    assert_equal(len(select_rows(labels, IndexArray(Shape(0)))), 0)

    with assert_raises():
        _ = select_rows(vector, array[DType.int64]([-1], [1]))
    with assert_raises():
        _ = select_rows(labels, array[DType.int64]([3], [1]))
    with assert_raises():
        _ = select_rows(vector, indices.reshape(Shape(1, 3)))
    with assert_raises():
        _ = select_rows(NDArray[DType.float64](Shape(1, 2, 3)), indices)
