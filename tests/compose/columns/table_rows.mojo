"""Row selection preserves table storage, ownership, and category identity."""
from moxie._categorical import NumericCategories, categorical_column
from moxie.compose import ColumnTable
from numojo import NDArrayStrides, Shape, zeros
from std.testing import assert_equal, assert_false, assert_raises


def main() raises:
    var data = zeros[DType.float32](Shape(4, 3))
    for i in range(data.size):
        data.unsafe_set(i, Float32(i))
    var view = data.view_with_layout(Shape(3, 2), NDArrayStrides(-3, -1), 11)
    var table = ColumnTable(view, ["left", "right"])
    var rows = zeros[DType.int64](Shape(3))
    rows.unsafe_set(0, 2)
    rows.unsafe_set(2, 2)
    var dense = table.take_rows(rows)
    assert_equal(dense.values().shape[0], 3)
    assert_equal(dense.values().unsafe_get(0), Float32(5))
    assert_equal(dense.values().unsafe_get(1), Float32(4))
    assert_equal(dense.values().unsafe_get(2), Float32(11))
    assert_equal(dense.values().unsafe_get(4), Float32(5))
    table.values().unsafe_set(0, Float32(99))
    assert_equal(dense.values().unsafe_get(2), Float32(11))

    var empty_matrix = zeros[DType.float64](Shape(3, 0))
    var mixed = ColumnTable(empty_matrix, [])
    var large = Int64(9007199254740993)
    var ids = zeros[DType.int64](Shape(3))
    ids.unsafe_set(0, large)
    ids.unsafe_set(1, large + 1)
    ids.unsafe_set(2, large + 2)
    var missing = zeros[DType.bool](Shape(3))
    missing.unsafe_set(0, True)
    mixed.add_numeric("raw_id", ids)
    mixed.add_categorical("id", categorical_column(ids, missing^), position=0)
    mixed.add_strings("text", ["", "unused", "café"], position=1)
    var selected = mixed.take_rows(rows)
    assert_equal(selected.schema().names, ["id", "text", "raw_id"])
    assert_equal(selected.schema().dtypes, ["category", "string", "int64"])
    assert_equal(selected.string_column(1), ["café", "", "café"])
    comptime IdCategories = NumericCategories[DType.int64]
    var categories = selected.categorical_columns[IdCategories]([0])
    ref category = categories[0]
    assert_equal(category.domain().size(), 2)
    assert_equal(category.domain().get(0), large + 1)
    assert_equal(category.get(0).value(), large + 2)
    assert_false(Bool(category.get(1)))
    assert_equal(category.get(2).value(), large + 2)
    var raw = selected._numeric_categories[DType.int64]([2])
    assert_equal(raw[0].get(1).value(), large)

    var empty = mixed.take_rows(zeros[DType.int64](Shape(0)))
    assert_equal(empty.rows(), 0)
    assert_equal(empty.schema().names, selected.schema().names)
    var empty_categories = empty.categorical_columns[IdCategories]([0])
    assert_equal(empty_categories[0].domain().size(), 2)
    assert_equal(len(empty.string_column(1)), 0)
    var dense_empty = table.take_rows(zeros[DType.int64](Shape(0)))
    assert_equal(dense_empty.values().shape[0], 0)
    assert_equal(dense_empty.values().shape[1], 2)

    with assert_raises(contains="rank one"):
        _ = mixed.take_rows(zeros[DType.int64](Shape(1, 1)))
    rows.unsafe_set(1, -1)
    with assert_raises(contains="out of bounds"):
        _ = mixed.take_rows(rows)
    rows.unsafe_set(1, 3)
    with assert_raises(contains="out of bounds"):
        _ = mixed.take_rows(rows)
    assert_equal(mixed.rows(), 3)
    print("PASS: native ColumnTable row selection")
