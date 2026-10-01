from moxie._categorical import CategoricalColumn, StringCategories, NumericCategories, categorical_column
from numojo import Shape, Item
import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.math import nan, inf


def strings() raises:
    var values: List[Optional[String]] = [Optional[String]("z"), None, Optional[String](""), Optional[String]("é"), Optional[String]("z")]
    var column = categorical_column(values)
    assert_equal(column.rows(), 5)
    var restored = column.values()
    assert_equal(restored[0].value(), "z")
    assert_false(restored[1])
    restored[0] = None
    assert_equal(column.get(0).value(), "z")
    assert_equal(column.domain().values(), List[String](["", "z", "é"]))
    assert_equal(column.codes()[Item(0)], Int64(1))
    assert_equal(column.codes()[Item(1)], Int64(-1))
    assert_equal(column.get(2).value(), "")
    assert_true(column.missing_mask()[Item(1)])
    assert_false(column.missing_mask()[Item(2)])
    values[0] = Optional[String]("changed")
    assert_equal(column.get(0).value(), "z")
    var cloned = column.copy()
    column.set(0, None)
    assert_equal(cloned.get(0).value(), "z")
    var codes = cloned.codes()
    codes[Item(0)] = -1
    assert_equal(cloned.get(0).value(), "z")
    var domain = cloned.domain().values()
    domain[1] = "changed"
    assert_equal(cloned.get(0).value(), "z")
    with assert_raises():
        cloned.set(0, Optional[String]("unknown"))
    assert_equal(cloned.get(0).value(), "z")
    with assert_raises():
        _ = cloned.get(-1)
    with assert_raises():
        _ = cloned.get(5)
    with assert_raises():
        _ = StringCategories(["x", "x"])
    var explicit_domain = StringCategories(["z", "a", "unused"])
    var explicit_values: List[Optional[String]] = [Optional[String]("a"), None]
    var explicit_column = CategoricalColumn[StringCategories].from_values(explicit_domain, explicit_values)
    assert_equal(explicit_column.codes()[Item(0)], Int64(1))
    var selection = nm.zeros[DType.int64](Shape(3))
    selection[Item(0)] = 1
    selection[Item(1)] = 0
    selection[Item(2)] = 0
    var selected = explicit_column.take(selection)
    assert_false(selected.get(0))
    assert_equal(selected.get(1).value(), "a")
    assert_equal(selected.domain().size(), 3)
    selected.set(1, None)
    assert_equal(explicit_column.get(0).value(), "a")
    selection[Item(0)] = -1
    with assert_raises():
        _ = explicit_column.take(selection)
    var invalid = nm.zeros[DType.int64](Shape(1))
    invalid[Item(0)] = -2
    with assert_raises():
        _ = CategoricalColumn(explicit_domain, invalid)
    invalid[Item(0)] = 3
    with assert_raises():
        _ = CategoricalColumn(explicit_domain, invalid)
    var bad_rank = nm.zeros[DType.int64](Shape(1, 1))
    with assert_raises():
        _ = CategoricalColumn(explicit_domain, bad_rank)
    with assert_raises():
        _ = explicit_column.take(bad_rank)
    var empty = categorical_column(List[Optional[String]]())
    assert_equal(empty.rows(), 0)
    assert_equal(empty.domain().size(), 0)
    var absent: List[Optional[String]] = [None, None]
    var all_missing = categorical_column(absent)
    assert_equal(all_missing.domain().size(), 0)
    assert_true(all_missing.missing_mask()[Item(0)])


def numeric[dtype: DType]() raises:
    var values = nm.zeros[dtype](Shape(4))
    values[Item(0)] = 1
    values[Item(1)] = 0
    values[Item(2)] = 1
    values[Item(3)] = 0
    var mask = nm.zeros[DType.bool](Shape(4))
    mask[Item(3)] = True
    var column = categorical_column(values, Optional[nm.NDArray[DType.bool]](mask.copy()))
    assert_equal(column.domain().size(), 2)
    var restored = column.values()
    assert_equal(restored[0].value(), Scalar[dtype](1))
    assert_false(restored[3])
    assert_equal(column.get(0).value(), Scalar[dtype](1))
    assert_false(column.get(3))
    var reverse = values.view_with_layout(Shape(4), nm.NDArrayStrides(-1), 3)
    var reversed_column = categorical_column(reverse)
    assert_equal(reversed_column.get(0).value(), Scalar[dtype](0))
    values[Item(0)] = 0
    assert_equal(column.get(0).value(), Scalar[dtype](1))
    var domain_copy = column.domain().values()
    domain_copy[Item(1)] = 0
    assert_equal(column.get(0).value(), Scalar[dtype](1))
    var copied = column.copy()
    column.set(0, None)
    assert_equal(copied.get(0).value(), Scalar[dtype](1))
    # Strided input codes and masks must be read in logical row order.
    var codes = copied.codes()
    var reversed_codes = codes.view_with_layout(Shape(4), nm.NDArrayStrides(-1), 3)
    var from_codes = CategoricalColumn(copied.domain(), reversed_codes)
    codes[Item(3)] = 0
    assert_false(from_codes.get(0))
    var reverse_mask = mask.view_with_layout(Shape(4), nm.NDArrayStrides(-1), 3)
    var masked = categorical_column(values, Optional[nm.NDArray[DType.bool]](reverse_mask.copy()))
    assert_false(masked.get(0))
    assert_true(masked.get(3))
    var empty = categorical_column(nm.zeros[dtype](Shape(0)))
    assert_equal(empty.rows(), 0)
    var wrong_mask = nm.zeros[DType.bool](Shape(2))
    with assert_raises():
        _ = categorical_column(values, Optional[nm.NDArray[DType.bool]](wrong_mask.copy()))
    with assert_raises():
        _ = categorical_column(nm.zeros[dtype](Shape(2, 2)))
    with assert_raises():
        _ = NumericCategories(nm.zeros[dtype](Shape(2)))
    var unsorted = nm.zeros[dtype](Shape(2))
    unsorted[Item(0)] = 1
    with assert_raises():
        _ = NumericCategories(unsorted)
    with assert_raises():
        _ = NumericCategories(nm.zeros[dtype](Shape(1, 1)))
    var all_mask = nm.zeros[DType.bool](Shape(4))
    for i in range(4):
        all_mask[Item(i)] = True
    var all_missing = categorical_column(values, Optional[nm.NDArray[DType.bool]](all_mask.copy()))
    assert_equal(all_missing.domain().size(), 0)
    assert_false(all_missing.get(3))


def floating[dtype: DType]() raises:
    var values = nm.zeros[dtype](Shape(3))
    values[Item(0)] = nan[dtype]()
    values[Item(1)] = 0.5
    values[Item(2)] = -2.25
    var column = categorical_column(values)
    assert_false(column.get(0))
    assert_equal(column.get(1).value(), Scalar[dtype](0.5))
    assert_equal(column.domain().get(0), Scalar[dtype](-2.25))
    values[Item(0)] = inf[dtype]()
    with assert_raises():
        _ = categorical_column(values)
    with assert_raises():
        _ = NumericCategories(values)


def main() raises:
    strings()
    numeric[DType.int64]()
    numeric[DType.bool]()
    numeric[DType.float32]()
    numeric[DType.float64]()
    floating[DType.float32]()
    floating[DType.float64]()
    var large = nm.zeros[DType.int64](Shape(2))
    large[Item(0)] = 9007199254740993
    large[Item(1)] = 9007199254740992
    var column = categorical_column(large)
    assert_equal(column.get(0).value(), Int64(9007199254740993))
    assert_equal(column.domain().size(), 2)
    print("categorical contracts passed")
