from moxie._labels import StringTarget, encode_column, type_of_target, check_classification_targets, owned_target
from numojo import NDArray, Shape, Item
import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises


def numeric_contracts[dtype: DType]() raises:
    var y = nm.zeros[dtype](Shape(3, 2))
    y[Item(0, 0)] = 9
    y[Item(1, 0)] = 2
    y[Item(2, 0)] = 9
    y[Item(0, 1)] = 3
    y[Item(1, 1)] = 4
    y[Item(2, 1)] = 5
    assert_equal(type_of_target(y), "multiclass-multioutput")
    var encoded = encode_column(y)
    var second = encode_column(y, 1)
    assert_equal(encoded.classes.size, 2)
    assert_equal(second.classes.size, 3)
    assert_equal(encoded.index.counts[Item(1)], Int64(2))
    assert_equal(encoded.index.first_indices[Item(0)], Int64(1))
    assert_equal(encoded.index.appearance_order[Item(0)], Int64(1))
    var first_seen = encoded.index.first_seen_codes()
    assert_equal(first_seen[Item(0)], Int64(0))
    assert_equal(first_seen[Item(1)], Int64(1))
    var decoded = encoded.decode(encoded.index.inverse)
    assert_equal(decoded[Item(0)], Scalar[dtype](9))
    var snapshot = owned_target(y)
    var cloned = encoded.copy()
    y[Item(0, 0)] = 100
    assert_equal(snapshot[Item(0, 0)], Scalar[dtype](9))
    assert_equal(encoded.classes[Item(1)], Scalar[dtype](9))
    encoded.classes[Item(1)] = 10
    assert_equal(cloned.classes[Item(1)], Scalar[dtype](9))
    with assert_raises():
        _ = cloned.encode(y)
    with assert_raises():
        _ = encode_column(y, -1)
    with assert_raises():
        _ = encode_column(y, 2)
    var codes = nm.zeros[DType.int64](Shape(1))
    codes[Item(0)] = -1
    with assert_raises():
        _ = cloned.decode(codes)
    codes[Item(0)] = 2
    with assert_raises():
        _ = cloned.decode(codes)
    var bad_rank = nm.zeros[DType.int64](Shape(1, 1))
    with assert_raises():
        _ = cloned.decode(bad_rank)
    var rank3 = nm.zeros[dtype](Shape(1, 1, 1))
    assert_equal(type_of_target(rank3), "unknown")
    with assert_raises():
        _ = owned_target(rank3)
    # Real strided storage, with negative row stride and nonzero offset.
    var reversed = snapshot.view_with_layout(Shape(3, 2), nm.NDArrayStrides(-2, 1), 4)
    var reverse_copy = owned_target(reversed)
    assert_equal(reverse_copy[Item(0, 1)], Scalar[dtype](5))
    var reverse_encoding = encode_column(reversed)
    assert_equal(reverse_encoding.index.inverse[Item(0)], Int64(1))
    snapshot[Item(2, 0)] = 77
    assert_equal(reverse_copy[Item(0, 0)], Scalar[dtype](9))
    var ids = nm.zeros[DType.int64](Shape(3))
    ids[Item(0)] = 1
    ids[Item(1)] = 0
    ids[Item(2)] = 1
    var reversed_ids = ids.view_with_layout(Shape(3), nm.NDArrayStrides(-1), 2)
    var recovered = cloned.decode(reversed_ids)
    assert_equal(recovered[Item(0)], Scalar[dtype](9))
    comptime if dtype != DType.int64:
        var fraction = nm.zeros[dtype](Shape(2))
        fraction[Item(0)] = 1.25
        fraction[Item(1)] = 1.5
        assert_equal(type_of_target(fraction), "continuous")
        with assert_raises():
            _ = encode_column(fraction)
        var groups = encode_column(fraction, classification=False)
        assert_equal(groups.classes.size, 2)
        var infinity = Scalar[dtype](Float64.MAX * 2)
        fraction[Item(0)] = infinity - infinity
        with assert_raises():
            _ = encode_column(fraction, classification=False)
        fraction[Item(0)] = infinity
        with assert_raises():
            _ = type_of_target(fraction)
    print("numeric contracts passed")


def main() raises:
    numeric_contracts[DType.int64]()
    numeric_contracts[DType.float32]()
    numeric_contracts[DType.float64]()
    var values: List[String] = ["z", "a", "z"]
    var y = StringTarget(values)
    values[0] = "changed"
    assert_equal(y.get(0), "z")
    var encoding = encode_column(y)
    assert_equal(encoding.classes[0], "a")
    assert_equal(encoding.classes[1], "z")
    var restored = encoding.decode(encoding.index.inverse)
    assert_equal(restored.get(0), "z")
    var copied = y.copy()
    y.set(0, "other")
    assert_equal(copied.get(0), "z")
    assert_equal(restored.get(0), "z")
    with assert_raises():
        _ = encoding.encode(y)
    var matrix = StringTarget(["z", "one", "a", "two"], 2, 2)
    assert_equal(type_of_target(matrix), "multiclass-multioutput")
    var left = encode_column(matrix)
    var right = encode_column(matrix, 1)
    assert_equal(left.classes[0], "a")
    assert_equal(right.classes[0], "one")
    with assert_raises():
        _ = StringTarget(["a"], 2, 2)
    with assert_raises():
        _ = StringTarget(["a"], -1, 1)
    with assert_raises():
        _ = matrix.get(2)
    with assert_raises():
        matrix.set(-1, "x")
    var empty = StringTarget(List[String]())
    var empty_encoding = encode_column(empty)
    assert_equal(len(empty_encoding.classes), 0)
    var empty_codes = empty_encoding.encode(empty)
    var empty_labels = empty_encoding.decode(empty_codes)
    assert_equal(empty_labels.rows(), 0)
    var invalid = nm.zeros[DType.int64](Shape(1))
    with assert_raises():
        _ = empty_encoding.decode(invalid)
    print("PASS string ownership, domains and invalid input contracts")
