from std.testing import assert_equal, assert_raises, assert_false
from moxie.compose._plumbing import (ColumnSelection, BlockLayout, feature_name, validate_feature_names, validate_target_mapping)


def main() raises:
    extension_probes()
    var selection = ColumnSelection([2, -3, 2], 3)
    assert_equal(selection.indices[1], 0)
    var rest = selection.remainder(3)
    assert_equal(len(rest.indices), 1)
    assert_equal(rest.indices[0], 1)
    var copied = selection.copy()
    copied.indices[0] = 1
    assert_equal(selection.indices[0], 2)
    var scalar = ColumnSelection([-1], 3, True)
    assert_equal(scalar.scalar, True)
    var empty = ColumnSelection(List[Int](), 0)
    assert_equal(len(empty.indices), 0)
    with assert_raises():
        var bad = ColumnSelection([3], 3)
    with assert_raises():
        var bad = ColumnSelection([-4], 3)
    with assert_raises():
        var bad = ColumnSelection([0, 1], 3, True)
    with assert_raises():
        var bad = selection.remainder(1)
    var layout = BlockLayout(2)
    layout.append(2, 3)
    layout.append(2, 0, False)
    layout.append(2, 2)
    assert_equal(layout.width, 5)
    assert_equal(layout.spans[1].stop, 0)
    assert_equal(layout.spans[2].start, 3)
    with assert_raises():
        layout.append(1, 4)
    with assert_raises():
        layout.append(2, -1)
    with assert_raises():
        layout.append(2, 1, False)
    assert_equal(layout.width, 5)
    assert_equal(feature_name("scale", "x0"), "scale__x0")
    assert_equal(feature_name("scale", "x0", False), "x0")
    validate_feature_names(["a", "b"])
    with assert_raises():
        validate_feature_names(["a", "a"])
    validate_target_mapping(False, False, False)
    validate_target_mapping(True, False, False)
    validate_target_mapping(False, True, True)
    with assert_raises():
        validate_target_mapping(True, True, True)
    with assert_raises():
        validate_target_mapping(False, True, False)
    with assert_raises():
        validate_target_mapping(False, False, True)
    print("PASS: compose metadata contracts")

# Concrete capability probes raise; numerical concatenation is a child engine.
from numojo import NDArray, Shape, ones
from moxie.compose import ColumnResolver, FeatureNamer, DenseBlockAssembler

struct Positional(ColumnResolver):
    def __init__(out self):
        pass
    comptime Schema = Int
    def resolve(self, schema: Int) raises -> ColumnSelection:
        return ColumnSelection([-1], schema, True)

struct Prefix(FeatureNamer):
    def __init__(out self):
        pass
    def name(self, transformer: String, feature: String) raises -> String:
        return feature_name(transformer, feature)

struct Blocks(DenseBlockAssembler):
    def __init__(out self):
        pass
    def concatenate[dtype: DType](self, blocks: List[NDArray[dtype]], layout: BlockLayout) raises -> NDArray[dtype]:
        raise Error("Contract probe: concatenation engine not implemented")


def extension_probes() raises:
    var resolver = Positional()
    var resolved = resolver.resolve(4)
    assert_equal(resolved.indices[0], 3)
    var naming = Prefix()
    assert_equal(naming.name("a", "b"), "a__b")
    var blocks = Blocks()
    var layout = BlockLayout(2)
    var data = List[NDArray[DType.float64]]()
    with assert_raises():
        var result = blocks.concatenate(data, layout)
