"""Native table ownership, schema, selection and fitted name alignment contracts."""
from moxie.compose import ColumnTransformer, ColumnEntry, ColumnTable, ColumnKey, make_column_selector, NoTransformer
from moxie.base import clone
from numojo import Shape, zeros
from std.testing import assert_equal, assert_false, assert_raises


def main() raises:
    var X = zeros[DType.float64](Shape(3, 3))
    for i in range(3):
        for j in range(3):
            X.unsafe_set(i * 3 + j, Float64(10 * i + j))
    var names: List[String] = ["age", "income", "unused"]
    var table = ColumnTable(X, names)
    X.unsafe_set(0, 99)
    names[0] = "changed"
    assert_equal(Float64(table.values().unsafe_get(0)), 0.0)
    assert_equal(table.schema().names[0], "age")
    var keys: List[String] = ["income", "age"]
    var model = ColumnTransformer(ColumnEntry("selected", "passthrough", keys.copy()))
    var output = model.fit_transform(table)
    assert_equal(Float64(output.unsafe_get(0)), 1.0)
    assert_equal(model.feature_names_in_.value()[0], "age")
    assert_equal(model.get_feature_names_out()[0], "selected__income")
    var reordered = zeros[DType.float64](Shape(3, 2))
    for i in range(3):
        reordered.unsafe_set(i * 2, Float64(10 * i + 1))
        reordered.unsafe_set(i * 2 + 1, Float64(10 * i))
    var unseen = ColumnTable(reordered, keys)
    var result = model.transform(unseen)
    for i in range(3):
        for j in range(2):
            assert_equal(Float64(result.unsafe_get(i * 2 + j)), Float64(output.unsafe_get(i * 2 + j)))
    var bad: List[String] = ["income", "unknown"]
    with assert_raises():
        _ = model.transform(ColumnTable(reordered, bad))
    var wrong: List[String] = ["wrong", "income", "unused"]
    with assert_raises():
        _ = model.get_feature_names_out(wrong)
    var copied = clone(model)
    assert_false(Bool(copied.feature_names_in_))
    var raw = ColumnTransformer(ColumnEntry("a", "passthrough", [0]))
    raw.fit(table)
    raw.fit(X)
    assert_false(Bool(raw.feature_names_in_))
    var floating: List[String] = ["floating"]
    var float32: List[String] = ["float32"]
    var selector = make_column_selector(dtype_include=floating.copy())
    assert_equal(len(selector(table).names), 3)
    assert_equal(len(make_column_selector(dtype_include=float32.copy())(table).names), 0)
    assert_equal(len(make_column_selector(dtype_exclude=floating.copy())(table).names), 0)
    with assert_raises():
        _ = make_column_selector(dtype_include=floating.copy(), dtype_exclude=floating.copy())(table)
    with assert_raises():
        _ = make_column_selector(pattern="a.*")(table)
    var selected = ColumnTransformer(ColumnEntry("a", "passthrough", selector.copy()))
    assert_equal(selected.fit_transform(table).shape[1], 3)
    var first: List[String] = ["age"]
    var named = ColumnTransformer(ColumnEntry("first", "passthrough", first.copy()), remainder="passthrough")
    named.fit(table)
    assert_equal(named.transformers_()[1].columns.names[0], "income")
    with assert_raises():
        _ = named.transform(unseen)
    var schema = table.schema()
    assert_equal(len(ColumnKey.name_slice("age", "income").indices(schema)), 2)
    assert_equal(len(ColumnKey(Slice(None, None, None)).indices(schema)), 3)
    with assert_raises():
        _ = ColumnKey.name_slice("unused", "age", -1).indices(schema)
    with assert_raises():
        _ = ColumnKey.name_slice("age", "income", 0).indices(schema)
    var duplicates: List[String] = ["age", "age", "unused"]
    with assert_raises():
        _ = ColumnKey("age").indices(ColumnTable(X, duplicates).schema())
    with assert_raises():
        _ = ColumnTable(X, first)
    var f32 = zeros[DType.float32](Shape(2, 3))
    var t32 = ColumnTable(f32, names)
    var m32 = ColumnTransformer(ColumnEntry[NoTransformer[DType.float32], DType.float32]("all", "passthrough", make_column_selector(dtype_include=float32.copy())))
    assert_equal(m32.fit_transform(t32).shape[1], 3)
    var snapshot = table.copy()
    table.values().unsafe_set(0, 123)
    assert_equal(Float64(snapshot.values().unsafe_get(0)), 0.0)
    print("PASS: named numeric tables, dtype selectors and transform alignment")
