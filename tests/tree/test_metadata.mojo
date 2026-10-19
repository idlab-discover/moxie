import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie._categorical import categorical_column
from moxie._labels import StringTarget
from moxie.tree import TreeTable, TreeColumn, DecisionTreeClassifier
from moxie.tree._classes import BaseDecisionTree, CategoricalFeatures

def check[classifier: Bool, extra: Bool]() raises:
    var numeric = nm.zeros[DType.float32](nm.Shape(6))
    var cells: List[Optional[String]] = [String("a"), String("b"), String("a"), String("b"), String("a"), String("b")]
    var names: List[String] = ["height", "color"]
    var table = TreeTable([TreeColumn(numeric.copy()), TreeColumn(categorical_column(cells))], feature_names=names.copy())
    names[0] = "mutated"
    var y = nm.zeros[DType.float64](nm.Shape(6))
    for i in range(6):
        y.unsafe_set(i, Float64(i % 2))
    var indices: List[Int] = [1]
    var mask: List[Bool] = [False, True]
    var selected_names: List[String] = ["color"]
    var options: List[CategoricalFeatures] = [CategoricalFeatures(indices^), CategoricalFeatures(mask^), CategoricalFeatures(selected_names^), CategoricalFeatures("from_dtype")]
    for option in options:
        var model = BaseDecisionTree[classifier, extra](random_state=0, categorical_features=option.copy())
        _ = model.fit(table, y)
        assert_equal(model.feature_names_in_.value()[0], "height")
        assert_true(model.is_categorical_.value()[1])
        assert_equal(model.score(table, y), 1.0)
        var clone = model.__sklearn_clone__()
        assert_true(not clone.feature_names_in_)
        _ = clone.fit(table, y)
        assert_equal(clone.score(table, y), 1.0)
        var wrong = TreeTable(table._columns, feature_names=Optional[List[String]](["color", "height"]))
        with assert_raises():
            _ = model.predict(wrong)
        with assert_raises():
            _ = model.apply(wrong)
        with assert_raises():
            _ = model.score(wrong, y)
        comptime if classifier:
            with assert_raises():
                _ = model.predict_proba(wrong)
            with assert_raises():
                _ = model.predict_log_proba(wrong)
        with assert_raises():
            _ = model.fit(table, nm.zeros[DType.float64](nm.Shape(2)))
        assert_equal(model.feature_names_in_.value()[0], "height")
        model.params.categorical_features = NoneType()
        _ = model.fit(nm.zeros[DType.float32](nm.Shape(6, 2)), y)
        assert_true(not model.feature_names_in_)
        assert_true(not model.is_categorical_)
    var bad = BaseDecisionTree[classifier, extra](categorical_features=List[Bool]([True]))
    with assert_raises():
        _ = bad.fit(table, y)
    bad.params.categorical_features = List[String](["absent"])
    with assert_raises():
        _ = bad.fit(table, y)
    bad.params.categorical_features = List[String](["color"])
    with assert_raises():
        _ = bad.fit(nm.zeros[DType.float32](nm.Shape(6, 2)), y)
    bad.params.categorical_features = "unsupported"
    with assert_raises():
        _ = bad.fit(table, y)
    bad.params.categorical_features = "from_dtype"
    _ = bad.fit(nm.zeros[DType.float32](nm.Shape(6, 2)), y)
    assert_true(not bad.is_categorical_)
    var numeric_table = TreeTable([TreeColumn(numeric.copy()), TreeColumn(numeric.copy())], categorical_mask=Optional[List[Bool]]([True, False]))
    _ = bad.fit(numeric_table, y)
    assert_true(bad.is_categorical_.value()[0])
    assert_true(not bad.is_categorical_.value()[1])

def main() raises:
    check[True, False]()
    check[True, True]()
    check[False, False]()
    check[False, True]()
    var cells: List[Optional[String]] = [String("a"), String("b")]
    var table = TreeTable([TreeColumn(categorical_column(cells))], feature_names=Optional[List[String]](["color"]))
    var model = DecisionTreeClassifier[string_labels=True](categorical_features="from_dtype", random_state=0)
    var y = StringTarget(["no", "yes"])
    _ = model.fit(table, y)
    assert_equal(model.predict(table).get(0), "no")
    assert_equal(model.score(table, y), 1.0)
    print("PASS names/order, selectors, dtype metadata, ownership, refit and typed mixed tables")
