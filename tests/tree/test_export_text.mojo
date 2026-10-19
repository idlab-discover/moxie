import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.tree import DecisionTreeClassifier, DecisionTreeRegressor, TreeTable, TreeColumn, export_text
from moxie._categorical import categorical_column
from moxie.tree._export import _fixed

def main() raises:
    assert_equal(_fixed(2.675, 2), "2.67")
    assert_equal(_fixed(1.25, 1), "1.2")
    assert_equal(_fixed(1.75, 1), "1.8")
    assert_equal(_fixed(-0.0, 3), "-0.000")
    assert_equal(_fixed(-0.001, 2), "-0.00")
    assert_equal(_fixed(9.999, 2), "10.00")
    var X = nm.zeros[DType.float32](nm.Shape(4, 1))
    var y = nm.zeros[DType.float64](nm.Shape(4))
    for i in range(4):
        X.unsafe_set(i, Float32(i))
        y.unsafe_set(i, Float64(i >= 2))
    var model = DecisionTreeClassifier(random_state=0)
    with assert_raises():
        _ = export_text(model)
    _ = model.fit(X, y)
    var report = export_text(model, class_names=Optional[List[String]](["no", "yes"]), show_weights=True)
    assert_equal(report, "|--- feature_0 <= 1.50\n|   |--- weights: [2.00, 0.00] class: no\n|--- feature_0 >  1.50\n|   |--- weights: [0.00, 2.00] class: yes\n")
    assert_equal(export_text(model, max_depth=0), export_text(model))
    with assert_raises():
        _ = export_text(model, max_depth=-1)
    with assert_raises():
        _ = export_text(model, spacing=0)
    with assert_raises():
        _ = export_text(model, decimals=-1)
    with assert_raises():
        _ = export_text(model, feature_names=Optional[List[String]]([]))
    with assert_raises():
        _ = export_text(model, class_names=Optional[List[String]](["one"]))
    var strings: List[Optional[String]] = [String("blue"), String("blue"), String("red"), String("red")]
    var table = TreeTable([TreeColumn(categorical_column(strings))])
    var cat = DecisionTreeClassifier(categorical_features=Optional[List[Int]]([0]), random_state=0)
    _ = cat.fit(table, y)
    var text = export_text(cat)
    assert_true("in {" in text)
    assert_true("blue" in text and "red" in text and "missing/unknown" in text)
    assert_true(" <= " not in text)
    print("PASS text export, parameter validation, category rules and decimal rounding")
