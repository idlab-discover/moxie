import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from moxie.tree._export import _dot_number
from moxie._labels import StringTarget


def main() raises:
    var X = nm.zeros[DType.float32](nm.Shape(4, 1))
    var y = nm.zeros[DType.float64](nm.Shape(4))
    for i in range(4):
        X.unsafe_set(i, Float32(i))
        y.unsafe_set(i, Float64(i >= 2))
    var model = DecisionTreeClassifier(random_state=0)
    with assert_raises():
        _ = export_graphviz(model)
    _ = model.fit(X, y)
    var expected = String('digraph Tree {\nnode [shape=box, fontname="helvetica"] ;\nedge [fontname="helvetica"] ;\n')
    expected += '0 [label="x[0] <= 1.5\\ngini = 0.5\\nsamples = 4\\nvalue = [2, 2]"] ;\n'
    expected += '1 [label="gini = 0.0\\nsamples = 2\\nvalue = [2, 0]"] ;\n'
    expected += '0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n'
    expected += '2 [label="gini = 0.0\\nsamples = 2\\nvalue = [0, 2]"] ;\n'
    expected += '0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;\n}'
    assert_equal(export_graphviz(model), expected)
    assert_true('label="(...)"' in export_graphviz(model, max_depth=0))
    assert_true('class = no' in export_graphviz(model, class_names=Optional[List[String]](["no", "yes"])))
    assert_true('a\\"b <= 1.5' in export_graphviz(model, feature_names=Optional[List[String]](['a"b'])))
    with assert_raises():
        _ = export_graphviz(model, max_depth=-1)
    with assert_raises():
        _ = export_graphviz(model, precision=-1)
    with assert_raises():
        _ = export_graphviz(model, label="bad")
    with assert_raises():
        _ = export_graphviz(model, feature_names=Optional[List[String]]([]))
    with assert_raises():
        _ = export_graphviz(model, class_names=Optional[List[String]]([]))
    assert_true('fillcolor="#e58139"' in export_graphviz(model, filled=True))
    assert_true("{rank=same ; 1; 2}" in export_graphviz(model, leaves_parallel=True))
    assert_true("x<SUB>0</SUB> &le; 1.5" in export_graphviz(model, special_characters=True))
    var categorical = DecisionTreeClassifier(random_state=0, categorical_features=Optional[List[Int]]([0]))
    _ = categorical.fit(X, y)
    assert_true(" in {" in export_graphviz(categorical))
    assert_equal(_dot_number(2.675, 2), "2.67")
    assert_equal(_dot_number(-0.0, 3), "-0.0")
    assert_equal(_dot_number(0.00001, 5), "0.00001")
    assert_equal(_dot_number(1.5, 0), "2.0")
    var regressor = DecisionTreeRegressor(random_state=0)
    _ = regressor.fit(X, y)
    assert_equal(export_graphviz(regressor, class_names=Optional[List[String]]([])), export_graphviz(regressor))
    for i in range(4):
        y.unsafe_set(i, 1.3625)
    _ = regressor.fit(X, y)
    assert_true("value = 1.362" in export_graphviz(regressor))
    var strings = StringTarget(["no", "no", "yes", "yes"])
    var typed = DecisionTreeClassifier[string_labels=True](random_state=0)
    _ = typed.fit(X, strings)
    assert_equal(export_graphviz(typed), expected)
    print("PASS DOT export, names, errors and native decimal formatting")
