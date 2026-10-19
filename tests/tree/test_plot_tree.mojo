import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, plot_tree, export_graphviz, TreeTable, TreeColumn, StringColumn
from moxie._labels import StringTarget
from moxie._categorical import categorical_column


def main() raises:
    var X = nm.zeros[DType.float32](nm.Shape(4, 1))
    var y = nm.zeros[DType.float64](nm.Shape(4))
    for i in range(4):
        X.unsafe_set(i, Float32(i))
        y.unsafe_set(i, Float64(i >= 2))
    var model = DecisionTreeClassifier(random_state=0)
    with assert_raises():
        _ = plot_tree(model)
    _ = model.fit(X, y)
    var svg = plot_tree(model, filled=True, rounded=True)
    assert_true('<svg xmlns="http://www.w3.org/2000/svg"' in svg)
    assert_true('data-node-id="2"' in svg)
    assert_true('data-parent="0" data-child="1"' in svg)
    assert_true('fill="#e58139"' in svg)
    assert_true('fill="#399de5"' in svg)
    assert_true('x[0] &lt;= 1.5' in svg)
    assert_equal(svg, plot_tree(model, filled=True, rounded=True))
    var names: Optional[List[String]] = Optional[List[String]](['A<&" B'])
    var classes: Optional[List[String]] = Optional[List[String]](['<script>&"', 'ok'])
    var escaped = plot_tree(model, feature_names=names, class_names=classes)
    assert_true('A&lt;&amp;&quot; B' in escaped)
    assert_true('&lt;script&gt;&amp;&quot;' in escaped)
    assert_true('<script>' not in escaped)
    assert_true('class = x[0]' in export_graphviz(model, class_names=Optional[List[String]](["x[0]", "x[1]"]), special_characters=True))
    assert_true('a\\\\b &lt;=' in plot_tree(model, feature_names=Optional[List[String]](["a\\\\b"])))
    var html = export_graphviz(model, feature_names=names, class_names=classes, special_characters=True)
    assert_true('A&lt;&amp;&quot; B &le;' in html)
    var colors: Optional[List[String]] = Optional[List[String]](["#FF0000", "#0000FF"])
    assert_true('fill="#ff0000"' in plot_tree(model, filled=True, fill_colors=colors))
    assert_true('fillcolor="#0000ff"' in export_graphviz(model, filled=True, fill_colors=colors))
    var truncated = plot_tree(model, max_depth=0)
    assert_true('fill="#c0c0c0"' in truncated)
    assert_true('>(...)</text>' in truncated)
    with assert_raises():
        _ = plot_tree(model, fontsize=Float64(0))
    with assert_raises():
        _ = plot_tree(model, fontsize=Float64("nan"))
    with assert_raises():
        _ = plot_tree(model, fontsize=Float64("inf"))
    with assert_raises():
        _ = plot_tree(model, fontsize=Float64(1e308))
    with assert_raises():
        _ = plot_tree(model, precision=-1)
    with assert_raises():
        _ = plot_tree(model, max_depth=-1)
    with assert_raises():
        _ = plot_tree(model, label="invalid")
    with assert_raises():
        _ = plot_tree(model, feature_names=Optional[List[String]]([]))
    with assert_raises():
        _ = plot_tree(model, class_names=Optional[List[String]]([]))
    with assert_raises():
        _ = plot_tree(model, filled=True, fill_colors=Optional[List[String]](["red", "blue"]))
    with assert_raises():
        _ = export_graphviz(model, filled=True, fill_colors=Optional[List[String]](["#gggggg", "#000000"]))
    with assert_raises():
        _ = plot_tree(model, filled=True, fill_colors=Optional[List[String]]([]))
    var categorical = ExtraTreeClassifier(random_state=0, categorical_features=Optional[List[Int]]([0]))
    _ = categorical.fit(X, y)
    assert_true(' in {' in plot_tree(categorical))
    assert_true('missing/unknown -&gt;' in plot_tree(categorical))
    assert_true(' in {' in export_graphviz(categorical))
    var categories: List[Optional[String]] = [String("red<&"), String("red<&"), String("blue"), String("blue")]
    var table = TreeTable([TreeColumn(categorical_column(categories))])
    var mixed = DecisionTreeClassifier(random_state=0, categorical_features=Optional[List[Int]]([0]))
    _ = mixed.fit(table, y)
    var mixed_svg = plot_tree(mixed, filled=True)
    assert_true(' in {' in mixed_svg)
    assert_true('missing/unknown -&gt;' in mixed_svg)
    assert_true('red&lt;&amp;' in mixed_svg or 'blue' in mixed_svg)
    with open("builds/tree-categorical.svg", "w") as output:
        output.write(mixed_svg)
    with open("builds/tree-categorical.dot", "w") as output:
        output.write(export_graphviz(mixed, filled=True, special_characters=True))
    var labels = StringTarget(["a", "a", "b", "b"])
    var typed = DecisionTreeClassifier[string_labels=True](random_state=0)
    _ = typed.fit(X, labels)
    assert_equal(plot_tree(typed), plot_tree(model))
    var constant = nm.zeros[DType.float64](nm.Shape(4, 2))
    var regression = DecisionTreeRegressor(random_state=0)
    _ = regression.fit(X, constant)
    assert_true('fill="#ffffff"' in plot_tree(regression, filled=True))
    assert_true('fillcolor="#ffffff"' in export_graphviz(regression, filled=True))
    with open("builds/tree-preview.svg", "w") as output:
        output.write(svg)
    with open("builds/tree-escaped.svg", "w") as output:
        output.write(escaped)
    with open("builds/tree-escaped.dot", "w") as output:
        output.write(html)
    print("PASS native SVG, escaping, color validation, categorical rules and constant multioutput colors")
