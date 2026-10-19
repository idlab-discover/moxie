from moxie.tree import consumer_a, consumer_b
from std.testing import assert_equal, assert_true
from moxie.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import numojo as nm

def main() raises:
    assert_equal(consumer_a(), 8.0)
    assert_equal(consumer_b(), 8.0)
    var X = nm.zeros[DType.float32](nm.Shape(4, 1))
    var y = nm.zeros[DType.float64](nm.Shape(4))
    for i in range(4):
        X.unsafe_set(i, Float32(i))
        y.unsafe_set(i, Float64(i >= 2))
    var model = DecisionTreeClassifier(random_state=0)
    _ = model.fit(X, y)
    assert_equal(model.get_n_leaves(), 2)
    assert_true('x[0] <= 1.5' in export_graphviz(model))
    assert_true('x[0] &lt;= 1.5' in plot_tree(model))
    print("Both consumers fit and export real trees with shared dense and RNG implementations")
