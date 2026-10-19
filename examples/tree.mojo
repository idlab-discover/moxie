"""Run with: pixi run mojo run -I builds/deps examples/decision_tree.mojo"""
import numojo as nm
from moxie.tree import DecisionTreeClassifier, DecisionTreeRegressor

def main() raises:
    var X = nm.zeros[DType.float32](nm.Shape(4, 1))
    var y = nm.zeros[DType.float64](nm.Shape(4))
    for i in range(4):
        X.unsafe_set(i, Float32(i))
        y.unsafe_set(i, Float64(i // 2))
    var classifier = DecisionTreeClassifier(random_state=7)
    _ = classifier.fit(X, y)
    var probabilities = classifier.predict_proba(X)
    print("Accuracy:", classifier.score(X, y))
    print("P(class 0 | first row):", probabilities[0].unsafe_get(0))
    print("Leaves:", classifier.get_n_leaves())
    var regressor = DecisionTreeRegressor(random_state=7, max_depth=1)
    _ = regressor.fit(X, y)
    print("R2:", regressor.score(X, y))
