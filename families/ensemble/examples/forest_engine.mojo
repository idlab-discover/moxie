"""Grow a random forest regressor with the shared forest engine.

The engine is what RandomForest*/ExtraTrees* build on: it validates and encodes
X once, draws one seed and one bootstrap per tree from the random state, and
builds each tree from bootstrap counts passed as sample weights, so rows are
never copied. Configure the member template like sklearn's forest does for its
trees (here RandomForestRegressor's default max_features=1.0).
"""
import numojo as nm
from moxie._random import RandomState
from moxie.tree import DecisionTreeRegressor
from moxie.tree._classes import FeatureCount
from moxie.ensemble._forest_engine import ForestEngine

def main() raises:
    var n = 200
    var X = nm.zeros[DType.float32](nm.Shape(n, 3))
    var y = nm.zeros[DType.float64](nm.Shape(n))
    var noise = RandomState(0)
    for i in range(n):
        for j in range(3):
            X.unsafe_set(i * 3 + j, Float32(noise.uniform(-2, 2)))
        y.unsafe_set(i, Float64(X.unsafe_get(i * 3)) ** 2 + 0.5 * Float64(X.unsafe_get(i * 3 + 1)) + noise.normal(0, 0.1))
    var template = DecisionTreeRegressor(max_features=FeatureCount(Float64(1)), min_samples_leaf=2)
    var forest = ForestEngine[False](template^, 50, oob_score=True)
    var random_state = RandomState(42)  # sklearn random_state=42
    forest.fit(X, y, None, random_state)
    print("Members:", len(forest), "OOB R^2:", forest.oob_score_)
    var importances = forest.feature_importances_()
    print("Importances:", importances.unsafe_get(0), importances.unsafe_get(1), importances.unsafe_get(2))
    var query = nm.zeros[DType.float32](nm.Shape(1, 3))
    query.unsafe_set(0, 1.5)
    print("Prediction at x0=1.5:", forest.predict(query).unsafe_get(0), "(true mean 2.25)")
    print("In-bag rows of member 0:", forest.estimators_samples_()[0].size, "with repeats")
