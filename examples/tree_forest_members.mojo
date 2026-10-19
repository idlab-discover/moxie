"""Run with: `pixi run mojo run -I builds/deps examples/forest_members.mojo`.

Prepare training input once, build several bootstrapped trees from it, encode
the query once and average member predictions, the way a forest uses trees.
"""
import numojo as nm
from moxie._random import RandomState
from moxie.tree import DecisionTreeRegressor
from moxie.tree._classes import FeatureCount

def main() raises:
    var n = 32
    var X = nm.zeros[DType.float32](nm.Shape(n, 2))
    var y = nm.zeros[DType.float64](nm.Shape(n))
    for i in range(n):
        X.unsafe_set(i * 2, Float32(i))
        X.unsafe_set(i * 2 + 1, Float32(i % 3))
        y.unsafe_set(i, Float64(i // 8) + Float64(i % 3) / 10)
    var template = DecisionTreeRegressor(max_features=FeatureCount(Float64(1)))
    var prepared = template.prepare_fit(X, y)  # validation and encoding happen once
    var draws = RandomState(0)
    var members = List[DecisionTreeRegressor]()
    for _ in range(3):
        var counts = nm.zeros[DType.float64](nm.Shape(n))  # one bootstrap sample
        for _ in range(n):
            var row = draws.randint(0, n)
            counts.unsafe_set(row, counts.unsafe_get(row) + 1)
        var tree = template.__sklearn_clone__()
        var params = tree.get_params()
        params.random_state = draws.randint(2147483647)
        _ = tree.set_params(params)
        _ = tree.fit_prepared(prepared, sample_weight=counts)
        members.append(tree^)
    var query = nm.zeros[DType.float32](nm.Shape(2, 2))
    query.unsafe_set(2, 30)
    query.unsafe_set(3, 2)
    var encoded = prepared.encoder.encode(query, False, False)  # once per prediction
    var mean = nm.zeros[DType.float64](nm.Shape(2))
    for m in range(len(members)):
        var prediction = members[m].predict_encoded(encoded)
        for i in range(2):
            mean.unsafe_set(i, mean.unsafe_get(i) + prediction.unsafe_get(i) / 3)
    print("Members:", len(members), "depths:", members[0].get_depth(), members[1].get_depth(), members[2].get_depth())
    print("Mean prediction for x=0:", mean.unsafe_get(0), "x=30:", mean.unsafe_get(1))
