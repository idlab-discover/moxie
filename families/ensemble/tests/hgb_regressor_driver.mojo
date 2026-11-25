from std.memory import bitcast
from std.sys import argv
from std.utils import Variant

from moxie.ensemble import HistGradientBoostingRegressor
from numojo import NDArray, Shape

from hgb_losses_driver import read_values


comptime _Values = NDArray[DType.float64]


def _run_case[origin: ImmOrigin](fields: List[StringSpan[origin]]) raises:
    var loss = String(fields[0])
    var quantile = bitcast[DType.float64](Int64(atol(String(fields[1]))))
    var rows = atol(String(fields[2]))
    var cols = atol(String(fields[3]))
    var queries = atol(String(fields[4]))
    var iterations = atol(String(fields[5]))
    var weighted = atol(String(fields[6])) != 0
    var mode = atol(String(fields[7]))
    var constraints = atol(String(fields[8]))
    var sampled = atol(String(fields[9])) != 0
    var seed = atol(String(fields[10]))
    var initial = atol(String(fields[11]))
    var categories = atol(String(fields[12]))
    var scoring = String(fields[13])
    var offset = 14

    var X = _Values(Shape(rows, cols))
    var query = _Values(Shape(queries, cols))
    var y = _Values(Shape(rows))
    read_values(fields, offset, X)
    read_values(fields, offset, query)
    read_values(fields, offset, y)
    var weights: Optional[_Values] = None
    if weighted:
        var w = _Values(Shape(rows))
        read_values(fields, offset, w)
        weights = w^

    var model = HistGradientBoostingRegressor(
        loss, quantile=quantile, learning_rate=0.125, max_iter=iterations, max_leaf_nodes=5,
        max_depth=3, min_samples_leaf=3, l2_regularization=0.25, max_bins=16,
        random_state=seed, max_features=0.6 if sampled else 1.0, n_iter_no_change=3,
        tol=0.125,
    )
    var params = model.get_params()
    params.scoring = None if scoring == "none" else Optional[String](scoring^)
    if categories:
        var indices = List[Int]()
        for col in range(cols):
            if categories & (1 << col):
                indices.append(col)
        params.categorical_features = indices^
    if mode != 4:
        params.early_stopping = Variant[Bool, String](mode != 0)
    if mode == 1:
        params.validation_fraction = None
    elif mode == 5:
        params.validation_fraction = Variant[Int, Float64](9)
    if constraints == 1:
        params.monotonic_cst = List[Int]([1, 1, 0, 0])
        var groups: List[List[Int]] = [[0, 1], [2, 3]]
        params.interaction_cst = Variant[String, List[List[Int]]](groups^)
    elif constraints == 2:
        params.interaction_cst = Variant[String, List[List[Int]]](String("pairwise"))
    elif constraints == 3:
        params.interaction_cst = Variant[String, List[List[Int]]](String("no_interactions"))
    if initial > 0:
        params.warm_start = True
        params.max_iter = initial
        params.max_depth = 2
    _ = model.set_params(params)

    var y_val = _Values(Shape(queries))
    var w_val = _Values(Shape(queries))
    if mode == 3:
        read_values(fields, offset, y_val)
        read_values(fields, offset, w_val)
    for pass_index in range(2 if initial > 0 else 1):
        if pass_index == 1:
            params.max_iter = iterations
            params.max_depth = 3
            params.random_state = seed + 1
            _ = model.set_params(params)
        if mode == 3:
            _ = model.fit(
                X, y, weights, X_val=query.view(), y_val=y_val.view(),
                sample_weight_val=w_val.view(),
            )
        else:
            _ = model.fit(X, y, weights)

    print(model.n_iter_, model.n_features_in_, model.n_trees_per_iteration_,
        Int(model.do_early_stopping_), Int(Bool(model.is_categorical_)),
        model.train_score_.size, model.validation_score_.size)
    if model.is_categorical_:
        for selected in model.is_categorical_.value():
            print(Int(selected))
    print(bitcast[DType.int64](model.score(X, y, weights)))
    for i in range(model.train_score_.size):
        print(bitcast[DType.int64](model.train_score_.unsafe_get(i)))
    for i in range(model.validation_score_.size):
        print(bitcast[DType.int64](model.validation_score_.unsafe_get(i)))
    for stage in model.staged_predict(query):
        for i in range(queries):
            print(bitcast[DType.int64](stage.unsafe_get(i)))
    var prediction = model.predict(query)
    for i in range(queries):
        print(bitcast[DType.int64](prediction.unsafe_get(i)))


def main() raises:
    for line in open(argv()[1], "r").read().split("\n"):
        if line.byte_length() != 0:
            _run_case(line.split(" "))
