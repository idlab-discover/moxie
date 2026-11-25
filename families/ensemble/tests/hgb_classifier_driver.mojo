from std.memory import bitcast
from std.sys import argv
from std.utils import Variant

from moxie.ensemble import HistGradientBoostingClassifier
from numojo import NDArray, Shape

from hgb_losses_driver import read_values


comptime _Values = NDArray[DType.float64]


def _emit(values: _Values):
    for i in range(values.size):
        print(bitcast[DType.int64](values.unsafe_get(i)))


def _run_case[origin: ImmOrigin](fields: List[StringSpan[origin]]) raises:
    var rows = atol(String(fields[0]))
    var cols = atol(String(fields[1]))
    var queries = atol(String(fields[2]))
    var classes = atol(String(fields[3]))
    var weighted = atol(String(fields[4])) != 0
    var class_weight = atol(String(fields[5]))
    var mode = atol(String(fields[6]))
    var constraints = atol(String(fields[7]))
    var sampled = atol(String(fields[8])) != 0
    var seed = atol(String(fields[9]))
    var initial = atol(String(fields[10]))
    var categories = atol(String(fields[11]))
    var scoring = atol(String(fields[12]))
    var offset = 13
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

    var model = HistGradientBoostingClassifier(
        learning_rate=0.125,
        max_iter=9,
        max_leaf_nodes=5,
        max_depth=3,
        min_samples_leaf=3,
        l2_regularization=0.25,
        max_bins=16,
        random_state=seed,
        max_features=0.65 if sampled else 1.0,
        n_iter_no_change=3,
        tol=0.02,
    )
    var params = model.get_params()
    var scorers: List[Optional[String]] = [
        String("loss"), None, String("accuracy"), String("balanced_accuracy"),
        String("neg_log_loss"), String("neg_brier_score"),
    ]
    params.scoring = scorers[scoring].copy()
    if categories:
        var mask = List[Bool](capacity=cols)
        for col in range(cols):
            mask.append(Bool(categories & (1 << col)))
        params.categorical_features = mask^
    if mode != 5:
        params.early_stopping = Variant[Bool, String](mode != 0)
    if mode == 1:
        params.validation_fraction = None
    elif mode == 4:
        params.validation_fraction = Variant[Int, Float64](12)
    if constraints == 1:
        params.monotonic_cst = List[Int]([1, 0, 0, 0])
    elif constraints == 2:
        params.interaction_cst = Variant[String, List[List[Int]]](
            String("pairwise")
        )
    if class_weight == 1:
        params.class_weight = String("balanced")
    elif class_weight == 2:
        var factors = Dict[Int64, Float64]()
        factors[0] = 2.0
        if classes > 1:
            factors[1] = 0.5
        params.class_weight = factors^
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
            params.max_iter = 9
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

    print(
        model.n_iter_,
        model.n_features_in_,
        model.n_trees_per_iteration_,
        Int(model.do_early_stopping_),
        Int(Bool(model.is_categorical_)),
        model.classes_.size,
        model.train_score_.size,
        model.validation_score_.size,
    )
    if model.is_categorical_:
        for selected in model.is_categorical_.value():
            print(Int(selected))
    print(bitcast[DType.int64](model.score(X, y, weights)))
    _emit(model.classes_)
    _emit(model.train_score_)
    _emit(model.validation_score_)
    for stage in model.staged_predict(query):
        _emit(stage)
    for stage in model.staged_predict_proba(query):
        _emit(stage)
    for stage in model.staged_decision_function(query):
        _emit(stage)
    _emit(model.predict(query))
    _emit(model.predict_proba(query))
    _emit(model.decision_function(query))


def main() raises:
    for line in open(argv()[1], "r").read().split("\n"):
        if line.byte_length() != 0:
            _run_case(line.split(" "))
