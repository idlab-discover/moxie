"""Measure a mean baseline as each fold's training subset grows."""
from validation_curve import RegressionAdapter, R2Scorer, Dense
from moxie.dummy import DummyRegressor as _DummyRegressor
from moxie.model_selection import learning_curve, KFold, IndexArray, IndexSplit
from numojo import Shape, Item, zeros
comptime DummyRegressor = _DummyRegressor[]  # Bind default parameters to the concrete estimator.


def main() raises:
    var X = zeros[DType.float64](Shape(12, 1))
    var y = Dense(Shape(12))
    for i in range(12):
        y[Item(i)] = Float64(i * i)
    var splits = List[IndexSplit]()
    var cursor = KFold(3).split(12)
    while True:
        var pair = cursor.next()
        if not pair:
            break
        splits.append(pair.value().copy())
    var sizes = IndexArray(Shape(3))
    sizes[Item(0)] = 2
    sizes[Item(1)] = 4
    sizes[Item(2)] = 8
    var result = learning_curve[RegressionAdapter, DType.int64](
        DummyRegressor(), X, y, RegressionAdapter(), R2Scorer(), splits,
        None, None, train_sizes=sizes.copy(), shuffle=True, random_state=42,
        return_times=True,
    )
    for size in range(result.train_sizes.size):
        for fold in range(len(splits)):
            print("training rows", result.train_sizes[Item(size)], "fold", fold,
                  "train R2", result.train_scores[Item(size, fold)],
                  "test R2", result.test_scores[Item(size, fold)],
                  "fit seconds", result.fit_times.value()[Item(size, fold)])
