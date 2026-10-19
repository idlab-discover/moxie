"""Compare a mean baseline's observed score with shuffled-target scores."""
from validation_curve import RegressionAdapter, R2Scorer, Dense
from moxie.dummy import DummyRegressor as _DummyRegressor
from moxie.model_selection import permutation_test_score, KFold, IndexSplit
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
    var result = permutation_test_score[RegressionAdapter](
        DummyRegressor(), X, y, RegressionAdapter(), R2Scorer(), splits,
        None, None, n_permutations=19, random_state=42,
    )
    print("observed mean CV R2", result.score, "p-value", result.pvalue)
    for i in range(result.permutation_scores.size):
        print("permutation", i, "mean CV R2", result.permutation_scores[Item(i)])
