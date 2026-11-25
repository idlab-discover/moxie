"""Tune a real DummyRegressor with the native typed search API."""
from moxie.dummy import DummyRegressor as _DummyRegressor
from moxie.model_selection import (
    DenseSupervisedAdapter, GridSearchCV, ParameterGrid, RandomizedSearchCV,
    cross_val_score,
)
from numojo import NDArray, Shape, Item
comptime DummyRegressor = _DummyRegressor[]  # Bind default parameters to the concrete estimator.

comptime Dense = NDArray[DType.float64]

comptime RegressionAdapter = DenseSupervisedAdapter[DummyRegressor]


def configure(mut estimator: DummyRegressor, params: Dict[String, Float64]) raises:
    for key in params.keys():
        if key != "constant":
            raise Error("ValueError: this adapter tunes only constant")
    if "constant" in params:
        var constant = Dense(Shape(1))
        constant[Item(0)] = params["constant"]
        _ = estimator.set_params(constant=Optional[Dense](constant^))


def main() raises:
    var X = Dense(Shape(10, 1))
    var y = Dense(Shape(10))
    for i in range(10):
        X[Item(i, 0)] = Float64(i)
        y[Item(i)] = Float64(i)
    var grid = Dict[String, List[Float64]]()
    grid["constant"] = [0.0, 2.0, 4.0, 5.0, 9.0]
    var model = DummyRegressor(strategy="constant")
    var adapter = RegressionAdapter(configure)
    var baseline = cross_val_score[RegressionAdapter](DummyRegressor(), X, y, adapter)
    print("baseline mean-score folds:", baseline.size)
    var search = GridSearchCV[RegressionAdapter](model, grid, adapter)
    _ = search.fit(X, y)
    print("grid winner:", search.best_params_()["constant"], "mean CV R2:", search.best_score_())
    var random = RandomizedSearchCV[RegressionAdapter](model, ParameterGrid[Float64](grid), adapter, n_iter=3, random_state=42)
    _ = random.fit(X, y)
    print("random winner:", random.best_params_()["constant"], "mean CV R2:", random.best_score_())
    print("refitted prediction:", random.predict(X)[Item(0)])
