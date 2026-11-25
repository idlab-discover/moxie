"""Use the public regression metrics in weighted model-selection scorers."""
from std.testing import assert_false

from moxie.dummy import DummyRegressor
from moxie.metrics import r2_score, root_mean_squared_error
from moxie.model_selection import CVAdapter, ErrorPolicy, IndexArray, IndexSplit
from moxie.model_selection import KFold, TypedScorer, cross_val_score, select_rows
from numojo import NDArray, Shape

comptime Dense = NDArray[DType.float64]
comptime Weights = Optional[Dense]
comptime Regressor = DummyRegressor[DType.float64]


@fieldwise_init
struct RegressionScorer(TypedScorer):
    var use_rmse: Bool

    comptime Estimator = Regressor
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = Weights

    def score(
        self, estimator: Regressor, X: Dense, y: Dense, metadata: Weights,
    ) raises -> Float64:
        var prediction = estimator.predict(X)
        if self.use_rmse:
            return -root_mean_squared_error(y, prediction, sample_weight=metadata)
        return r2_score(y, prediction, sample_weight=metadata)


@fieldwise_init
struct Adapter(CVAdapter):
    comptime Scorer = RegressionScorer

    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise Error("ValueError: incompatible feature and target shapes")
        return y.shape[0]

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return select_rows(X, rows)

    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return select_rows(y, rows)

    def metadata_rows(self, metadata: Weights, rows: IndexArray) raises -> Weights:
        if metadata:
            return select_rows(metadata.value(), rows)
        return None

    def fit(self, mut estimator: Regressor, X: Dense, y: Dense, metadata: Weights) raises:
        if metadata:
            _ = estimator.fit(X, y, sample_weight=metadata.value())
        else:
            _ = estimator.fit(X, y)


def main() raises:
    var X = Dense(Shape(12, 1))
    var y = Dense(Shape(12))
    var sw = Dense(Shape(12))
    for i in range(12):
        X.unsafe_set(i, Float64(i))
        y.unsafe_set(i, Float64((i * i + i) % 13))
        sw.unsafe_set(i, Float64(i % 5 + 1))
    var model = Regressor()
    var splits = List[IndexSplit]()
    var cursor = KFold(3).split(12)
    while True:
        var split = cursor.next()
        if not split:
            break
        splits.append(split.take())

    for use_rmse in [False, True]:
        var scores = cross_val_score[Adapter](
            model, X, y, Adapter(), RegressionScorer(use_rmse), splits, sw.view(), sw.view(),
            error_score=ErrorPolicy(True, 0),
        )
        assert_false(model.__sklearn_is_fitted__())
        print(
            "neg_rmse" if use_rmse else "r2",
            scores.unsafe_get(0), scores.unsafe_get(1), scores.unsafe_get(2),
        )
