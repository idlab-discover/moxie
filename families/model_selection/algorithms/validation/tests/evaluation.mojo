from moxie.dummy import DummyRegressor as _DummyRegressor
from moxie.model_selection import TypedScorer, ParameterAdapter, DefaultCVAdapter, PredictionAdapter, cross_validate, cross_val_score, cross_val_predict, KFold, StratifiedKFold, SplitLabels, IndexArray, IndexSplit, ErrorPolicy
from moxie.model_selection._split import _rows
from numojo import NDArray, Shape, Item, zeros
from std.testing import assert_equal, assert_true, assert_false, assert_raises
comptime DummyRegressor = _DummyRegressor[]  # Bind default parameters to the concrete estimator.

comptime Dense = NDArray[DType.float64]

@fieldwise_init
struct Scorer(TypedScorer):
    comptime Estimator = DummyRegressor
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = Optional[Dense]
    var fail: Bool
    def score(self, estimator: DummyRegressor, X: Dense, y: Dense, metadata: Self.Metadata) raises -> Float64:
        if self.fail:
            raise Error("intentional scoring failure")
        return estimator.score(X, y, metadata)

struct Adapter(PredictionAdapter, DefaultCVAdapter, ParameterAdapter):
    comptime Scorer = Scorer
    comptime Value = String
    comptime PredictionDType = DType.float64
    var stratified: Bool
    var n_splits: Int
    def __init__(out self, stratified: Bool = False, *, n_splits: Int = 4):
        self.stratified = stratified
        self.n_splits = n_splits
    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise Error("ValueError: invalid dense shapes")
        return y.shape[0]
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)
    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return _rows(y, rows)
    def metadata_rows(self, metadata: Optional[Dense], rows: IndexArray) raises -> Optional[Dense]:
        if metadata:
            return _rows(metadata.value(), rows)
        return None
    def validate_metadata(self, metadata: Optional[Dense], n_samples: Int) raises:
        if metadata:
            ref weights = metadata.value()
            if weights.ndim != 1 or weights.shape[0] != n_samples:
                raise Error("ValueError: expected one weight per sample")
    def fit(self, mut estimator: DummyRegressor, X: Dense, y: Dense, metadata: Optional[Dense]) raises:
        if metadata:
            _ = estimator.fit(X, y, sample_weight=metadata.value())
        else:
            _ = estimator.fit(X, y)
    def configure(self, mut estimator: DummyRegressor, params: Dict[String, String]) raises:
        for key in params:
            if key != "strategy":
                raise Error("ValueError: unknown parameter")
        _ = estimator.set_params(strategy=params["strategy"].copy())
    def predict(self, estimator: DummyRegressor, X: Dense) raises -> Dense:
        return estimator.predict(X)
    def default_scorer(self) -> Scorer:
        return Scorer(False)
    def default_cv(self, X: Dense, y: Dense) raises -> List[IndexSplit]:
        var splits = List[IndexSplit]()
        var n = self.n_samples(X, y)
        var cursor = (
            StratifiedKFold(2).split(n, y=SplitLabels(y))
            if self.stratified else KFold(self.n_splits).split(n)
        )
        while True:
            var pair = cursor.next()
            if not pair:
                return splits^
            splits.append(pair.take())


def main() raises:
    var X = zeros[DType.float64](Shape(8, 2))
    var y = Dense(Shape(8))
    for i in range(8):
        y[Item(i)] = Float64(i)
    var estimator = DummyRegressor()
    var splits = Adapter().default_cv(X, y)
    var scorers: List[Scorer] = [Scorer(False), Scorer(True)]
    var result = cross_validate[Adapter](
        estimator, X, y, Adapter(), scorers^, ["r2", "bad"],
        return_train_score=True, return_estimator=True, return_indices=True,
        error_score=ErrorPolicy(False, -999),
    )
    assert_false(estimator.__sklearn_is_fitted__())
    assert_equal(len(result.estimators.value()), 4)
    assert_equal(len(result.indices.value()), 4)
    for fold in range(4):
        assert_true(result.estimators.value()[fold].__sklearn_is_fitted__())
        assert_equal(result.scores.test_scores[Item(0, fold, 1)], -999.0)
        assert_true(result.status[fold].message.byte_length() > 0)
        assert_true(result.status[fold].fit_time >= 0)
    var scores = cross_val_score[Adapter](estimator, X, y, Adapter(), Scorer(False), splits, None, None)
    var prediction = cross_val_predict[Adapter](estimator, X, y, Adapter(), splits, None)
    assert_equal(prediction.size, 8)
    for fold in range(4):
        var expected: Float64 = (28.0 - Float64(4 * fold + 1)) / 6.0
        assert_equal(prediction[Item(2 * fold)], expected)
        assert_equal(scores[Item(fold)], result.scores.test_scores[Item(0, fold, 0)])
        print("score", fold, scores[Item(fold)])
    for row in range(8):
        print("prediction", row, prediction[Item(row)])
    with assert_raises():
        _ = cross_val_score[Adapter](estimator, X, y, Adapter(), Scorer(True), splits, None, None, error_score=ErrorPolicy(True, 0))
    with assert_raises(contains="intentional scoring failure"):
        _ = cross_val_score[Adapter](
            estimator, X, y, Adapter(), scoring=Scorer(True), error_score=ErrorPolicy(True, 0),
        )
    var weights = Dense(Shape(8))
    for i in range(8):
        weights[Item(i)] = Float64(i + 1)
    var five_fold = Adapter(n_splits=5)
    var default_prediction = cross_val_predict[Adapter](estimator, X, y, five_fold)
    var weighted_default = cross_val_predict[Adapter](
        estimator, X, y, five_fold, fit_metadata=weights.copy(),
    )
    var weighted_prediction = cross_val_predict[Adapter](
        estimator, X, y, five_fold, cv=Adapter().default_cv(X, y), fit_metadata=weights.copy(),
    )
    assert_equal(default_prediction.ndim, 1)
    assert_equal(weighted_default.ndim, 1)
    assert_equal(default_prediction.size, 8)
    assert_equal(weighted_default.size, 8)
    for row in range(8):
        print("prediction_default", row, default_prediction[Item(row)])
        print("prediction_weighted_default", row, weighted_default[Item(row)])
    assert_false(estimator.__sklearn_is_fitted__())
    with assert_raises(contains="one weight per sample"):
        _ = cross_val_predict[Adapter](estimator, X, y, Adapter(), fit_metadata=Dense(Shape(9)))
    with assert_raises(contains="at least one split"):
        _ = cross_val_predict[Adapter](estimator, X, y, Adapter(), cv=List[IndexSplit]())
    for fold in range(4):
        var total: Float64 = 0
        var mass: Float64 = 0
        for i in range(8):
            if i // 2 != fold:
                total += Float64(i * (i + 1))
                mass += Float64(i + 1)
        assert_equal(weighted_prediction[Item(2 * fold)], total / mass)
    var invalid = DummyRegressor(strategy="invalid")
    with assert_raises():
        _ = cross_val_score[Adapter](invalid, X, y, Adapter(), Scorer(False), splits, None, None)
    splits.append(splits[0].copy())
    with assert_raises():
        _ = cross_val_predict[Adapter](estimator, X, y, Adapter(), splits, None)
    print("PASS evaluation lifecycle, real estimator, score errors and partition checks")
