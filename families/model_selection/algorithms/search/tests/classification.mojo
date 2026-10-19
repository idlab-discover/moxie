from moxie.dummy import DummyClassifier as _DummyClassifier, Label
from moxie.model_selection import SearchScorer, SearchAdapter, GridSearchCV, RandomizedSearchCV, ParameterGrid, IndexArray, IndexSplit, StratifiedKFold
from moxie.model_selection._split import _rows
from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_true
comptime DummyClassifier = _DummyClassifier[multioutput=True]  # Row-wise label targets, as before the Moxie merge.

comptime Dense = NDArray[DType.float64]
comptime Labels = List[List[Label]]

struct Accuracy(SearchScorer):
    comptime Estimator = DummyClassifier
    comptime Input = Dense
    comptime Target = Labels
    comptime Metadata = Optional[Dense]
    def __init__(out self):
        pass
    def score(self, estimator: DummyClassifier, X: Dense, y: Labels, metadata: Self.Metadata) raises -> Float64:
        return estimator.score(X, y, metadata)

struct ClassificationAdapter(SearchAdapter):
    comptime Scorer = Accuracy
    comptime Value = String
    def __init__(out self):
        pass
    def n_samples(self, X: Dense, y: Labels) raises -> Int:
        if X.ndim != 2 or X.shape[0] != len(y):
            raise Error("ValueError: mismatched classification rows")
        return len(y)
    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return _rows(X, rows)
    def target_rows(self, y: Labels, rows: IndexArray) raises -> Labels:
        var result = Labels()
        for i in range(rows.size):
            result.append(y[Int(rows[Item(i)])].copy())
        return result^
    def metadata_rows(self, metadata: Optional[Dense], rows: IndexArray) raises -> Optional[Dense]:
        if metadata:
            return _rows(metadata.value(), rows)
        return None
    def fit(self, mut estimator: DummyClassifier, X: Dense, y: Labels, metadata: Optional[Dense]) raises:
        if metadata:
            _ = estimator.fit(X, y, sample_weight=metadata.value())
        else:
            _ = estimator.fit(X, y)
    def configure(self, mut estimator: DummyClassifier, params: Dict[String, String]) raises:
        for key in params.keys():
            if key != "strategy":
                raise Error("ValueError: unknown parameter")
        if "strategy" in params:
            _ = estimator.set_params(strategy=params["strategy"].copy())
    def default_scorer(self) raises -> Accuracy:
        return Accuracy()
    def default_cv(self, X: Dense, y: Labels) raises -> List[IndexSplit]:
        var labels = IndexArray(Shape(self.n_samples(X, y)))
        for i in range(len(y)):
            labels[Item(i)] = Int64(y[i][0].integer)
        var cursor = StratifiedKFold().split(len(y), labels^)
        var result = List[IndexSplit]()
        while True:
            var pair = cursor.next()
            if not pair:
                break
            result.append(pair.value().copy())
        return result^


def main() raises:
    var X = Dense(Shape(15, 2))
    var y = Labels()
    for i in range(15):
        X[Item(i, 0)] = Float64(i)
        X[Item(i, 1)] = Float64(i % 2)
        y.append(List[Label]([Label(10 if i < 10 else 20)]))
    var params = Dict[String, List[String]]()
    params["strategy"] = ["most_frequent", "prior", "uniform", "stratified"]
    var model = DummyClassifier(random_state=42)
    var search = GridSearchCV[ClassificationAdapter](model, params, ClassificationAdapter(), return_train_score=True)
    search.fit(X, y, None, None)
    assert_equal(search.best_index_(), 0)
    assert_equal(search.n_splits_(), 5)
    assert_equal(search.best_params_()["strategy"], "most_frequent")
    assert_equal(search.predict(X)[0][0].integer, 10)
    var probabilities = search.predict_proba(X)
    assert_equal(probabilities[0][Item(0, 0)], 1.0)
    assert_equal(probabilities[0][Item(0, 1)], 0.0)
    var logs = search.predict_log_proba(X)
    assert_equal(logs[0][Item(0, 0)], 0.0)
    var result = search.cv_results_()
    for c in range(4):
        print("class_mean", c, result.mean_test_score[Item(c, 0)])
        print("class_rank", c, result.rank_test_score[Item(c, 0)])
        for f in range(5):
            print("class_test", c, f, result.scores.test_scores[Item(c, f, 0)])
    var random = RandomizedSearchCV[ClassificationAdapter](model, ParameterGrid[String](params), ClassificationAdapter(), n_iter=3, random_state=42)
    random.fit(X, y, None, None)
    assert_equal(random.best_params_()["strategy"], "prior")
    print("PASS classification, stratified defaults, string candidates and probability delegation")
