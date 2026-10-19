"""Lifecycle/capability probe, not an implemented statistical estimator."""
from moxie.base import BaseEstimator, InverseTransformer, DecisionFunction, SampleScorer, clone
from moxie.model_selection import GridSearchCV, SearchScorer, SearchAdapter, KFold, IndexArray, IndexSplit
from std.testing import assert_equal, assert_true

@fieldwise_init
struct Probe(BaseEstimator, InverseTransformer, DecisionFunction, SampleScorer):
    comptime Params = Int
    comptime Input = Int
    comptime Transformed = Int
    comptime Decision = Int
    comptime SampleScores = Int
    var parameter: Int
    var fit_count: Int
    def get_params(self, deep: Bool = True) raises -> Int:
        return self.parameter
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Int) raises -> ref[self] Self:
        self.parameter = params
        return self
    def __sklearn_clone__(self) raises -> Self:
        return Self(self.parameter, 0)
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fit_count > 0
    def transform(self, X: Int) raises -> Int:
        assert_equal(self.fit_count, 1)
        return X + self.parameter
    def inverse_transform(self, X: Int) raises -> Int:
        assert_equal(self.fit_count, 1)
        return X - self.parameter
    def decision_function(self, X: Int) raises -> Int:
        assert_equal(self.fit_count, 1)
        return self.parameter
    def score_samples(self, X: Int) raises -> Int:
        assert_equal(self.fit_count, 1)
        return self.parameter

@fieldwise_init
struct Scorer(SearchScorer):
    comptime Estimator = Probe
    comptime Input = Int
    comptime Target = Int
    comptime Metadata = Int
    var offset: Int
    def score(self, estimator: Probe, X: Int, y: Int, metadata: Int) raises -> Float64:
        assert_equal(estimator.fit_count, 1)
        return Float64(estimator.parameter + self.offset)

struct Adapter(SearchAdapter):
    comptime Value = Int
    comptime Scorer = Scorer
    def __init__(out self):
        pass
    def n_samples(self, X: Int, y: Int) raises -> Int:
        return X
    def input_rows(self, X: Int, rows: IndexArray, train: IndexArray) raises -> Int:
        return rows.size
    def target_rows(self, y: Int, rows: IndexArray) raises -> Int:
        return y
    def metadata_rows(self, metadata: Int, rows: IndexArray) raises -> Int:
        return metadata
    def fit(self, mut estimator: Probe, X: Int, y: Int, metadata: Int) raises:
        assert_equal(estimator.fit_count, 0)
        estimator.fit_count += 1
    def configure(self, mut estimator: Probe, params: Dict[String, Int]) raises:
        if "parameter" in params:
            _ = estimator.set_params(params["parameter"])
    def default_scorer(self) raises -> Scorer:
        return Scorer(0)
    def default_cv(self, X: Int, y: Int) raises -> List[IndexSplit]:
        var cursor = KFold().split(X)
        var splits = List[IndexSplit]()
        while True:
            var pair = cursor.next()
            if not pair:
                break
            splits.append(pair.value().copy())
        return splits^


def main() raises:
    var model = Probe(10, 99)
    var grid = Dict[String, List[Int]]()
    grid["parameter"] = [1, 3, 2]
    var search = GridSearchCV[Adapter](model, grid, Adapter(), scoring=List[Scorer]([Scorer(0), Scorer(1)]), metric_names=List[String](["a", "b"]), refit="a")
    search.fit(10, 0, 0, 0)
    assert_equal(search.best_index_(), 1)
    assert_equal(model.fit_count, 99)
    assert_equal(search.transform(10), 13)
    assert_equal(search.inverse_transform(13), 10)
    assert_equal(search.decision_function(10), 3)
    assert_equal(search.score_samples(10), 3)
    var clone_search = clone(search)
    assert_true(not clone_search.__sklearn_is_fitted__())
    print("PASS public search imports, nonpredictive estimator, one fit per multimetric fold and delegated capabilities")
