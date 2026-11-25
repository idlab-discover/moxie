"""Required metadata stays explicit and does not advertise supervised fit."""
from search_support import Adapter, Scorer, DummyRegressor, Dense, data, grid
from moxie.base import SupervisedEstimator, RegressorMixin, clone
from moxie.model_selection import SearchScorer, SearchAdapter, GridSearchCV, RandomizedSearchCV, ParameterGrid, IndexArray, IndexSplit
from std.testing import assert_equal, assert_false

@fieldwise_init
struct RequiredMetadata(Copyable):
    var weights: Optional[Dense]

struct RequiredScorer(SearchScorer):
    comptime Estimator = DummyRegressor
    comptime Input = Dense
    comptime Target = Dense
    comptime Metadata = RequiredMetadata

    def __init__(out self):
        pass

    def score(self, estimator: DummyRegressor, X: Dense, y: Dense, metadata: RequiredMetadata) raises -> Float64:
        return Scorer(0).score(estimator, X, y, metadata.weights)

struct RequiredAdapter(SearchAdapter):
    comptime Scorer = RequiredScorer
    comptime Value = Float64

    def __init__(out self):
        pass

    def n_samples(self, X: Dense, y: Dense) raises -> Int:
        return Adapter().n_samples(X, y)

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return Adapter().input_rows(X, rows, train)

    def target_rows(self, y: Dense, rows: IndexArray) raises -> Dense:
        return Adapter().target_rows(y, rows)

    def metadata_rows(self, metadata: RequiredMetadata, rows: IndexArray) raises -> RequiredMetadata:
        return RequiredMetadata(Adapter().metadata_rows(metadata.weights, rows))

    def fit(self, mut estimator: DummyRegressor, X: Dense, y: Dense, metadata: RequiredMetadata) raises:
        Adapter().fit(estimator, X, y, metadata.weights)

    def configure(self, mut estimator: DummyRegressor, params: Dict[String, Float64]) raises:
        Adapter().configure(estimator, params)

    def default_scorer(self) raises -> RequiredScorer:
        return RequiredScorer()

    def default_cv(self, X: Dense, y: Dense) raises -> List[IndexSplit]:
        return Adapter().default_cv(X, y)


def main() raises:
    comptime assert not conforms_to(RequiredMetadata, Defaultable)
    comptime assert not conforms_to(GridSearchCV[RequiredAdapter], SupervisedEstimator)
    comptime assert not conforms_to(RandomizedSearchCV[RequiredAdapter], SupervisedEstimator)
    comptime assert not conforms_to(GridSearchCV[RequiredAdapter], RegressorMixin)
    var arrays = data()
    var params = grid([4.0])
    var metadata = RequiredMetadata(None)
    var search = GridSearchCV[RequiredAdapter](DummyRegressor(), params, RequiredAdapter())
    assert_equal(search.fit(arrays[0], arrays[1], metadata, metadata).predict(arrays[0]).item(0), 4.0)
    assert_equal(search.score(arrays[0], arrays[1], metadata=metadata), -8.5)
    assert_false(clone(search).__sklearn_is_fitted__())
    var random = RandomizedSearchCV[RequiredAdapter](DummyRegressor(), ParameterGrid[Float64](params), RequiredAdapter(), n_iter=1, random_state=1)
    assert_equal(random.fit(arrays[0], arrays[1], metadata, metadata).predict(arrays[0]).item(0), 4.0)
    print("PASS required metadata search")
