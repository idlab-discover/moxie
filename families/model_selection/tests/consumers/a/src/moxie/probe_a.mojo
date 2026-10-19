from std.testing import assert_equal, assert_false, assert_raises
from moxie.base import BaseEstimator
from moxie.model_selection import TypedScorer, FitAdapter, clone_fit_score
from moxie._dense import row_squared_norms
from moxie._random import RandomState
from numojo import NDArray, Shape, full, Item

@fieldwise_init
struct ProbeA(BaseEstimator):
    comptime Params = Int
    var parameter: Int
    var fitted: Bool
    def get_params(self, deep: Bool = True) raises -> Int:
        return self.parameter
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Int) raises -> ref[self] Self:
        self.parameter = params
        return self
    def __sklearn_clone__(self) raises -> Self:
        return Self(self.parameter, False)
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted

@fieldwise_init
struct ScorerA(TypedScorer):
    comptime Estimator = ProbeA
    comptime Input = NDArray[DType.float64]
    comptime Target = Int
    comptime Metadata = Int
    def score(self, estimator: ProbeA, X: Self.Input, y: Int, metadata: Int) raises -> Float64:
        if metadata < 0:
            raise Error("Scorer probe failure")
        assert_equal(estimator.fitted, True)
        return Float64(estimator.parameter + y + metadata)

@fieldwise_init
struct AdapterA(FitAdapter):
    comptime Scorer = ScorerA
    def fit(self, mut estimator: ProbeA, X: NDArray[DType.float64], y: Int, metadata: Int) raises:
        if metadata < 0:
            raise Error("Fit adapter probe failure")
        assert_false(estimator.fitted)
        estimator.fitted = True
        estimator.parameter += y + metadata

def checkA() raises:
    var original = ProbeA(2, True)
    var X = full[DType.float64](Shape(2, 2), 3)
    assert_equal(clone_fit_score[AdapterA](original, AdapterA(), ScorerA(), X, 4, X, 5, 6, 7), 24.0)
    assert_equal(original.parameter, 2)
    assert_equal(original.fitted, True)
    var norms = row_squared_norms(X)
    assert_equal(Float64(norms[Item(0)]), 18.0)
    var stream = RandomState(42)
    assert_equal(stream.next_uint32(), UInt32(1608637542))
    with assert_raises():
        _ = clone_fit_score[AdapterA](original, AdapterA(), ScorerA(), X, 4, X, 5, -1, 7)
    with assert_raises():
        _ = clone_fit_score[AdapterA](original, AdapterA(), ScorerA(), X, 4, X, 5, 6, -1)
    assert_equal(original.parameter, 2)
    _ = original.set_params(3)
    assert_equal(clone_fit_score[AdapterA](original, AdapterA(), ScorerA(), X, 4, X, 5, 6, 7), 25.0)
