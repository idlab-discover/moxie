"""Typed heterogeneous/seed protocol probes; no estimator algorithms."""
from moxie.base import clone
from moxie._random import RandomState
from moxie.ensemble._engine_api import SeededEstimator, EstimatorBank, ProbabilityBank, DecisionBank
from moxie.ensemble._records import Values, NamedEstimator, ClassAlignment
from moxie.ensemble._plumbing import next_estimator_seed
from std.testing import assert_equal, assert_false, assert_raises
from numojo import full, Shape

@fieldwise_init
struct BankParams(Copyable, Movable):
    var first_seed: Int
    var second_seed: Int

struct Bank(ProbabilityBank, DecisionBank, SeededEstimator):
    comptime Params = BankParams
    comptime Input = Values
    comptime Target = Values
    var params: BankParams
    var fitted: Bool
    def __init__(out self):
        self.params = BankParams(0, 0)
        self.fitted = False
    def get_params(self, deep: Bool = True) raises -> BankParams:
        return self.params.copy()
    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: BankParams
    ) raises -> ref[self] Self:
        self.params = params.copy()
        return self
    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.params = self.params.copy()
        return result^
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted
    def set_random_states(mut self, mut rng: RandomState) raises:
        self.params.first_seed = next_estimator_seed(rng)
        self.params.second_seed = next_estimator_seed(rng)
    def fit_members(mut self, X: Values, y: Values, weights: Optional[Values]) raises:
        raise Error("Scaffold: fit not implemented")
    def member_count(self) -> Int:
        return 0
    def member_predict(self, index: Int, X: Values) raises -> Values:
        raise Error("Scaffold: prediction not implemented")
    def member_predict_proba(self, index: Int, X: Values) raises -> List[Values]:
        raise Error("Scaffold: probability not implemented")
    def class_alignment(self, index: Int, output: Int) raises -> ClassAlignment:
        raise Error("Scaffold: class alignment not implemented")
    def member_decision_function(self, index: Int, X: Values) raises -> Values:
        raise Error("Scaffold: decision not implemented")

def seed[T: SeededEstimator](mut estimator: T, mut rng: RandomState) raises:
    estimator.set_random_states(rng)

def bank[T: EstimatorBank](mut estimator: T, X: T.Input, y: T.Target) raises:
    with assert_raises():
        estimator.fit_members(X, y, None)
    with assert_raises():
        _ = estimator.member_predict(0, X)

def probability[T: ProbabilityBank](estimator: T, X: T.Input) raises:
    with assert_raises():
        _ = estimator.member_predict_proba(0, X)

def decision[T: DecisionBank](estimator: T, X: T.Input) raises:
    with assert_raises():
        _ = estimator.member_decision_function(0, X)

def main() raises:
    var estimator = Bank()
    var rng = RandomState(42)
    seed(estimator, rng)
    assert_equal(estimator.params.first_seed, 1608637542)
    assert_equal(estimator.params.second_seed, 1273642419)
    estimator.fitted = True
    var cloned = clone(estimator)
    assert_false(cloned.__sklearn_is_fitted__())
    estimator.params.first_seed = 7
    assert_equal(cloned.params.first_seed, 1608637542)
    var named = NamedEstimator[Bank]("bank", cloned^)
    assert_equal(named.name, "bank")
    named.estimator = None
    assert_false(Bool(named.estimator))
    # set_params returns a borrow of the same estimator, so calls chain.
    assert_equal(estimator.set_params(BankParams(3, 4)).get_params().second_seed, 4)
    assert_equal(estimator.params.first_seed, 3)
    var X = full[DType.float64](Shape(2, 1), 1)
    bank(estimator, X, X)
    probability(estimator, X)
    decision(estimator, X)
    print("PASS typed bank/drop, seeded clone and optional capability contracts")
