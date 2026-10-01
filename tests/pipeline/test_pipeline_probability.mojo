"""Probability dispatch probes and real dense DummyClassifier integration."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import SupervisedEstimator, ProbabilityPredictor, LogProbabilityPredictor, TransformerMixin, FitTransformer, ClassifierMixin, clone
from moxie.pipeline import Pipeline
from moxie.dummy import DummyClassifier, Label
from numojo import NDArray, Shape, zeros
from test_sequential import Shift, Recorder, Fused


struct ProbabilityProbe[logs: Bool = True](SupervisedEstimator, ProbabilityPredictor,
                                         LogProbabilityPredictor where logs):
    comptime Input = Int
    comptime Target = Int
    comptime Prediction = Int
    comptime Probabilities = Int
    comptime Params = Int
    var child: Recorder
    def __init__(out self):
        self.child = Recorder()
    def get_params(self, deep: Bool = True) raises -> Int:
        return 0
    def set_params(mut self, params: Int) raises:
        pass
    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return self.child.fitted
    def fit(mut self, X: Int, y: Int) raises:
        self.child.fit(X, y)
    def predict(self, X: Int) raises -> Int:
        raise Error("probability methods must not call predict")
    def predict_proba(self, X: Int) raises -> Int:
        if X < 0:
            raise Error("ValueError: probability probe input")
        return 100 * X + self.child.seen
    def predict_log_proba(self, X: Int) raises -> Int where Self.logs:
        if X < 0:
            raise Error("ValueError: log probability probe input")
        # Deliberately distinct token: proves direct dispatch, not log(proba).
        return -10 * X - self.child.seen


struct LabelCopy[dtype: DType](TransformerMixin, FitTransformer):
    comptime Input = NDArray[Self.dtype]
    comptime Target = List[Label]
    comptime Transformed = Self.Input
    comptime Params = Int
    var fitted: Bool
    def __init__(out self):
        self.fitted = False
    def get_params(self, deep: Bool = True) raises -> Int:
        return 0
    def set_params(mut self, params: Int) raises:
        pass
    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted
    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self.fitted = True
        return X.copy()
    def transform(self, X: Self.Input) raises -> Self.Transformed:
        if not self.fitted:
            raise Error("NotFittedError: label copy fixture")
        return X.copy()


def through_trait[P: ProbabilityPredictor](pipe: P, X: P.Input) raises -> P.Probabilities:
    return pipe.predict_proba(X)


def through_log_trait[P: LogProbabilityPredictor](pipe: P, X: P.Input) raises -> P.Probabilities:
    return pipe.predict_log_proba(X)


def dense[dtype: DType]() raises:
    var X = zeros[dtype](Shape(4, 2))
    var y: List[Label] = [Label(9), Label(2), Label(9), Label(9)]
    var pipe = Pipeline(LabelCopy[dtype](), DummyClassifier[False, dtype]())
    with assert_raises():
        _ = pipe.predict_proba(X)
    with assert_raises():
        _ = pipe.predict_log_proba(X)
    pipe.fit(X, y)
    var probs = through_trait(pipe, X)
    var logs = through_log_trait(pipe, X)
    comptime assert type_of(probs) == NDArray[DType.float64]
    assert_equal(probs.shape[0], 4)
    assert_equal(probs.shape[1], 2)
    assert_equal(logs.shape[0], 4)
    assert_equal(logs.shape[1], 2)
    assert_equal(pipe.estimator.classes_[0].integer, 2)
    assert_equal(pipe.estimator.classes_[1].integer, 9)
    for i in range(4):
        assert_equal(Float64(probs.item(i, 0)), 0.25)
        assert_equal(Float64(probs.item(i, 1)), 0.75)
    print("prob_dense", Int(dtype == DType.float64), probs.item(0, 0), probs.item(0, 1), logs.item(0, 0), logs.item(0, 1))
    probs.itemset(0, 0)
    assert_equal(Float64(pipe.predict_proba(X).item(0, 0)), 0.25)
    var fresh = clone(pipe)
    with assert_raises():
        _ = fresh.predict_proba(X)
    with assert_raises():
        _ = fresh.predict_log_proba(X)
    var params = fresh.get_params()
    params.estimator.strategy = "most_frequent"
    fresh.set_params(params)
    fresh.fit(X, y)
    var hard = fresh.predict_proba(X)
    var hard_logs = fresh.predict_log_proba(X)
    assert_equal(Float64(hard.item(0, 0)), 0.0)
    assert_equal(Float64(hard.item(0, 1)), 1.0)
    print("prob_hard", Int(dtype == DType.float64), hard.item(0, 0), hard.item(0, 1), hard_logs.item(0, 0), hard_logs.item(0, 1))
    var one: List[Label] = [Label(7), Label(7), Label(7), Label(7)]
    pipe.fit(X, one)
    assert_equal(pipe.predict_proba(X).shape[1], 1)
    assert_equal(Float64(pipe.predict_proba(X).item(0)), 1.0)
    assert_equal(Float64(pipe.predict_log_proba(X).item(0)), 0.0)
    assert_equal(Float64(fresh.predict_proba(X).item(0, 1)), 1.0)


def main() raises:
    comptime assert not conforms_to(Pipeline[Shift, Recorder], ProbabilityPredictor)
    comptime assert conforms_to(Pipeline[Shift, ProbabilityProbe[False]], ProbabilityPredictor)
    comptime assert not conforms_to(Pipeline[Shift, ProbabilityProbe[False]], LogProbabilityPredictor)
    comptime assert not conforms_to(Pipeline[Shift, ProbabilityProbe[True]], ClassifierMixin)
    var pipe = Pipeline(Shift(), ProbabilityProbe())
    with assert_raises():
        _ = pipe.predict_proba(1)
    with assert_raises():
        _ = pipe.predict_log_proba(1)
    pipe.fit(5, 7)
    assert_equal(through_trait(pipe, 1), 307)
    assert_equal(through_log_trait(pipe, 1), -37)
    print("prob_scalar", pipe.predict_proba(1), pipe.predict_log_proba(1))
    with assert_raises():
        _ = pipe.predict_proba(-10)
    with assert_raises():
        _ = pipe.predict_log_proba(-10)
    var fresh = clone(pipe)
    with assert_raises():
        _ = fresh.predict_log_proba(1)
    var params = fresh.get_params()
    params.transformer = 4
    fresh.set_params(params)
    fresh.fit(5, 7)
    pipe.fit(8, 7)
    print("prob_refit", fresh.predict_proba(1), fresh.predict_log_proba(1), pipe.predict_proba(1), pipe.predict_log_proba(1))
    with assert_raises():
        pipe.fit(-1, 7)
    assert_true(pipe.__sklearn_is_fitted__())
    with assert_raises():
        _ = pipe.predict_proba(1)
    with assert_raises():
        _ = pipe.predict_log_proba(1)
    with assert_raises():
        pipe.fit(5, -1)
    assert_false(pipe.__sklearn_is_fitted__())
    with assert_raises():
        _ = pipe.predict_proba(1)
    with assert_raises():
        _ = pipe.predict_log_proba(1)
    var fused = Pipeline(Fused(), ProbabilityProbe())
    fused.fit(5, 7)
    print("prob_fused", fused.predict_proba(1), fused.predict_log_proba(1))
    assert_equal(fused.transformer.calls, 1)
    var nested = Pipeline(Shift(2), Pipeline(Shift(4), ProbabilityProbe()))
    nested.fit(5, 7)
    print("prob_nested", nested.predict_proba(1), nested.predict_log_proba(1))
    var only = Pipeline(Shift(), ProbabilityProbe[False]())
    only.fit(5, 7)
    assert_equal(only.predict_proba(1), 307)
    dense[DType.float32]()
    dense[DType.float64]()
