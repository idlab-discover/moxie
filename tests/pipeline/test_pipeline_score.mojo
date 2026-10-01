"""Score routing probes and real dense DummyRegressor integration."""
from std.testing import assert_equal, assert_false, assert_true, assert_raises
from numojo import Shape, zeros
from moxie.base import SupervisedScorer, Predictor, ClassifierMixin, RegressorMixin, clone
from moxie.pipeline import Pipeline
from moxie.dummy import DummyRegressor
from test_sequential import Shift, Recorder, Fused
from test_real_pipeline import CopyStep


struct ScoreRecorder(SupervisedScorer, Predictor):
    comptime Input = Int
    comptime Target = Int
    comptime Prediction = Int
    comptime SampleWeight = Int
    comptime Params = Int
    var child: Recorder
    def __init__(out self):
        self.child = Recorder()
    def get_params(self, deep: Bool = True) raises -> Int:
        return self.child.get_params(deep)
    def set_params(mut self, params: Int) raises:
        self.child.set_params(params)
    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return self.child.__sklearn_is_fitted__()
    def fit(mut self, X: Int, y: Int) raises:
        self.child.fit(X, y)
    def predict(self, X: Int) raises -> Int:
        raise Error("score must delegate directly, not call predict")
    def score(self, X: Int, y: Int, sample_weight: Optional[Int] = None) raises -> Float64:
        if not self.child.fitted:
            raise Error("NotFittedError: score recorder")
        if y < 0 or (sample_weight and sample_weight.value() < 0):
            raise Error("ValueError: rejected score arguments")
        return Float64(X * 100 + y * 10 + self.child.seen +
                       (sample_weight.value() if sample_weight else -1))


def through_trait[S: SupervisedScorer](estimator: S, X: S.Input, y: S.Target) raises -> Float64 where conforms_to(S.SampleWeight, Deinitable):
    return estimator.score(X, y)


def dense[dtype: DType]() raises:
    var X = zeros[dtype](Shape(4, 2))
    var y = zeros[dtype](Shape(4))
    var target = zeros[dtype](Shape(4))
    var weights = zeros[dtype](Shape(4))
    for i in range(4):
        y.itemset(i, Scalar[dtype](i * 2))
        target.itemset(i, Scalar[dtype](0 if i < 2 else 2))
        weights.itemset(i, Scalar[dtype](1 if i < 2 else 3))
    var pipe = Pipeline(CopyStep[dtype](), DummyRegressor[dtype]())
    with assert_raises():
        _ = pipe.score(X, y)
    pipe.fit(X, y)
    assert_equal(through_trait(pipe, X, y), 0.0)
    assert_equal(pipe.score(X, target), -4.0)
    assert_equal(pipe.score(X, target, sample_weight=weights.copy()), -3.0)
    assert_equal(Float64(weights.item(0)), 1.0)
    assert_equal(Float64(target.item(3)), 2.0)
    var fresh = clone(pipe)
    with assert_raises():
        _ = fresh.score(X, target)
    var short = zeros[dtype](Shape(2))
    with assert_raises():
        _ = pipe.score(X, target, sample_weight=short.copy())
    with assert_raises():
        _ = pipe.score(X, short)
    print("score_dense", Int(dtype == DType.float64), pipe.score(X, y),
          pipe.score(X, target), pipe.score(X, target, sample_weight=weights.copy()))


def main() raises:
    comptime assert not conforms_to(Pipeline[Shift, Recorder], SupervisedScorer)
    comptime assert conforms_to(Pipeline[Shift, ScoreRecorder], SupervisedScorer)
    comptime assert not conforms_to(Pipeline[Shift, ScoreRecorder], ClassifierMixin)
    comptime assert not conforms_to(Pipeline[Shift, ScoreRecorder], RegressorMixin)
    var pipe = Pipeline(Shift(), ScoreRecorder())
    with assert_raises():
        _ = pipe.score(1, 7)
    pipe.fit(5, 7)
    assert_equal(through_trait(pipe, 1, 7), 376.0)
    assert_equal(pipe.score(1, 7, sample_weight=3), 380.0)
    assert_equal(pipe.score(1, 7, sample_weight=None), 376.0)
    print("score_scalar", pipe.score(1, 7), pipe.score(1, 7, sample_weight=3))
    with assert_raises():
        _ = pipe.score(1, -1)
    with assert_raises():
        _ = pipe.score(1, 7, sample_weight=-1)
    var fresh = clone(pipe)
    with assert_raises():
        _ = fresh.score(1, 7)
    var params = fresh.get_params()
    params.transformer = 4
    fresh.set_params(params)
    fresh.fit(5, 7)
    assert_equal(fresh.score(1, 7), 578.0)
    pipe.fit(8, 7)
    assert_equal(pipe.score(1, 7), 379.0)
    print("score_refit", fresh.score(1, 7), pipe.score(1, 7))
    with assert_raises():
        pipe.fit(-1, 7)
    assert_true(pipe.__sklearn_is_fitted__())
    with assert_raises():
        _ = pipe.score(1, 7)
    with assert_raises():
        pipe.fit(5, -1)
    assert_false(pipe.__sklearn_is_fitted__())
    with assert_raises():
        _ = pipe.score(1, 7)
    var fused = Pipeline(Fused(), ScoreRecorder())
    fused.fit(5, 7)
    assert_equal(fused.score(1, 7), 1281.0)
    assert_equal(fused.transformer.calls, 1)
    print("score_fused", fused.score(1, 7))
    var nested = Pipeline(Shift(2), Pipeline(Shift(4), ScoreRecorder()))
    nested.fit(5, 7)
    assert_equal(nested.score(1, 7), 780.0)
    assert_equal(nested.score(1, 7, sample_weight=5), 786.0)
    print("score_nested", nested.score(1, 7), nested.score(1, 7, sample_weight=5))
    dense[DType.float32]()
    dense[DType.float64]()
