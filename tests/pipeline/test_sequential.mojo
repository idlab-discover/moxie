from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import SupervisedEstimator, UnsupervisedEstimator, TransformerMixin, FitTransformer, Predictor, clone
from moxie.pipeline import Pipeline, PredictivePipeline


struct Shift(SupervisedEstimator, TransformerMixin):
    comptime Input = Int
    comptime Target = Int
    comptime Transformed = Int
    comptime Params = Int
    var offset: Int
    var fitted: Bool
    var seen: Int
    var seen_y: Int

    def __init__(out self, offset: Int = 2):
        self.offset = offset
        self.fitted = False
        self.seen = 0
        self.seen_y = 0

    def get_params(self, deep: Bool = True) raises -> Int:
        return self.offset

    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Int,
    ) raises -> ref[self] Self:
        self.offset = params
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.offset)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted

    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Int,
        ref[y_origin] y: Int,
    ) raises -> ref[self] Self:
        self.fitted = False
        if X < 0:
            raise Error("ValueError: negative training input")
        self.seen = X
        self.seen_y = y
        self.fitted = True
        return self

    def transform(self, X: Int) raises -> Int:
        if not self.fitted:
            raise Error("NotFittedError: Shift")
        return X + self.offset


struct Fused(SupervisedEstimator, FitTransformer, TransformerMixin):
    comptime Input = Int
    comptime Target = Int
    comptime Transformed = Int
    comptime Params = Int
    var calls: Int
    def __init__(out self):
        self.calls = 0
    def get_params(self, deep: Bool = True) raises -> Int:
        return 0
    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Int,
    ) raises -> ref[self] Self:
        return self
    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return self.calls > 0
    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Int,
        ref[y_origin] y: Int,
    ) raises -> ref[self] Self:
        raise Error("fit must not run when fit_transform exists")
    def fit_transform(mut self, X: Int, y: Optional[Int] = None) raises -> Int:
        self.calls += 1
        return X + y.value() + 100
    def transform(self, X: Int) raises -> Int:
        return X + 10


struct Unsupervised(UnsupervisedEstimator, TransformerMixin):
    comptime Input = Int
    comptime Target = Int
    comptime Transformed = Int
    comptime Params = Int
    var seen_y: Int
    def __init__(out self):
        self.seen_y = 0
    def get_params(self, deep: Bool = True) raises -> Int:
        return 0
    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Int,
    ) raises -> ref[self] Self:
        return self
    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return True
    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Int,
        y: Optional[Int] = None,
    ) raises -> ref[self] Self:
        self.seen_y = y.value()
        return self
    def transform(self, X: Int) raises -> Int:
        return X * 2


struct Recorder(SupervisedEstimator, Predictor):
    comptime Input = Int
    comptime Target = Int
    comptime Prediction = Int
    comptime Params = Int
    var seen: Int
    var seen_y: Int
    var fitted: Bool
    def __init__(out self):
        self.seen = 0
        self.seen_y = 0
        self.fitted = False
    def get_params(self, deep: Bool = True) raises -> Int:
        return 0
    def set_params[
        origin: MutOrigin,
        params_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[params_origin] params: Int,
    ) raises -> ref[self] Self:
        return self
    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted
    def fit[
        origin: MutOrigin,
        x_origin: ImmOrigin,
        y_origin: ImmOrigin,
    ](
        ref[origin] self,
        ref[x_origin] X: Int,
        ref[y_origin] y: Int,
    ) raises -> ref[self] Self:
        self.fitted = False
        if y < 0:
            raise Error("ValueError: negative target")
        self.seen = X
        self.seen_y = y
        self.fitted = True
        return self
    def predict(self, X: Int) raises -> Int:
        if not self.fitted:
            raise Error("NotFittedError: Recorder")
        return X * 3 + self.seen


def through_trait[P: PredictivePipeline](pipeline: P, X: P.Input) raises -> P.Prediction:
    return pipeline.predict(X)


def main() raises:
    var pipe = Pipeline(Shift(), Recorder(), transformer_name="shift", estimator_name="record")
    with assert_raises():
        _ = pipe.predict(1)
    _ = pipe.fit(5, 7)
    assert_equal(pipe.transformer.seen, 5)
    assert_equal(pipe.transformer.seen_y, 7)
    assert_equal(pipe.estimator.seen, 7)
    assert_equal(pipe.estimator.seen_y, 7)
    assert_equal(pipe.predict(1), 16)
    assert_equal(through_trait(pipe, 1), 16)
    print("fallback", pipe.estimator.seen, pipe.estimator.seen_y, pipe.predict(1))
    var independent = clone(pipe)
    assert_false(independent.__sklearn_is_fitted__())
    assert_false(independent.transformer.fitted)
    assert_equal(independent.transformer_name, "shift")
    var params = independent.get_params()
    params.transformer = 4
    _ = independent.set_params(params)
    _ = independent.fit(5, 7)
    assert_equal(independent.predict(1), 24)
    print("clone", independent.predict(1), pipe.predict(1))
    assert_equal(pipe.predict(1), 16)
    _ = pipe.fit(8, 3)
    assert_equal(pipe.estimator.seen, 10)
    # The fitted query follows the final estimator, even when a prefix fit fails.
    with assert_raises():
        _ = pipe.fit(-1, 3)
    assert_true(pipe.__sklearn_is_fitted__())
    print("prefix_failure", Int(pipe.__sklearn_is_fitted__()))
    with assert_raises():
        _ = pipe.predict(1)
    with assert_raises():
        _ = pipe.fit(5, -1)
    assert_false(pipe.__sklearn_is_fitted__())
    assert_true(pipe.transformer.fitted)
    print("final_failure", Int(pipe.__sklearn_is_fitted__()), Int(pipe.transformer.fitted))
    var fused = Pipeline(Fused(), Recorder())
    _ = fused.fit(5, 7)
    assert_equal(fused.transformer.calls, 1)
    assert_equal(fused.estimator.seen, 112)
    assert_equal(fused.predict(1), 145)
    print("fused", fused.transformer.calls, fused.estimator.seen, fused.predict(1))
    var unsupervised = Pipeline(Unsupervised(), Recorder())
    _ = unsupervised.fit(5, 7)
    assert_equal(unsupervised.transformer.seen_y, 7)
    assert_equal(unsupervised.estimator.seen, 10)
    print("unsupervised", unsupervised.transformer.seen_y, unsupervised.estimator.seen)
    var nested = Pipeline(Shift(2), Pipeline(Shift(4), Recorder()))
    _ = nested.fit(5, 7)
    assert_equal(nested.estimator.estimator.seen, 11)
    assert_equal(nested.predict(1), 32)
    print("nested", nested.estimator.estimator.seen, nested.predict(1))
    with assert_raises():
        _ = Pipeline(Shift(), Recorder(), transformer_name="same", estimator_name="same")
    with assert_raises():
        _ = Pipeline(Shift(), Recorder(), transformer_name="memory")
    params.transformer_name = params.estimator_name
    with assert_raises():
        _ = independent.set_params(params)
    assert_equal(independent.transformer_name, "shift")
    print("Sequential execution, clone, refit, dispatch and failure contracts passed")
