import numojo as nm
from std.math import abs
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from moxie import clone
from moxie._random import check_random_state
from moxie.base import SupervisedEstimator
from moxie.impute import IterativeBound, IterativeImputer, IterativePosteriorRegressor
from moxie.linear_model import LinearRegression
from moxie.pipeline import make_pipeline


comptime _Linear = LinearRegression[DType.float64]


struct _DefaultStreamRegressor(SupervisedEstimator, IterativePosteriorRegressor, Copyable):
    comptime Input = _Linear.Input
    comptime Target = _Linear.Target
    comptime Prediction = _Linear.Prediction
    comptime Params = _Linear.Params
    var _linear: _Linear

    def __init__(out self) raises:
        self._linear = _Linear()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._linear.get_params(deep)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        _ = self._linear.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        _ = result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self._linear.__sklearn_is_fitted__()

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target,
    ) raises -> ref[self] Self:
        var stream = check_random_state()
        _ = stream[].next_uint32()
        _ = self._linear.fit(X, y)
        return self

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        var stream = check_random_state()
        _ = stream[].next_uint32()
        return self._linear.predict(X)

    def predict_std(self, X: Self.Input) raises -> Tuple[Self.Prediction, Self.Prediction]:
        var mean = self.predict(X)
        # Use zero deviation at input 2 to exercise the deterministic fallback.
        var deviation = nm.ones[DType.float64](nm.Shape(mean.size))
        for i in range(mean.size):
            if X.unsafe_get(i * X.shape[1]) == 2:
                deviation.unsafe_set(i, 0)
        return (mean^, deviation^)


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(5, 2))
    var y = nm.zeros[DType.float64](nm.Shape(5))
    for i in range(5):
        X.unsafe_set(2 * i, Float64(i + 1))
        X.unsafe_set(2 * i + 1, Float64(2 * (i + 1)))
        y.unsafe_set(i, Float64(3 * (i + 1)))
    X.unsafe_set(3, Float64("nan"))
    var imputer = IterativeImputer[DType.float64, LinearRegression[DType.float64]](
        LinearRegression(), random_state=0,
    )
    var filled = imputer.fit_transform(X)
    assert_true(abs(filled.unsafe_get(3) - 4) < 1e-12)
    assert_equal(imputer.imputation_sequence_[0].estimator.n_features_in_, 1)
    var pipeline = make_pipeline(clone(imputer), LinearRegression())
    _ = pipeline.fit(X, y)
    var fresh = clone(pipeline)
    assert_false(fresh.__sklearn_is_fitted__())
    var pred = pipeline.predict(X)
    assert_equal(pred.shape[0], y.shape[0])
    for i in range(y.size):
        assert_true(abs(pred.unsafe_get(i) - y.unsafe_get(i)) < 1e-12)
    var default_stream = IterativeImputer[DType.float64, _DefaultStreamRegressor](
        _DefaultStreamRegressor(), imputation_order="random",
    )
    var restored = default_stream.fit_transform(X)
    assert_true(abs(restored.unsafe_get(3) - 4) < 1e-12)
    assert_true(abs(default_stream.transform(X).unsafe_get(3) - 4) < 1e-12)
    var unsupported = IterativeImputer[DType.float64, _Linear](
        _Linear(), sample_posterior=True,
    )
    with assert_raises():
        _ = unsupported.fit(X)
    X.unsafe_set(7, Float64("nan"))
    var posterior = IterativeImputer[DType.float64, _DefaultStreamRegressor](
        _DefaultStreamRegressor(), sample_posterior=True, max_iter=2, random_state=0,
        min_value=IterativeBound(Float64(3)), max_value=IterativeBound(Float64(9)),
    )
    var sampled = posterior.fit_transform(X)
    var replay = posterior.transform(X)
    assert_true(abs(sampled.unsafe_get(3) - 4) < 1e-12)
    assert_true(abs(replay.unsafe_get(3) - 4) < 1e-12)
    assert_true(sampled.unsafe_get(7) != replay.unsafe_get(7))
    assert_true(replay.unsafe_get(7) > 3 and replay.unsafe_get(7) < 9)
    # Make the downstream regression depend on the imputed column.
    y.unsafe_set(1, 7)
    y.unsafe_set(3, -4)
    var stochastic_pipeline = make_pipeline(clone(posterior), LinearRegression())
    _ = stochastic_pipeline.fit(X, y)
    var first = stochastic_pipeline.predict(X)
    var second = stochastic_pipeline.predict(X)
    assert_true(first.unsafe_get(3) != second.unsafe_get(3))
    print("PASS: iterative regressors, downstream pipeline and default RNG composition")
