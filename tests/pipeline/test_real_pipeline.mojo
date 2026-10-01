"""Real DummyRegressor integration; CopyStep is an explicit test fixture."""
from numojo import NDArray, Shape
from std.testing import assert_equal, assert_false, assert_true, assert_raises
from moxie.base import TransformerMixin, FitTransformer, clone
from moxie.pipeline import Pipeline
from moxie.dummy import DummyRegressor


struct CopyStep[dtype: DType](TransformerMixin, FitTransformer):
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]
    comptime Params = Int
    var fitted: Bool
    var target_rows: Int
    def __init__(out self):
        self.fitted = False
        self.target_rows = 0
    def get_params(self, deep: Bool = True) raises -> Int:
        return 0
    def set_params(mut self, params: Int) raises:
        pass
    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return self.fitted
    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self.target_rows = y.value().shape[0]
        self.fitted = True
        return X.copy()
    def transform(self, X: Self.Input) raises -> Self.Transformed:
        if not self.fitted:
            raise Error("NotFittedError: CopyStep")
        return X.copy()


def check[dtype: DType]() raises:
    var X = NDArray[dtype](Shape(4, 2))
    var y = NDArray[dtype](Shape(4))
    for i in range(4):
        y.itemset(i, Scalar[dtype](i * 2))
        for j in range(2):
            X.itemset(i * 2 + j, Scalar[dtype](i + j))
    var pipe = Pipeline(CopyStep[dtype](), DummyRegressor[dtype](),
                        transformer_name="copy", estimator_name="dummy")
    with assert_raises():
        _ = pipe.predict(X)
    pipe.fit(X, y)
    assert_true(pipe.__sklearn_is_fitted__())
    assert_equal(pipe.transformer.target_rows, 4)
    var prediction = pipe.predict(X)
    assert_equal(prediction.ndim, 1)
    assert_equal(prediction.shape[0], 4)
    for i in range(4):
        assert_equal(Float64(prediction.item(i)), 3.0)
        assert_equal(Float64(y.item(i)), Float64(i * 2))
    assert_equal(Float64(X.item(0)), 0.0)
    var other = clone(pipe)
    assert_false(other.__sklearn_is_fitted__())
    assert_false(other.transformer.fitted)
    var params = other.get_params()
    params.estimator.strategy = "median"
    other.set_params(params)
    y.itemset(0, 100)
    other.fit(X, y)
    var median = other.predict(X)
    assert_equal(Float64(median.item(0)), 5.0)
    assert_equal(Float64(pipe.predict(X).item(0)), 3.0)
    pipe.fit(X, y)
    assert_equal(Float64(pipe.predict(X).item(0)), 28.0)
    var nested = Pipeline(CopyStep[dtype](), pipe^)
    nested.fit(X, y)
    assert_equal(Float64(nested.predict(X).item(0)), 28.0)
    print("real", Int(dtype == DType.float64), Float64(prediction.item(0)), Float64(median.item(0)), Float64(nested.predict(X).item(0)))


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
