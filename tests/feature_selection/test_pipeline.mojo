import numojo as nm
from std.testing import assert_equal, assert_false
from moxie.base import clone
from moxie.pipeline import Pipeline
from moxie.dummy import DummyRegressor
from moxie.feature_selection import VarianceThreshold, SelectKBest, BuiltinScorer

def check[dtype: DType]() raises:
    var X = nm.zeros[dtype](nm.Shape(6, 3))
    var y = nm.zeros[dtype](nm.Shape(6))
    for i in range(6):
        X.unsafe_set(3 * i, 7)
        X.unsafe_set(3 * i + 1, SIMD[dtype, 1](i))
        X.unsafe_set(3 * i + 2, SIMD[dtype, 1](i % 2))
        y.unsafe_set(i, SIMD[dtype, 1](i))
    var pipe = Pipeline(VarianceThreshold[dtype](), Pipeline(SelectKBest(BuiltinScorer[dtype]("r_regression"), k=1), DummyRegressor[dtype]()))
    _ = pipe.fit(X, y)
    assert_equal(pipe.transformer.get_support(True)[List[Int]], [1, 2])
    assert_equal(pipe.estimator.transformer.get_support(True)[List[Int]], [0])
    assert_equal(pipe.predict(X).shape[0], 6)
    assert_equal(Float64(pipe.predict(X).unsafe_get(0)), 2.5)
    var fresh = clone(pipe)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(Float64(X.unsafe_get(0)), 7.0)

def main() raises:
    check[DType.float32]()
    check[DType.float64]()
