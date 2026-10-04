"""Input feature-count forwarding with an explicit dimensionality-changing probe."""
from numojo import NDArray, Shape, zeros
from std.testing import assert_equal, assert_false, assert_raises
from moxie.base import clone
from moxie.pipeline import Pipeline, FeatureUnion, UnionSentinel, UnionPair, UnionInputFeatures
from moxie._pipeline import StepMode
from moxie.dummy import DummyRegressor
from union_fixtures import TestTransformer
from test_real_pipeline import CopyStep


def through_trait[P: UnionInputFeatures](pipe: P) raises -> Int:
    return pipe.n_features_in_()


def check[dtype: DType]() raises:
    var X = zeros[dtype](Shape(4, 3))
    var y = zeros[dtype](Shape(4))
    var pipe = Pipeline(TestTransformer[dtype](kind=1), DummyRegressor[dtype]())
    with assert_raises():
        _ = pipe.n_features_in_()
    _ = pipe.fit(X, y)
    assert_equal(pipe.n_features_in_(), 3)
    assert_equal(pipe.transformer.transform(X).shape[1], 1)
    assert_equal(through_trait(pipe), 3)
    print("features", Int(dtype == DType.float64), pipe.n_features_in_(), pipe.transformer.transform(X).shape[1])
    var fresh = clone(pipe)
    with assert_raises():
        _ = fresh.n_features_in_()
    var wider = zeros[dtype](Shape(4, 5))
    _ = fresh.fit(wider, y)
    assert_equal(fresh.n_features_in_(), 5)
    assert_equal(pipe.n_features_in_(), 3)
    _ = pipe.fit(wider, y)
    assert_equal(pipe.n_features_in_(), 5)
    # No pipeline-wide fitted check: only the first step owns this attribute.
    pipe.estimator = DummyRegressor[dtype]()
    assert_false(pipe.__sklearn_is_fitted__())
    assert_equal(pipe.n_features_in_(), 5)
    print("features_refit", Int(dtype == DType.float64), fresh.n_features_in_(), pipe.n_features_in_())
    pipe.transformer = TestTransformer[dtype](kind=1)
    with assert_raises():
        _ = pipe.n_features_in_()
    _ = pipe.fit(X, y)
    var params = pipe.get_params()
    params.transformer.kind = 4
    _ = pipe.set_params(params)
    assert_equal(pipe.n_features_in_(), 3)
    with assert_raises():
        _ = pipe.fit(X, y)
    with assert_raises():
        _ = pipe.n_features_in_()
    var missing = Pipeline(CopyStep[dtype](), DummyRegressor[dtype](), transformer_name="copy")
    _ = missing.fit(X, y)
    try:
        _ = missing.n_features_in_()
        raise Error("expected missing capability")
    except error:
        assert_equal(String(error), "AttributeError: first pipeline step copy does not expose n_features_in_")
    var branches = UnionPair(UnionSentinel[dtype]("a", StepMode.passthrough),
                             UnionSentinel[dtype]("b", StepMode.passthrough))
    var nested = Pipeline(FeatureUnion(branches^),
                          Pipeline(TestTransformer[dtype](kind=1), DummyRegressor[dtype]()))
    _ = nested.fit(X, y)
    assert_equal(nested.n_features_in_(), 3)
    assert_equal(nested.estimator.n_features_in_(), 6)
    print("features_union", Int(dtype == DType.float64), nested.n_features_in_(), nested.estimator.n_features_in_())


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("Pipeline input-feature contracts passed")
