from numojo import NDArray, Shape, NDArrayStrides, zeros
from std.math import nan, isnan
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone, TransformerMixin, FitTransformer
from moxie._pipeline import StepMode
from moxie.pipeline import FeatureUnion, UnionPair, UnionSentinel, union_step, Pipeline
from moxie.dummy import DummyRegressor
from union_fixtures import TestTransformer, SupervisedOnly, FusedOnly
from test_real_pipeline import CopyStep


def check[dtype: DType]() raises:
    var X = zeros[dtype](Shape(2, 2))
    for i in range(4):
        X.itemset(i, Scalar[dtype](i + 1))
    var y = zeros[dtype](Shape(2))
    y.itemset(0, 2)
    y.itemset(1, 4)
    var weights: Dict[String, Float64] = {"double": 0.5, "fused": -2.0}
    var union = FeatureUnion(UnionPair(
        union_step[dtype]("double", TestTransformer[dtype](2)),
        union_step[dtype]("fused", TestTransformer[dtype, True](3, 1))), transformer_weights=weights.copy())
    assert_false(union.__sklearn_is_fitted__())
    with assert_raises():
        _ = union.transform(X)
    _ = union.fit(X, y.copy())
    assert_equal(union.transformer_list.first.estimator.fit_calls, 1)
    assert_equal(union.transformer_list.second.estimator.fit_calls, 1)
    assert_equal(union.transformer_list.second.estimator.fused_calls, 0)
    assert_equal(union.transformer_list.second.estimator.target_sum, 6.0)
    assert_true(union.__sklearn_is_fitted__())
    var result = union.transform(X)
    assert_equal(result.shape[0], 2)
    assert_equal(result.shape[1], 3)
    assert_equal(Float64(result.item(0, 0)), 1.0)
    assert_equal(Float64(result.item(0, 1)), 2.0)
    assert_equal(Float64(result.item(0, 2)), -6.0)
    var trained = union.fit_transform(X, y.copy())
    assert_equal(union.transformer_list.second.estimator.fused_calls, 1)
    assert_equal(Float64(trained.item(0, 2)), -26.0)
    assert_equal(Float64(X.item(0, 0)), 1.0)
    var copied = clone(union)
    assert_false(copied.__sklearn_is_fitted__())
    assert_equal(copied.transformer_list.first.estimator.fit_calls, 0)
    var params = copied.get_params()
    params.transformer_weights.value()["double"] = 4
    params.transformer_list.first.estimator.factor = 5
    _ = copied.set_params(params)
    params.transformer_weights.value()["double"] = 9
    assert_equal(copied.transformer_weights.value()["double"], 4.0)
    assert_equal(union.transformer_weights.value()["double"], 0.5)
    assert_equal(union.transformer_list.first.estimator.factor, 2.0)
    params.transformer_list.second.name = "double"
    with assert_raises():
        _ = copied.set_params(params)
    assert_equal(copied.transformer_list.second.name, "fused")
    var names = union.get_feature_names_out()
    assert_equal(names[0], "double__x0")
    assert_equal(names[2], "fused__x0")
    union.verbose_feature_names_out = False
    with assert_raises():
        _ = union.get_feature_names_out()
    # All-drop keeps the statically selected dtype, including Float32.
    var dropped = FeatureUnion(UnionPair(UnionSentinel[dtype]("a", StepMode.drop), UnionSentinel[dtype]("b", StepMode.drop)))
    assert_true(dropped.__sklearn_is_fitted__())
    var empty = dropped.transform(X)
    assert_equal(empty.shape[0], 2)
    assert_equal(empty.shape[1], 0)
    assert_equal(len(dropped.get_feature_names_out()), 0)
    var sentinel_params = dropped.get_params()
    sentinel_params.transformer_list.first.mode = StepMode.passthrough
    _ = dropped.set_params(sentinel_params)
    var passed = dropped.fit_transform(X)
    assert_equal(Float64(passed.item(1, 1)), 4.0)
    passed.itemset(0, 99)
    assert_equal(Float64(X.item(0)), 1.0)
    names = dropped.get_feature_names_out()
    assert_equal(names[1], "a__x1")
    var bad_names: List[String] = ["bad"]
    with assert_raises():
        _ = dropped.get_feature_names_out(bad_names.copy())

    assert_equal(union.n_features_in_(), 2)
    # Feature count delegates to the first branch, including external child refits.
    var three_cols = zeros[dtype](Shape(2, 3))
    _ = union.transformer_list.first.estimator.fit(three_cols)
    assert_equal(union.n_features_in_(), 3)
    with assert_raises():
        _ = FeatureUnion(UnionPair(UnionSentinel[dtype]("x"), UnionSentinel[dtype]("x")))
    with assert_raises():
        _ = FeatureUnion(UnionSentinel[dtype]("verbose"))
    with assert_raises():
        _ = FeatureUnion(UnionSentinel[dtype]("x__y"))
    with assert_raises():
        _ = FeatureUnion(UnionSentinel[dtype]("ok"), n_jobs=2)
    with assert_raises():
        _ = FeatureUnion(UnionSentinel[dtype]("ok"), transformer_weights=weights.copy())
    with assert_raises():
        _ = UnionSentinel[dtype]("bad", 99)
    # Shape validation occurs after branch execution, as in sklearn's hstack.
    var bad = FeatureUnion(UnionPair(union_step[dtype]("bad", TestTransformer[dtype](1, 2)),
                                    union_step[dtype]("later", TestTransformer[dtype]())))
    with assert_raises():
        _ = bad.fit_transform(X)
    assert_true(bad.transformer_list.second.estimator.fitted)
    bad.transformer_list.first.estimator.kind = 3
    with assert_raises():
        _ = bad.fit_transform(X)
    bad.transformer_list.first.estimator.kind = 4
    with assert_raises():
        _ = bad.fit(X)
    assert_false(bad.__sklearn_is_fitted__())
    bad.transformer_list.first.estimator.kind = 0
    _ = bad.fit(X)
    assert_true(bad.__sklearn_is_fitted__())
    bad.transformer_list.second.estimator.kind = 4
    with assert_raises():
        _ = bad.fit(X)
    assert_true(bad.transformer_list.first.estimator.fitted)
    assert_false(bad.transformer_list.second.estimator.fitted)
    # A recursive branch tree is a flat ordered list, including drop positions.
    var many = FeatureUnion(UnionPair(UnionSentinel[dtype]("drop", StepMode.drop),
        UnionPair(UnionSentinel[dtype]("raw"), union_step[dtype]("scaled", TestTransformer[dtype](3)))))
    var many_out = many.fit_transform(X)
    assert_equal(many_out.shape[1], 4)
    assert_equal(Float64(many_out.item(1, 3)), 12.0)
    with assert_raises():
        _ = many.n_features_in_()  # first declared branch is drop
    var rank_one = zeros[dtype](Shape(2))
    with assert_raises():
        _ = many.transform(rank_one)
    var zero_rows = zeros[dtype](Shape(0, 2))
    assert_equal(many.transform(zero_rows).shape[0], 0)
    var zero_cols = zeros[dtype](Shape(3, 0))
    assert_equal(many.transform(zero_cols).shape[1], 0)
    var one = FeatureUnion(union_step[dtype]("one", TestTransformer[dtype](1)))
    var independent_output = one.fit_transform(X)
    independent_output.itemset(0, 99)
    assert_equal(Float64(X.item(0)), 1.0)
    var base = zeros[dtype](Shape(3, 4))
    for i in range(12):
        base.itemset(i, Scalar[dtype](i + 1))
    var reversed_view = base.view_with_layout(Shape(2, 2), NDArrayStrides(-4, 2), 8)
    var views = FeatureUnion(UnionPair(UnionSentinel[dtype]("a"), UnionSentinel[dtype]("b")))
    var view_out = views.transform(reversed_view)
    assert_equal(Float64(view_out.item(0, 0)), 9.0)
    assert_equal(Float64(view_out.item(0, 1)), 11.0)
    assert_equal(Float64(view_out.item(1, 3)), 7.0)
    view_out.itemset(0, -1)
    assert_equal(Float64(base.item(8)), 9.0)
    # Nonfinite features are forwarded; the union does not impose estimator validation.
    X.itemset(0, nan[dtype]())
    var nonfinite = views.transform(X)
    assert_true(isnan(nonfinite.item(0)))
    # A union is a real prefix for the implemented Pipeline and DummyRegressor.
    X.itemset(0, 1)
    var model = Pipeline(views^, DummyRegressor[dtype]())
    _ = model.fit(X, y)
    var prediction = model.predict(X)
    assert_equal(prediction.shape[0], 2)
    assert_equal(Float64(prediction.item(0)), 3.0)
    assert_false(clone(model).__sklearn_is_fitted__())
    var supervised = FeatureUnion(union_step[dtype]("supervised", SupervisedOnly[dtype]()))
    with assert_raises():
        _ = supervised.fit(X)
    _ = supervised.fit(X, y.copy())
    assert_equal(Float64(supervised.transform(X).item(0)), 1.0)
    var fused_only = FeatureUnion(union_step[dtype]("fused", FusedOnly[dtype]()))
    with assert_raises():
        _ = fused_only.fit(X)
    assert_equal(Float64(fused_only.fit_transform(X).item(0)), 1.0)
    var no_names = FeatureUnion(union_step[dtype]("no_names", CopyStep[dtype]()))
    with assert_raises():
        _ = no_names.get_feature_names_out()
    with assert_raises():
        _ = no_names.n_features_in_()
    # Nested unions are also native transformer constituents.
    var nested = FeatureUnion(union_step[dtype]("nested", FeatureUnion(UnionSentinel[dtype]("raw"))))
    assert_equal(Float64(nested.fit_transform(X).item(0)), 1.0)
    assert_equal(nested.get_feature_names_out()[0], "nested__raw__x0")
    var silent = FeatureUnion(UnionSentinel[dtype]("raw"), n_jobs=1, verbose_feature_names_out=False)
    _ = silent.fit(X)
    assert_equal(silent.get_feature_names_out()[0], "x0")
    var all_dropped = FeatureUnion(UnionSentinel[dtype]("drop", StepMode.drop), verbose_feature_names_out=False)
    with assert_raises():
        _ = all_dropped.get_feature_names_out()
    print("FeatureUnion Float", 32 if dtype == DType.float32 else 64, "contracts passed")


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
