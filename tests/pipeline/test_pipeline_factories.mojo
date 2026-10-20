"""make_pipeline/make_union naming and equivalence with explicit construction."""
from numojo import NDArray, Shape
from std.testing import assert_equal, assert_false, assert_true
from moxie.base import clone
from moxie.pipeline import Pipeline, FeatureUnion, UnionPair, union_step, make_pipeline, make_union
from moxie.preprocessing import StandardScaler, MinMaxScaler
from moxie.decomposition import PCA
from moxie.svm import SVC
from test_pipeline_prefix_target import data


def joined(names: List[String]) -> String:
    var line = String()
    for name in names:
        line += " " + name
    return line


def main() raises:
    var d = data()
    var X = d[0].copy()
    var y = d[1].copy()

    var pipe = make_pipeline(StandardScaler(), SVC())
    assert_equal(pipe.transformer_name, "standardscaler")
    assert_equal(pipe.estimator_name, "svc")
    var explicit = Pipeline(StandardScaler(), SVC())
    var expected = explicit.fit(X, y).decision_function(X)
    var actual = pipe.fit(X, y).decision_function(X)
    for i in range(12):
        assert_equal(Float64(actual.item(i)), Float64(expected.item(i)))
    var fresh = clone(pipe)
    assert_false(fresh.__sklearn_is_fitted__())
    assert_equal(fresh.transformer_name, "standardscaler")
    print("factory_pipeline", pipe.transformer_name, pipe.estimator_name)

    var union = make_union(StandardScaler(), MinMaxScaler())
    assert_equal(union.n_jobs, None)
    assert_false(union.verbose)
    assert_true(union.verbose_feature_names_out)
    var features = union.fit_transform(X)
    var manual = FeatureUnion(UnionPair(union_step("a", StandardScaler()), union_step("b", MinMaxScaler())))
    var manual_features = manual.fit_transform(X)
    assert_equal(features.shape[1], 6)
    for i in range(72):
        assert_equal(Float64(features.item(i)), Float64(manual_features.item(i)))
    print("factory_union" + joined(union.transformer_list.names()) + " |" + joined(union.get_feature_names_out()))

    var same = make_union(StandardScaler(), StandardScaler(), verbose_feature_names_out=False)
    assert_false(same.verbose_feature_names_out)
    print("factory_duplicates" + joined(same.transformer_list.names()))

    var three = make_union(PCA(n_components=1), StandardScaler(), PCA(n_components=2))
    _ = three.fit(X)
    assert_equal(three.transform(X).shape[1], 6)
    print("factory_three" + joined(three.transformer_list.names()))

    var narrow = make_union[DType.float32](StandardScaler[DType.float32](), MinMaxScaler[DType.float32]())
    var X32 = X.astype[DType.float32]()
    assert_equal(narrow.fit_transform(X32).shape[1], 6)
