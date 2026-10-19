from moxie.consumer_a import ConsumerA
from moxie.consumer_b import ConsumerB
from moxie._dense import row_squared_norms
import numojo as nm
from std.testing import assert_equal, assert_false, assert_raises
from moxie.base import clone

def main() raises:
    var a = ConsumerA()
    var b = ConsumerB()
    var x = nm.ones[DType.float64](nm.Shape(2, 3))
    with assert_raises():
        _ = a.transform(x)
    _ = a.fit(x)
    _ = b.fit(x)
    var unfitted = clone(a)
    assert_false(unfitted.__sklearn_is_fitted__())
    _ = unfitted.set_params([False, True, False])
    _ = unfitted.fit(x)
    assert_equal(unfitted.transform(x).shape[1], 1)
    assert_equal(a.transform(x).shape[1], 2)
    var first = a.transform(x)
    var second = b.transform(x)
    assert_equal(first.shape[1], 2)
    assert_equal(Float64(row_squared_norms(first).unsafe_get(0)), 2.0)
    assert_equal(Float64(row_squared_norms(second).unsafe_get(0)), 2.0)
    first.unsafe_set(0, 9)
    assert_equal(Float64(second.unsafe_get(0)), 1.0)
    _ = a.set_params([True, True, True])
    _ = a.fit(x)
    assert_equal(a.transform(x).shape[1], 3)
    assert_equal(b.transform(x).shape[1], 2)

    from moxie.feature_selection import VarianceThreshold
    var variance_input = nm.zeros[DType.float32](nm.Shape(2, 1))
    variance_input.unsafe_set(1, 2)
    var variance = VarianceThreshold[DType.float32]()
    _ = variance.fit(variance_input)
    assert_equal(variance.transform(variance_input).shape[1], 1)

    from moxie.feature_selection import SelectKBest, BuiltinScorer, SelectionParameter
    var labels = nm.zeros[DType.float32](nm.Shape(2))
    labels.unsafe_set(1, 1)
    var univariate = SelectKBest(BuiltinScorer[DType.float32]("r_regression"), k=SelectionParameter(1))
    _ = univariate.fit(variance.transform(variance_input), labels^)
    var univariate_output = univariate.transform(variance_input)
    assert_equal(univariate_output.shape[1], 1)
    assert_equal(univariate.get_support(True)[List[Int]][0], 0)
    univariate_output.unsafe_set(0, 9)
    assert_equal(variance_input.unsafe_get(0), 0)

    from moxie.feature_selection import SelectFromModel, LassoImportance
    from moxie.linear_model import Lasso
    var model_target = nm.zeros[DType.float32](nm.Shape(2))
    model_target.unsafe_set(1, 4)
    var from_model = SelectFromModel(LassoImportance(Lasso[DType.float32](alpha=0.01)))
    _ = from_model.fit(variance_input, model_target^)
    assert_equal(from_model.transform(variance_input).shape[1], 1)

    from moxie.feature_selection import RFE, FeatureCount, SelectorMixin
    comptime assert conforms_to(RFE[LassoImportance[DType.float32]], SelectorMixin)
    var rfe_X = nm.zeros[DType.float32](nm.Shape(4, 2))
    var rfe_y = nm.zeros[DType.float32](nm.Shape(4))
    rfe_X.unsafe_set(0, 1)
    rfe_X.unsafe_set(3, 1)
    rfe_X.unsafe_set(4, 1)
    rfe_X.unsafe_set(7, 1)
    rfe_y.unsafe_set(0, 4)
    rfe_y.unsafe_set(2, 4)
    var rfe = RFE(LassoImportance(Lasso[DType.float32](alpha=0.01, fit_intercept=False)), n_features_to_select=FeatureCount(1))
    _ = rfe.fit(rfe_X, rfe_y)
    assert_equal(rfe.get_support(True)[List[Int]][0], 0)
    assert_equal(rfe.ranking_()[1], 2)
    var rfe_output = rfe.transform(rfe_X)
    rfe_output.unsafe_set(0, 99)
    assert_equal(rfe_X.unsafe_get(0), 1)
    assert_equal(rfe.predict(rfe_X).shape[0], 4)

    from moxie.feature_selection import MutualInfoScorer, DiscreteFeatures, mutual_info_classif
    var mi = SelectKBest(MutualInfoScorer[DType.float32](discrete_features=DiscreteFeatures(True), random_state=0), k=1)
    _ = mi.fit(rfe_X, rfe_y.copy())
    assert_equal(mi.transform(rfe_X).shape[1], 1)
    var mi_scores = mutual_info_classif(rfe_X, rfe_y, random_state=0, n_neighbors=1)
    assert_equal(mi_scores.shape[0], 2)

    from moxie.feature_selection import SequentialFeatureSelector, DenseSupervisedCV, SequentialCount, SequentialCV
    comptime SequentialAdapter = DenseSupervisedCV[Lasso[DType.float32], DType.float32]
    var sequential = SequentialFeatureSelector[SequentialAdapter](Lasso[DType.float32](alpha=0.01, fit_intercept=False), n_features_to_select=SequentialCount(1), cv=SequentialCV(2))
    _ = sequential.fit(rfe_X, rfe_y)
    assert_equal(sequential.n_features_to_select_(), 1)
    assert_equal(sequential.get_support(True)[List[Int]][0], 0)
    assert_equal(sequential.transform(rfe_X).shape[1], 1)

    from moxie.feature_selection import RFECV
    comptime RecursiveCV = DenseSupervisedCV[LassoImportance[DType.float32], DType.float32]
    var rfecv = RFECV[RecursiveCV](LassoImportance(Lasso[DType.float32](alpha=0.01, fit_intercept=False)), cv=SequentialCV(2))
    _ = rfecv.fit(rfe_X, rfe_y)
    assert_equal(rfecv.n_features_(), 1)
    assert_equal(rfecv.get_support(True)[List[Int]][0], 0)
    assert_equal(rfecv.cv_results_()["n_features"][List[Int]], [1, 2])
    assert_equal(rfecv.predict(rfe_X).shape[0], 4)
    var selected_cv = rfecv.transform(rfe_X)
    selected_cv.unsafe_set(0, 99)
    assert_equal(rfe_X.unsafe_get(0), 1)

    from moxie.decomposition import IncrementalPCA
    from moxie.feature_selection import IncrementalPCAImportance
    var incremental_X = nm.zeros[DType.float64](nm.Shape(4, 2))
    for i in range(4):
        incremental_X.unsafe_set(i * 2, Float64(i))
    var incremental = SelectFromModel(IncrementalPCAImportance(IncrementalPCA[DType.float64](1)), importance_getter="components_")
    _ = incremental.partial_fit(incremental_X)
    _ = incremental.partial_fit(incremental_X)
    assert_equal(incremental.estimator_().model.n_samples_seen_, 8)
    assert_equal(incremental.get_support(True)[List[Int]], [0])

    from moxie.tree import DecisionTreeClassifier
    from moxie.feature_selection import DecisionTreeClassifierImportance
    from moxie._labels import StringTarget
    comptime TreeAdapter = DecisionTreeClassifierImportance[DType.float64, DType.int64, True]
    var tree_x = nm.zeros[DType.float64](nm.Shape(6, 2))
    for i in range(6):
        tree_x[nm.Item(i, 0)] = SIMD[DType.float64, 1](i % 2)
    var tree_y = StringTarget(["z", "a", "z", "a", "z", "a"])
    var tree_selector = SelectFromModel(TreeAdapter(DecisionTreeClassifier[DType.int64, True](random_state=42)))
    _ = tree_selector.fit(tree_x, tree_y.copy())
    assert_equal(tree_selector.get_support(True)[List[Int]], [0])
    assert_equal(tree_selector.estimator_().classes_(), ["a", "z"])

    from moxie.feature_selection import (SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect, chi2, f_classif, f_oneway, f_regression, r_regression, mutual_info_regression)
    comptime assert conforms_to(RFE[LassoImportance[DType.float32]], SelectorMixin)
    var percentile = SelectPercentile(BuiltinScorer[DType.float32]("f_regression"), percentile=50)
    _ = percentile.fit(rfe_X, rfe_y.copy())
    assert_equal(percentile.transform(rfe_X).shape[1], 1)
    var fpr = SelectFpr(BuiltinScorer[DType.float32]("f_regression"))
    var fdr = SelectFdr(BuiltinScorer[DType.float32]("f_regression"))
    var fwe = SelectFwe(BuiltinScorer[DType.float32]("f_regression"))
    var generic = GenericUnivariateSelect(BuiltinScorer[DType.float32]("f_regression"))
    _ = fpr.fit(rfe_X, rfe_y.copy())
    _ = fdr.fit(rfe_X, rfe_y.copy())
    _ = fwe.fit(rfe_X, rfe_y.copy())
    _ = generic.fit(rfe_X, rfe_y.copy())
    assert_equal(len(fpr.get_support(False)[List[Bool]]), 2)
    assert_equal(len(fdr.get_support(False)[List[Bool]]), 2)
    assert_equal(len(fwe.get_support(False)[List[Bool]]), 2)
    assert_equal(len(generic.get_support(False)[List[Bool]]), 2)
    assert_equal(chi2(rfe_X, rfe_y).scores.shape[0], 2)
    assert_equal(f_classif(rfe_X, rfe_y).scores.shape[0], 2)
    assert_equal(f_regression(rfe_X, rfe_y).scores.shape[0], 2)
    assert_equal(r_regression(rfe_X, rfe_y).shape[0], 2)
    assert_equal(mutual_info_regression(rfe_X, rfe_y, random_state=0, n_neighbors=1).shape[0], 2)
    var groups = List[nm.NDArray[DType.float32]]()
    groups.append(nm.zeros[DType.float32](nm.Shape(3, 2)))
    groups.append(nm.ones[DType.float32](nm.Shape(3, 2)))
    assert_equal(f_oneway(groups).scores.shape[0], 2)
