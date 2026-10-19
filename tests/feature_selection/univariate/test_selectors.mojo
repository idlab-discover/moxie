import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone
from moxie.feature_selection import ConfiguredScorer, ScoreResult
from moxie.feature_selection import SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect, BuiltinScorer, SelectionParameter

struct Float32Regression(ConfiguredScorer):
    comptime dtype = DType.float32
    comptime score_dtype = DType.float64
    comptime Target = nm.NDArray[DType.float32]

    def __init__(out self):
        pass

    def clone_config(self) -> Self:
        return Self()

    def score(self, x: nm.NDArray[Self.dtype], y: Optional[Self.Target]) raises -> ScoreResult[Self.score_dtype]:
        from moxie.feature_selection import f_regression
        return f_regression(x, y.value())

def main() raises:
    var x = nm.ones[DType.float64](nm.Shape(6, 3))
    var y = nm.zeros[DType.float64](nm.Shape(6))
    for i in range(6):
        y.unsafe_set(i, Float64(i // 3))
        x.unsafe_set(i * 3, Float64(i))
        x.unsafe_set(i * 3 + 1, Float64(i % 2))
    var selector = SelectKBest(k=SelectionParameter(1))
    with assert_raises():
        _ = selector.get_support()
    _ = selector.fit(x, y.copy())
    assert_true(selector.__sklearn_is_fitted__())
    var result = selector.transform(x)
    assert_equal(result.shape[1], 1)
    assert_equal(selector.n_features_in_(), 3)
    var scores = selector.scores_()
    scores.unsafe_set(0, -900)
    assert_true(selector.scores_().unsafe_get(0) != -900)
    var duplicate = clone(selector)
    assert_false(duplicate.__sklearn_is_fitted__())
    _ = selector.set_params(k=SelectionParameter("all"))
    assert_equal(selector.transform(x).shape[1], 3)
    _ = selector.set_params(k=SelectionParameter(0))
    assert_equal(selector.transform(x).shape[1], 0)
    _ = selector.set_params(k=SelectionParameter(1))
    selector.fit_named(x, y.copy(), [String("a"), String("b"), String("c")])
    assert_equal(selector.get_feature_names_out()[0], "a")
    assert_equal(selector.inverse_transform(selector.transform(x)).shape[1], 3)
    var percentile = SelectPercentile(percentile=50)
    _ = percentile.fit(x, y.copy())
    assert_equal(percentile.transform(x).shape[1], 1)
    var generic = GenericUnivariateSelect(mode="k_best", param=SelectionParameter(1))
    _ = generic.fit(x, y.copy())
    assert_equal(generic.transform(x).shape[1], 1)
    var fpr = SelectFpr()
    _ = fpr.fit(x, y.copy())
    _ = fpr.get_support()
    var fdr = SelectFdr()
    _ = fdr.fit(x, y.copy())
    _ = fdr.get_support()
    var fwe = SelectFwe()
    _ = fwe.fit(x, y.copy())
    _ = fwe.get_support()
    var regression = SelectKBest(BuiltinScorer("f_regression"), k=SelectionParameter(2))
    _ = regression.fit(x, y.copy())
    assert_equal(regression.transform(x).shape[1], 2)
    var x32 = x.astype[DType.float32]()
    var y32 = y.astype[DType.float32]()
    var float32_selector = SelectKBest(BuiltinScorer[DType.float32](), k=SelectionParameter(1))
    _ = float32_selector.fit(x32, y32.copy())
    var selected32: nm.NDArray[DType.float32] = float32_selector.transform(x32)
    var restored32: nm.NDArray[DType.float32] = float32_selector.inverse_transform(selected32)
    var scores64: nm.NDArray[DType.float64] = float32_selector.scores_()
    assert_equal(selected32.shape[1], 1)
    assert_equal(restored32.shape[1], 3)
    assert_equal(float32_selector.get_support(True)[List[Int]][0], 0)
    assert_equal(selected32.unsafe_get(5), 5)
    scores64.unsafe_set(0, -900)
    assert_true(float32_selector.scores_().unsafe_get(0) != -900)
    var duplicate32 = clone(float32_selector)
    assert_false(duplicate32.__sklearn_is_fitted__())
    _ = float32_selector.set_params(k=SelectionParameter(0))
    assert_equal(float32_selector.transform(x32).shape[1], 0)
    _ = float32_selector.set_params(k=SelectionParameter("all"))
    assert_equal(float32_selector.transform(x32).shape[1], 3)
    float32_selector.fit_named(x32, y32.copy(), [String("a"), String("b"), String("c")])
    assert_equal(float32_selector.get_feature_names_out()[2], "c")
    var custom = SelectKBest(Float32Regression(), k=SelectionParameter(1))
    _ = custom.fit(x32, y32.copy())
    var custom_scores: nm.NDArray[DType.float64] = custom.scores_()
    assert_equal(custom.transform(x32).shape[1], 1)
    assert_equal(custom.get_support(True)[List[Int]][0], 0)
    var correlation = SelectKBest(BuiltinScorer[DType.float32]("r_regression"), k=SelectionParameter(1))
    _ = correlation.fit(x32, y32.copy())
    assert_false(Bool(correlation.pvalues_()))
    var missing_p = SelectFpr(BuiltinScorer[DType.float32]("r_regression"))
    _ = missing_p.fit(x32, y32.copy())
    with assert_raises():
        _ = missing_p.get_support()
    print("selector lifecycle contracts passed")
