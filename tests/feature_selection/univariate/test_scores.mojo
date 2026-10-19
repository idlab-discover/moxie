import numojo as nm
from std.math import nan, inf, isnan
from std.testing import assert_equal, assert_raises, assert_true
from moxie.feature_selection import f_classif, f_regression, r_regression, f_oneway, chi2, SelectKBest, BuiltinScorer, SelectionParameter

def check_strided_scores[dtype: DType]() raises:
    var backing = nm.zeros[dtype](nm.Shape(128))
    var target_backing = nm.zeros[dtype](nm.Shape(32))
    for i in range(128):
        backing.unsafe_set(i, nan[dtype]())
    for i in range(32):
        target_backing.unsafe_set(i, nan[dtype]())
    var x = backing.view_with_layout(nm.Shape(6, 3), nm.NDArrayStrides(-10, 2), 60)
    var y = target_backing.view_with_layout(nm.Shape(6), nm.NDArrayStrides(-2), 20)
    var dense_x = nm.zeros[dtype](nm.Shape(6, 3))
    var dense_y = nm.zeros[dtype](nm.Shape(6))
    for i in range(6):
        y.unsafe_set(-2 * i, SIMD[dtype, 1](i % 3))
        dense_y.unsafe_set(i, SIMD[dtype, 1](i % 3))
        for j in range(3):
            var value = SIMD[dtype, 1](1 + (i * i + 3 * i * j + j) % 11)
            x.unsafe_set(-10 * i + 2 * j, value)
            dense_x.unsafe_set(i * 3 + j, value)
    var names: List[String] = ["chi2", "f_classif", "f_regression", "r_regression"]
    for name in names:
        var selector = SelectKBest(BuiltinScorer[dtype](name), k=SelectionParameter(2))
        var dense_selector = SelectKBest(BuiltinScorer[dtype](name), k=SelectionParameter(2))
        _ = selector.fit(x, y.copy())
        _ = dense_selector.fit(dense_x, dense_y.copy())
        var scores = selector.scores_()
        var dense_scores = dense_selector.scores_()
        for j in range(3):
            assert_equal(scores.unsafe_get(j), dense_scores.unsafe_get(j))
        var selected: nm.NDArray[dtype] = selector.transform(x)
        var dense_selected: nm.NDArray[dtype] = dense_selector.transform(dense_x)
        for i in range(12):
            assert_equal(selected.unsafe_get(i), dense_selected.unsafe_get(i))
        selected.unsafe_set(0, -99)
        assert_equal(x.unsafe_get(0), dense_x.unsafe_get(0))
        scores.unsafe_set(0, -99)
        assert_equal(selector.scores_().unsafe_get(0), dense_scores.unsafe_get(0))
    assert_true(isnan(backing.unsafe_get(0)))
    assert_true(isnan(target_backing.unsafe_get(0)))
    for i in range(6):
        assert_equal(y.unsafe_get(-2 * i), dense_y.unsafe_get(i))
        for j in range(3):
            assert_equal(x.unsafe_get(-10 * i + 2 * j), dense_x.unsafe_get(i * 3 + j))

def check_mixed_targets[dtype: DType, target_dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(6, 2))
    var y = nm.zeros[target_dtype](nm.Shape(6))
    var labels = nm.zeros[dtype](nm.Shape(6))
    for i in range(6):
        x.unsafe_set(2 * i, SIMD[dtype, 1](i + 1))
        x.unsafe_set(2 * i + 1, SIMD[dtype, 1](i % 2 + 1))
        labels.unsafe_set(i, SIMD[dtype, 1](i % 3))
        # Float64 labels must not be narrowed to the feature dtype.
        y.unsafe_set(i, SIMD[target_dtype, 1](16777216 + i % 3) if target_dtype == DType.float64 else SIMD[target_dtype, 1](i % 3))
    var actual = f_classif(x, y)
    var expected = f_classif(x, labels)
    var actual_chi = chi2(x, y)
    var expected_chi = chi2(x, labels)
    for j in range(2):
        assert_equal(actual.scores.unsafe_get(j), expected.scores.unsafe_get(j))
        assert_equal(actual_chi.scores.unsafe_get(j), expected_chi.scores.unsafe_get(j))
    var indicators = nm.zeros[target_dtype](nm.Shape(6, 3))
    for i in range(6):
        indicators.unsafe_set(3 * i + i % 3, 1)
    var multilabel = chi2(x, indicators)
    for j in range(2):
        assert_equal(multilabel.scores.unsafe_get(j), expected_chi.scores.unsafe_get(j))
    comptime if target_dtype == DType.float64:
        var precise_y = nm.zeros[DType.float64](nm.Shape(6))
        for i in range(6):
            precise_y.unsafe_set(i, 1 + Float64(i % 3) / 268435456)
        var correlation = r_regression(x, precise_y)
        var expected_correlation = r_regression(x, labels)
        for j in range(2):
            assert_equal(correlation.unsafe_get(j), expected_correlation.unsafe_get(j))
    var selector = SelectKBest(BuiltinScorer[dtype, target_dtype]("chi2"), k=SelectionParameter(1))
    _ = selector.fit(x, y.copy())
    var duplicate = selector.__sklearn_clone__()
    assert_true(not duplicate.__sklearn_is_fitted__())
    _ = duplicate.fit(x, y.copy())
    assert_equal(duplicate.get_support(True)[List[Int]][0], selector.get_support(True)[List[Int]][0])
    var transformed: nm.NDArray[dtype] = selector.transform(x)
    assert_equal(transformed.shape[1], 1)
    var bad_y = nm.zeros[target_dtype](nm.Shape(5))
    with assert_raises():
        _ = r_regression(x, bad_y)
    y.unsafe_set(0, nan[target_dtype]())
    with assert_raises():
        _ = f_regression(x, y)
    with assert_raises():
        _ = selector.fit(x, y.copy())
    assert_equal(selector.scores_().unsafe_get(0), expected_chi.scores.unsafe_get(0))

def main() raises:
    check_mixed_targets[DType.float32, DType.float64]()
    check_mixed_targets[DType.float64, DType.float32]()
    check_strided_scores[DType.float32]()
    check_strided_scores[DType.float64]()
    var x = nm.ones[DType.float64](nm.Shape(4, 2))
    var y = nm.zeros[DType.float64](nm.Shape(4))
    y.unsafe_set(2, 1)
    y.unsafe_set(3, 1)
    assert_equal(r_regression(x, y).unsafe_get(0), 0)
    assert_true(isnan(r_regression(x, y, force_finite=False).unsafe_get(0)))
    assert_true(isnan(f_classif(x, y).scores.unsafe_get(0)))
    assert_equal(f_regression(x, y).pvalues.value().unsafe_get(0), 1)
    x.unsafe_set(0, -1)
    with assert_raises():
        _ = chi2(x, y)
    x.unsafe_set(0, nan[DType.float64]())
    with assert_raises():
        _ = f_classif(x, y)
    x.unsafe_set(0, inf[DType.float64]())
    with assert_raises():
        _ = f_regression(x, y)
    x.unsafe_set(0, 1)
    var wrong_y = nm.ones[DType.float64](nm.Shape(3))
    with assert_raises():
        _ = r_regression(x, wrong_y)
    var empty = nm.ones[DType.float64](nm.Shape(0, 2))
    with assert_raises():
        _ = f_classif(empty, y)
    var groups = List[nm.NDArray[DType.float64]]()
    groups.append(nm.ones[DType.float64](nm.Shape(3, 2)))
    groups.append(nm.ones[DType.float64](nm.Shape(2, 2)))
    assert_true(isnan(f_oneway(groups).scores.unsafe_get(0)))
    var indicators = nm.zeros[DType.float64](nm.Shape(4, 2))
    indicators.unsafe_set(0, 1)
    indicators.unsafe_set(3, 1)
    indicators.unsafe_set(4, 1)
    indicators.unsafe_set(7, 1)
    assert_equal(chi2(x, indicators).scores.unsafe_get(0), 0)
    indicators.unsafe_set(0, 2)
    with assert_raises():
        _ = chi2(x, indicators)
    print("score edge contracts passed")
