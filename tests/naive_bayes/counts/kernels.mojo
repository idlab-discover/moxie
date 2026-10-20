import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie._naive_bayes_dense import CountStatistics, FeatureAlpha, prepare_batch
from moxie._nb_counts import CountsUpdater, CountsFinalizer, MultinomialNB, ComplementNB

def test_alpha_layouts[complement: Bool](norm: Bool) raises:
    var counts = CountStatistics(
        nm.ones[DType.float64](nm.Shape(2)),
        nm.ones[DType.float64](nm.Shape(2, 3)),
    )
    counts.feature_count[nm.Item(0, 1)] = 4
    counts.feature_count[nm.Item(1, 2)] = 7
    var contiguous = nm.zeros[DType.float64](nm.Shape(3))
    contiguous[nm.Item(0)] = 0
    contiguous[nm.Item(1)] = 0.25
    contiguous[nm.Item(2)] = 2
    var expected_alpha = FeatureAlpha(1, contiguous.copy())
    for stride in range(-2, 3, 4):
        var backing = nm.full[DType.float64](nm.Shape(8), 9)
        var offset = 1 if stride > 0 else 5
        var view = backing.view_with_layout(nm.Shape(3), nm.NDArrayStrides(stride), offset)
        for f in range(3):
            view[nm.Item(f)] = contiguous[nm.Item(f)]
        var alpha = FeatureAlpha(1, view^)
        for force in range(2):
            var expected = CountsFinalizer[complement](norm).finalize(counts, expected_alpha, Bool(force))
            var actual = CountsFinalizer[complement](norm).finalize(counts, alpha, Bool(force))
            for c in range(2):
                for f in range(3):
                    assert_equal(actual[nm.Item(c, f)], expected[nm.Item(c, f)])

def main() raises:
    test_alpha_layouts[False](False)
    test_alpha_layouts[True](False)
    test_alpha_layouts[True](True)
    var X = nm.ones[DType.float64](nm.Shape(2, 2))
    X[nm.Item(0, 1)] = 0
    var y: List[Int] = [0, 1]
    var empty = List[Int]()
    var batch = prepare_batch(X, y, empty, y.copy()).batch.copy()
    var counts = CountStatistics(nm.zeros[DType.float64](nm.Shape(2)), nm.zeros[DType.float64](nm.Shape(2, 2)))
    CountsUpdater().accumulate(X, batch, counts)
    assert_equal(counts.class_count.item(0), 1)
    assert_equal(counts.feature_count.item(0, 1), 0)
    var vector = nm.ones[DType.float64](nm.Shape(2))
    vector[nm.Item(1)] = 2
    var alpha = FeatureAlpha(1.0, vector^)
    var logp = CountsFinalizer[False](False).finalize(counts, alpha, True)
    assert_true(abs(logp.item(0, 0) + 0.6931471805599453) < 1e-7)
    assert_true(abs(logp.item(1, 0) + 0.916290731874155) < 1e-7)
    var cn = CountsFinalizer[True](True).finalize(counts, alpha, True)
    assert_true(abs(cn.item(0, 0) + cn.item(0, 1) - 1) < 1e-7)
    var copy = counts.copy()
    CountsUpdater().accumulate(X, batch, counts)
    assert_equal(copy.class_count.item(0), 1)
    assert_equal(counts.class_count.item(0), 2)
    var m = MultinomialNB(alpha_values=alpha.values.copy())
    _ = m.fit(X, y)
    assert_equal(m.predict(X)[1], 0)
    var probabilities = m.predict_proba(X)
    assert_true(abs(probabilities[0][0] - 5.0 / 9.0) < 1e-7)
    var cloned = m.__sklearn_clone__()
    assert_equal(cloned.__sklearn_is_fitted__(), False)
    with assert_raises():
        _ = cloned.predict(X)
    var fitted_copy = m.copy()
    m.feature_count_[nm.Item(0, 0)] = 99
    assert_equal(fitted_copy.feature_count_.item(0, 0), 1)
    _ = m.fit(X, y)
    assert_equal(m.feature_count_.item(0, 0), 1)
    var c = ComplementNB(norm=True)
    _ = c.fit(X, y)
    assert_equal(c.predict(X)[0], 0)
    var f32 = nm.ones[DType.float32](nm.Shape(2, 2))
    f32[nm.Item(0, 1)] = 0
    var inc = MultinomialNB(alpha_values=alpha.values.copy())
    _ = inc.partial_fit(f32, y, y.copy())
    assert_equal(inc.feature_count_.item(0, 1), 0)
    _ = inc.partial_fit(f32, y)
    assert_equal(inc.class_count_.item(0), 2)
    f32[nm.Item(0, 0)] = -1
    with assert_raises():
        _ = inc.partial_fit(f32, y)
    assert_equal(inc.class_count_.item(0), 2)
    var params = m.get_params()
    params.alpha_values.value()[nm.Item(0)] = 100
    assert_equal(m.alpha_values.value().item(0), 1)
    var alpha_before = m.alpha_values.value().item(0)
    var copied_model = m.copy()
    copied_model.alpha_values.value()[nm.Item(0)] = 7
    assert_equal(m.alpha_values.value().item(0), alpha_before)
    _ = m.set_params(alpha=0.25)
    assert_equal(Bool(m.alpha_values), False)
    _ = m.fit(X, y)
    var weights: List[Float64] = [2, 0]
    _ = m.fit(X, y, sample_weight=weights.copy())
    assert_equal(m.class_count_.item(1), 0)
    var zero: List[Float64] = [0, 0]
    with assert_raises():
        _ = m.partial_fit(X, y, sample_weight=zero.copy())
    assert_equal(m.class_count_.item(0), 2)
    var bad: List[Int] = [0, 99]
    with assert_raises():
        _ = m.partial_fit(X, bad)
    var short: List[Int] = [0]
    with assert_raises():
        _ = m.fit(X, short)
    assert_equal(m.class_count_.item(0), 2)
    var prior: List[Float64] = [1]
    var wrong_prior = MultinomialNB(class_prior=prior^)
    with assert_raises():
        _ = wrong_prior.fit(X, y)
    var first = MultinomialNB()
    with assert_raises():
        _ = first.partial_fit(X, y)
    var labels: List[Int] = [9, 0, -3]
    with assert_raises():
        _ = first.partial_fit(X, short.copy(), labels.copy())
    var observed: List[Int] = [0, 9]
    _ = first.partial_fit(X, observed, labels.copy())
    assert_equal(first.classes_[0], -3)
    assert_equal(first.class_count_.item(0), 0)
    var backwards = X.view_with_layout(nm.Shape(2, 2), nm.NDArrayStrides(2, -1), 1)
    var strided = MultinomialNB()
    _ = strided.fit(backwards, y)
    assert_equal(strided.feature_count_.item(0, 0), 0)
    assert_equal(strided.feature_count_.item(0, 1), 1)
    var bad_alpha = nm.ones[DType.float64](nm.Shape(3))
    var invalid = ComplementNB(alpha_values=bad_alpha^)
    with assert_raises():
        _ = invalid.fit(X, y)
    var list_X: List[List[Float64]] = [[1, 0], [1, 1]]
    var compatibility = MultinomialNB()
    _ = compatibility.fit(list_X, y)
    assert_equal(compatibility.feature_count_.item(0, 1), 0)
    assert_equal(compatibility.predict(list_X)[0], 0)
    print("PASS kernels and estimator lifecycle checks")
