"""MI API, ownership, RNG and selector contracts beyond numerical parity."""
import numojo as nm
from std.math import abs, nan, inf
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie._random import RandomState, seed_global_random_state
from moxie.feature_selection import mutual_info_classif, mutual_info_regression, DiscreteFeatures, MutualInfoScorer, SelectKBest, SelectPercentile, SelectFpr


def equal(a: nm.NDArray[DType.float64], b: nm.NDArray[DType.float64]) raises:
    assert_equal(a.shape[0], b.shape[0])
    for i in range(a.shape[0]):
        assert_equal(a.unsafe_get(i), b.unsafe_get(i))


def contracts[dt: DType]() raises:
    var x = nm.zeros[dt](nm.Shape(16, 3))
    var y = nm.zeros[dt](nm.Shape(16))
    for i in range(16):
        x.unsafe_set(i * 3, SIMD[dt, 1](i % 2))
        x.unsafe_set(i * 3 + 1, SIMD[dt, 1]((i // 2) % 2))
        x.unsafe_set(i * 3 + 2, 1)
        y.unsafe_set(i, SIMD[dt, 1](i % 2))
    var saved = x.copy()
    var first = mutual_info_classif(x, y, discrete_features=DiscreteFeatures(True), random_state=7)
    assert_true(abs(first.unsafe_get(0) - 0.6931471805599453) < 1e-14)
    assert_equal(first.unsafe_get(1), Float64(0))
    assert_equal(first.unsafe_get(2), Float64(0))
    for i in range(48):
        assert_equal(x.unsafe_get(i), saved.unsafe_get(i))
    var auto_scores = mutual_info_classif(x, y, random_state=7)
    equal(auto_scores, mutual_info_classif(x, y, discrete_features=DiscreteFeatures(False), random_state=7, n_jobs=-1))
    var indices: List[Int] = [-3, 2]
    var mask: List[Bool] = [True, False, True]
    equal(mutual_info_classif(x, y, discrete_features=DiscreteFeatures(indices^), random_state=7), mutual_info_classif(x, y, discrete_features=DiscreteFeatures(mask^), random_state=7))
    var state = RandomState(7)
    equal(auto_scores, mutual_info_classif(x, y, random_state=state))
    var advanced = mutual_info_classif(x, y, random_state=state)
    var replay = RandomState(7)
    _ = mutual_info_classif(x, y, random_state=replay)
    equal(advanced, mutual_info_classif(x, y, random_state=replay))
    seed_global_random_state(7)
    equal(auto_scores, mutual_info_classif(x, y))
    seed_global_random_state(7)
    equal(auto_scores, mutual_info_classif(x, y, random_state=None))
    var discrete_rng = RandomState(7)
    _ = mutual_info_classif(x, y, discrete_features=DiscreteFeatures(True), random_state=discrete_rng)
    var fresh = RandomState(7)
    assert_equal(discrete_rng.next_uint32(), fresh.next_uint32())
    var owned_mask: List[Bool] = [True, True, True]
    var tagged = DiscreteFeatures(owned_mask^)
    var owned_config = MutualInfoScorer[dt](discrete_features=tagged.copy())
    tagged[List[Bool]][0] = False
    assert_true(owned_config.discrete_features[List[Bool]][0])
    var independent = owned_config.clone_config()
    owned_config.discrete_features[List[Bool]][0] = False
    assert_true(independent.discrete_features[List[Bool]][0])
    var config = MutualInfoScorer[dt](discrete_features=DiscreteFeatures(True), random_state=7)
    var best = SelectKBest(config.copy(), k=1)
    with assert_raises():
        _ = best.scores_()
    _ = best.fit(x, y.copy())
    assert_false(Bool(best.pvalues_()))
    assert_true(best.get_support()[List[Bool]][0])
    var transformed = best.transform(x)
    assert_equal(transformed.shape[1], 1)
    var inverse = best.inverse_transform(transformed)
    for i in range(16):
        assert_equal(inverse.unsafe_get(i * 3), x.unsafe_get(i * 3))
        assert_equal(inverse.unsafe_get(i * 3 + 1), SIMD[dt, 1](0))
    var clone = best.__sklearn_clone__()
    assert_false(clone.__sklearn_is_fitted__())
    _ = clone.fit(x, y.copy())
    equal(clone.scores_(), best.scores_())
    _ = best.fit(x, y.copy())
    equal(best.scores_(), first)
    var percentile = SelectPercentile(config.copy(), percentile=50)
    _ = percentile.fit(x, y.copy())
    assert_true(percentile.get_support()[List[Bool]][0])
    var regression = SelectKBest(MutualInfoScorer[dt](classification=False, random_state=7), k=1)
    _ = regression.fit(x, y.copy())
    assert_false(Bool(regression.pvalues_()))
    var pvalue_selector = SelectFpr(config.copy())
    _ = pvalue_selector.fit(x, y.copy())
    with assert_raises():
        _ = pvalue_selector.get_support()
    with assert_raises():
        _ = config.score(x, None)
    with assert_raises():
        _ = mutual_info_classif(x, y, n_neighbors=0)
    with assert_raises():
        _ = mutual_info_regression(x, y, n_neighbors=16)
    with assert_raises():
        _ = mutual_info_classif(x, y, n_jobs=0)
    with assert_raises():
        _ = mutual_info_classif(x, y, random_state=-1)
    with assert_raises():
        _ = mutual_info_classif(x, y, discrete_features=DiscreteFeatures(String("bad")))
    with assert_raises():
        var empty_indices = List[Int]()
        _ = mutual_info_classif(x, y, discrete_features=DiscreteFeatures(empty_indices^))
    with assert_raises():
        var bad: List[Int] = [3]
        _ = mutual_info_classif(x, y, discrete_features=DiscreteFeatures(bad^))
    with assert_raises():
        var bad: List[Bool] = [True]
        _ = mutual_info_classif(x, y, discrete_features=DiscreteFeatures(bad^))
    with assert_raises():
        var short = nm.zeros[dt](nm.Shape(15))
        _ = mutual_info_classif(x, short)
    with assert_raises():
        var multi = nm.zeros[dt](nm.Shape(16, 2))
        _ = mutual_info_classif(x, multi)
    with assert_raises():
        var empty = nm.zeros[dt](nm.Shape(0, 3))
        _ = mutual_info_classif(empty, y)
    var column_target = y.reshape(nm.Shape(16, 1))
    equal(auto_scores, mutual_info_classif(x, column_target, random_state=7))
    y.unsafe_set(0, 0.5)
    with assert_raises():
        _ = mutual_info_classif(x, y)
    y.unsafe_set(0, 0)
    x.unsafe_set(0, nan[dt]())
    with assert_raises():
        _ = mutual_info_classif(x, y)
    x.unsafe_set(0, 0)
    y.unsafe_set(0, inf[dt]())
    with assert_raises():
        _ = mutual_info_regression(x, y)
    for i in range(16):
        y.unsafe_set(i, SIMD[dt, 1](i))
    with assert_raises():
        _ = mutual_info_classif(x, y)
    # copy=False changes only Float64 continuous columns; Float32 is upcast.
    var before = x.copy()
    _ = mutual_info_regression(x, y, random_state=7, copy=False)
    comptime if dt == DType.float64:
        assert_true(x.unsafe_get(0) != before.unsafe_get(0))
    else:
        for i in range(48):
            assert_equal(x.unsafe_get(i), before.unsafe_get(i))


def main() raises:
    contracts[DType.float32]()
    contracts[DType.float64]()
    print("PASS mutual information contracts")
