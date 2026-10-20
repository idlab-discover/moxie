from std.testing import assert_equal, assert_true, assert_false, assert_raises, assert_almost_equal
import numojo as nm
from moxie._random import RandomState
from moxie.tree._classes import BaseDecisionTree, FeatureCount
from moxie.ensemble._records import Values, Indices, SamplingPlan, ClassAlignment
from moxie.ensemble._sampling import (
    MaxSamples, BootstrapSampler, BaggingSampler, numpy_sum, get_n_samples_bootstrap,
    generate_sample_indices, generate_unsampled_indices, sample_counts, class_weight_factors,
)
from moxie.ensemble._aggregation import ForestAggregator, new_oob, forest_oob_denominators, mean_feature_importances
from moxie.ensemble._forest_engine import BaseEnsemble, ForestEngine, ForestClassWeight

comptime Regressor = BaseDecisionTree[False]
comptime Classifier = BaseDecisionTree[True]

def data(n: Int) raises -> Tuple[nm.NDArray[DType.float32], Values, Values]:
    var X = nm.zeros[DType.float32](nm.Shape(n, 3))
    var y = nm.zeros[DType.float64](nm.Shape(n))
    var codes = nm.zeros[DType.float64](nm.Shape(n))
    var rng = RandomState(5)
    for i in range(n):
        for j in range(3):
            X.unsafe_set(i * 3 + j, Float32(rng.uniform(-1, 1)))
        y.unsafe_set(i, Float64(X.unsafe_get(i * 3)) * 2 + rng.uniform(0, 0.1))
        codes.unsafe_set(i, Float64(i % 3) if X.unsafe_get(i * 3 + 1) > 0 else Float64(0))
    return (X^, y^, codes^)

def same(a: Values, b: Values) raises:
    assert_equal(a.size, b.size)
    for i in range(a.size):
        assert_equal(a.unsafe_get(i), b.unsafe_get(i))

def base_ensemble() raises:
    var ensemble = BaseEnsemble[Regressor](Regressor(max_depth=3), 4)
    assert_equal(len(ensemble), 0)
    var member = ensemble.make_estimator()
    assert_false(member.__sklearn_is_fitted__())
    assert_equal(member.params.max_depth.value(), 3)
    ensemble.estimators_.append(member^)
    ensemble.estimators_.append(Regressor(max_depth=5))
    assert_equal(len(ensemble), 2)
    assert_equal(ensemble[-1].params.max_depth.value(), 5)
    with assert_raises():
        _ = ensemble[2].params.max_depth

def samplers() raises:
    var rng = RandomState(3)
    var forest = BootstrapSampler()
    var selection = forest.select(SamplingPlan(20, 4, 20, 4, True, False), None, rng)
    assert_equal(selection.rows.size, 20)
    assert_equal(selection.features.size, 4)
    var total = 0
    for i in range(20):
        total += Int(selection.multiplicities.unsafe_get(i))
    assert_equal(total, 20)
    var oob = forest.unsampled(selection)
    for r in range(oob.size):
        assert_equal(selection.multiplicities.unsafe_get(Int(oob.unsafe_get(r))), 0)
    # The trait sampler and the seed-replayable forest helper draw the same rows.
    var replay = RandomState(11)
    var rows = forest.select(SamplingPlan(20, 4, 20, 4, True, False), None, replay).rows.copy()
    var helper = generate_sample_indices(11, 20, 20, None)
    for i in range(20):
        assert_equal(rows.unsafe_get(i), helper.unsafe_get(i))
    with assert_raises():
        _ = forest.select(SamplingPlan(20, 4, 20, 2, True, False), None, rng)
    var bagging = BaggingSampler()
    var subset = bagging.select(SamplingPlan(30, 6, 10, 3, False, False), None, rng)
    assert_equal(subset.features.size, 3)
    for i in range(30):
        assert_true(subset.multiplicities.unsafe_get(i) <= 1)  # without replacement
    var mass = nm.full[DType.float64](nm.Shape(30), 1)
    mass.unsafe_set(0, 0)
    var weighted = bagging.select(SamplingPlan(30, 6, 29, 6, False, True), mass^, rng)
    assert_equal(weighted.multiplicities.unsafe_get(0), 0)  # zero mass is never drawn
    var negative = nm.full[DType.float64](nm.Shape(4), 1)
    negative.unsafe_set(2, -1)
    with assert_raises():
        _ = generate_sample_indices(1, 4, 4, negative^)

def bootstrap_counts() raises:
    var warnings = List[String]()
    assert_equal(get_n_samples_bootstrap(50, MaxSamples(NoneType()), None, warnings), 50)
    assert_equal(get_n_samples_bootstrap(50, MaxSamples(80), None, warnings), 80)
    var w = nm.full[DType.float64](nm.Shape(50), 2)
    assert_equal(get_n_samples_bootstrap(50, MaxSamples(Float64(0.5)), w^, warnings), 50)  # scales sum(w)=100
    assert_equal(len(warnings), 0)
    assert_equal(get_n_samples_bootstrap(50, MaxSamples(Float64(0.1)), None, warnings), 5)
    assert_equal(len(warnings), 1)
    with assert_raises():
        _ = get_n_samples_bootstrap(50, MaxSamples(0), None, warnings)
    with assert_raises():
        _ = get_n_samples_bootstrap(50, MaxSamples(Float64(0)), None, warnings)
    var x = nm.zeros[DType.float64](nm.Shape(9))
    for i in range(9):
        x.unsafe_set(i, 0.1)
    assert_equal(numpy_sum(x), 0.9)  # pairwise order; a sequential sum gives 0.8999999999999999

def aggregation() raises:
    var aggregator = ForestAggregator()
    var sums = nm.zeros[DType.float64](nm.Shape(2, 3))
    var proba = nm.full[DType.float64](nm.Shape(2, 2), 0.5)
    var columns = nm.zeros[DType.int64](nm.Shape(2))
    columns.unsafe_set(0, 2)
    aggregator.add_probability(proba, ClassAlignment(columns^, 3), 2, sums)
    assert_equal(sums.unsafe_get(2), 1.0)
    assert_equal(sums.unsafe_get(0), 1.0)
    assert_equal(sums.unsafe_get(1), 0.0)
    var state = new_oob(3, 1)
    var rows = nm.zeros[DType.int64](nm.Shape(1))
    rows.unsafe_set(0, 2)
    aggregator.add_oob(rows, nm.full[DType.float64](nm.Shape(1, 1), 4), state)
    var warnings = List[String]()
    var denominators = forest_oob_denominators(state.counts, warnings)
    assert_equal(len(warnings), 1)  # rows 0 and 1 have no OOB prediction
    var mean = aggregator.normalize(state.sums, denominators)
    assert_equal(mean.unsafe_get(0), 0.0)
    assert_equal(mean.unsafe_get(2), 4.0)
    var none = List[Values]()
    assert_equal(mean_feature_importances(none, 3).unsafe_get(1), 0.0)

def regression_engine() raises:
    var parts = data(60)
    ref X = parts[0]
    ref y = parts[1]
    var forest = ForestEngine[False](Regressor(max_features=FeatureCount(Float64(1))), 6, oob_score=True)
    with assert_raises():
        _ = forest.apply(X)
    var rng = RandomState(0)
    forest.fit(X, y, None, rng)
    assert_equal(len(forest), 6)
    assert_true(forest.has_oob())
    assert_true(forest.oob_score_ > 0.5)
    assert_false(forest.base.estimator.params.random_state.__bool__())  # configuration stays unseeded
    var leaves = forest.apply(X)
    assert_equal(leaves.shape[1], 6)
    var importances = forest.feature_importances_()
    assert_almost_equal(importances.unsafe_get(0) + importances.unsafe_get(1) + importances.unsafe_get(2), 1.0)
    var first = forest.predict(X)
    var again = RandomState(0)
    forest.fit(X, y, None, again)  # a normal fit replaces members deterministically
    same(first, forest.predict(X))
    # Warm-start growth equals a cold fit with the final size (seeds are replayed).
    var warm = ForestEngine[False](Regressor(max_features=FeatureCount(Float64(1))), 3, warm_start=True)
    var cold = ForestEngine[False](Regressor(max_features=FeatureCount(Float64(1))), 6)
    var r1 = RandomState(4)
    warm.fit(X, y, None, r1)
    warm.base.n_estimators = 6
    var r2 = RandomState(4)
    warm.fit(X, y, None, r2)
    var r3 = RandomState(4)
    cold.fit(X, y, None, r3)
    same(warm.predict(X), cold.predict(X))
    var r4 = RandomState(4)
    warm.fit(X, y, None, r4)
    assert_equal(len(warm), 6)
    assert_equal(len(warm.warnings_), 1)  # no growth
    warm.base.n_estimators = 2
    with assert_raises():
        warm.fit(X, y, None, r4)
    var no_bootstrap = ForestEngine[False](Regressor(), 2, bootstrap=False, max_samples=MaxSamples(10))
    with assert_raises():
        no_bootstrap.fit(X, y, None, r4)
    var no_oob = ForestEngine[False](Regressor(), 2, bootstrap=False, oob_score=True)
    with assert_raises():
        no_oob.fit(X, y, None, r4)
    var weighted = ForestEngine[False](Regressor(), 2, class_weight=ForestClassWeight(String("balanced")))
    with assert_raises():
        weighted.fit(X, y, None, r4)
    var bad = nm.full[DType.float64](nm.Shape(59), 1)
    with assert_raises():
        forest.fit(X, y, bad^, r4)

def classification_engine() raises:
    var parts = data(90)
    ref X = parts[0]
    ref codes = parts[2]
    var forest = ForestEngine[True](Classifier(max_features=FeatureCount(String("sqrt"))), 8, oob_score=True,
        class_weight=ForestClassWeight(String("balanced_subsample")))
    var rng = RandomState(2)
    forest.fit(X, codes, None, rng)
    assert_equal(forest.n_classes_[0], 3)
    var proba = forest.predict_proba(X)
    assert_equal(len(proba), 1)
    for i in range(90):
        assert_almost_equal(proba[0].unsafe_get(i * 3) + proba[0].unsafe_get(i * 3 + 1) + proba[0].unsafe_get(i * 3 + 2), 1.0)
    assert_equal(forest.oob_decision_function_[0].shape[1], 3)
    assert_equal(forest.base.estimators_[0].classes_[0].unsafe_get(2), 2.0)  # members see class codes
    var factors = List[Values]()
    var mapping = Dict[Float64, Float64]()
    mapping[2.0] = 4.0
    var classes = nm.zeros[DType.float64](nm.Shape(3))
    for c in range(3):
        classes.unsafe_set(c, Float64(c))
    factors.append(class_weight_factors(classes, mapping))
    assert_equal(factors[0].unsafe_get(0), 1.0)
    assert_equal(factors[0].unsafe_get(2), 4.0)
    var explicit = ForestEngine[True](Classifier(), 3, class_weight=ForestClassWeight(factors^))
    explicit.fit(X, codes, None, rng)
    var preset = ForestEngine[True](Classifier(), 3, class_weight=ForestClassWeight(String("balance")))
    with assert_raises():
        preset.fit(X, codes, None, rng)
    var unknown = Dict[Float64, Float64]()
    unknown[7.0] = 2.0
    unknown[1.0] = 2.0
    with assert_raises():
        _ = class_weight_factors(classes, unknown)

def main() raises:
    base_ensemble()
    samplers()
    bootstrap_counts()
    aggregation()
    regression_engine()
    classification_engine()
    print("PASS forest engine: BaseEnsemble, samplers, aggregation, OOB, warm start and validation")
