"""Public forest lifecycle, labels, shapes, defaults and failure contracts."""
import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.ensemble._forest import Forest
from moxie.tree._classes import SampleCount, FeatureCount, ClassWeight
from moxie.ensemble._sampling import MaxSamples
from moxie.base import clone, LogProbabilityPredictor, WeightedSupervisedEstimator
from moxie.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, ProbabilisticEnsemble, EnsembleRegressor, DenseForestInspection


def classifier_contract[E: ProbabilisticEnsemble & LogProbabilityPredictor & WeightedSupervisedEstimator & DenseForestInspection](model: E):
    pass


def regressor_contract[E: EnsembleRegressor & WeightedSupervisedEstimator & DenseForestInspection](model: E):
    pass


def parameter_updates[classifier: Bool, extra: Bool]() raises:
    var model = Forest[classifier, extra]()
    var constraints: List[Int] = [1, 0]
    _ = model.set_params(
        n_estimators=7, criterion="entropy" if classifier else "absolute_error",
        max_depth=3, min_samples_split=SampleCount(Float64(0.25)),
        min_samples_leaf=SampleCount(2), min_weight_fraction_leaf=0.1,
        max_features=FeatureCount("sqrt"), max_leaf_nodes=5,
        min_impurity_decrease=0.01, bootstrap=True, oob_score=True,
        n_jobs=1, random_state=0, verbose=1, warm_start=True,
        class_weight=ClassWeight("balanced"), ccp_alpha=0.02,
        max_samples=MaxSamples(Float64(0.75)), monotonic_cst=constraints.copy(),
    )
    var config = model.get_params()
    assert_equal(config.n_estimators, 7)
    assert_equal(config.criterion, "entropy" if classifier else "absolute_error")
    assert_equal(config.max_depth.value(), 3)
    assert_equal(config.min_samples_split[Float64], Float64(0.25))
    assert_equal(config.min_samples_leaf[Int], 2)
    assert_equal(config.min_weight_fraction_leaf, Float64(0.1))
    assert_equal(config.max_features[String], "sqrt")
    assert_equal(config.max_leaf_nodes.value(), 5)
    assert_equal(config.min_impurity_decrease, Float64(0.01))
    assert_true(config.bootstrap and config.oob_score and config.warm_start)
    assert_equal(config.n_jobs.value(), 1)
    assert_equal(config.random_state.value(), 0)
    assert_equal(config.verbose, 1)
    assert_equal(config.class_weight[String], "balanced")
    assert_equal(config.ccp_alpha, Float64(0.02))
    assert_equal(config.max_samples[Float64], Float64(0.75))
    constraints[0] = -1
    assert_equal(config.monotonic_cst.value()[0], 1)
    var cloned = clone(model)
    config.monotonic_cst.value()[0] = 0
    assert_equal(model.monotonic_cst.value()[0], 1)
    assert_equal(cloned.monotonic_cst.value()[0], 1)
    _ = model.set_params()
    _ = model.set_params(n_estimators=4)
    assert_equal(model.max_depth.value(), 3)
    assert_equal(model.random_state.value(), 0)
    assert_equal(model.monotonic_cst.value()[0], 1)
    _ = model.set_params(
        max_depth=None, max_leaf_nodes=None, n_jobs=None, random_state=None,
        monotonic_cst=None, max_features=FeatureCount(NoneType()),
        class_weight=ClassWeight(NoneType()), max_samples=MaxSamples(NoneType()),
        verbose=0, warm_start=False, oob_score=False, bootstrap=False,
        min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0, ccp_alpha=0.0,
    )
    assert_true(not model.max_depth and not model.max_leaf_nodes)
    assert_true(not model.n_jobs and not model.random_state and not model.monotonic_cst)
    assert_true(model.max_features.isa[NoneType]())
    assert_true(model.class_weight.isa[NoneType]())
    assert_true(model.max_samples.isa[NoneType]())
    assert_true(not model.warm_start and not model.oob_score and not model.bootstrap)
    assert_equal(model.verbose, 0)
    assert_equal(model.min_weight_fraction_leaf, Float64(0))
    assert_equal(model.min_impurity_decrease, Float64(0))
    assert_equal(model.ccp_alpha, Float64(0))
    var X = nm.zeros[DType.float32](nm.Shape(12, 2))
    var y = nm.zeros[DType.float64](nm.Shape(12))
    for i in range(12):
        X.unsafe_set(2 * i, Float32(i))
        X.unsafe_set(2 * i + 1, Float32(i % 3))
        y.unsafe_set(i, Float64(i % 2))
    _ = model.set_params(random_state=42, min_samples_split=SampleCount(2),
        min_samples_leaf=SampleCount(1)).fit(X, y)
    var before = model.apply(X)
    _ = model.set_params(max_depth=1, min_samples_leaf=SampleCount(3))
    assert_true(model.__sklearn_is_fitted__())
    var unchanged = model.apply(X)
    for i in range(before.size):
        assert_equal(before.unsafe_get(i), unchanged.unsafe_get(i))
    var cold = Forest[classifier, extra](4,
        criterion="entropy" if classifier else "absolute_error",
        max_depth=1, min_samples_leaf=3, max_features=NoneType(),
        bootstrap=False, random_state=42)
    _ = cold.fit(X, y)
    _ = model.fit(X, y)
    var refitted = model.apply(X)
    var expected = cold.apply(X)
    for i in range(expected.size):
        assert_equal(refitted.unsafe_get(i), expected.unsafe_get(i))
    _ = model.set_params(max_depth=0)
    with assert_raises():
        _ = model.fit(X, y)


def main() raises:
    parameter_updates[True, False]()
    parameter_updates[False, False]()
    parameter_updates[True, True]()
    parameter_updates[False, True]()
    var X = nm.zeros[DType.float32](nm.Shape(12, 2))
    var y = nm.zeros[DType.float64](nm.Shape(12))
    for i in range(12):
        X.unsafe_set(2 * i, Float32(i))
        X.unsafe_set(2 * i + 1, Float32(i % 3))
        y.unsafe_set(i, Float64(-3 if i < 6 else 7))
    var classifier = RandomForestClassifier(12, random_state=42, bootstrap=False)
    with assert_raises():
        _ = classifier.predict(X)
    classifier_contract(classifier)
    _ = classifier.fit(X, y)
    assert_equal(classifier.n_classes_, 2)
    assert_equal(classifier.classes_.unsafe_get(0), Float64(-3))
    assert_equal(classifier.classes_.unsafe_get(1), Float64(7))
    var predicted = classifier.predict(X)
    var probabilities = classifier.predict_proba(X)
    var logs = classifier.predict_log_proba(X)
    assert_equal(predicted.ndim, 1)
    assert_equal(probabilities.shape[1], 2)
    assert_equal(logs.shape[1], 2)
    assert_equal(classifier.score(X, y), Float64(1))
    assert_equal(len(classifier), 12)
    assert_equal(classifier.estimators_()[0].classes_[0].unsafe_get(0), Float64(0))
    assert_true(classifier[-1].__sklearn_is_fitted__())
    var cloned = clone(classifier)
    assert_true(not cloned.__sklearn_is_fitted__())
    assert_equal(cloned.n_estimators, 12)
    _ = classifier.set_params(warm_start=True, n_estimators=16)
    _ = classifier.fit(X, y)
    var cold = RandomForestClassifier(16, random_state=42, bootstrap=False)
    _ = cold.fit(X, y)
    var grown = classifier.apply(X)
    var cold_leaves = cold.apply(X)
    for i in range(grown.size):
        assert_equal(grown.unsafe_get(i), cold_leaves.unsafe_get(i))
    classifier.n_estimators = 15
    with assert_raises():
        _ = classifier.fit(X, y)
    var params = classifier.get_params()
    params.n_estimators = 3
    params.warm_start = False
    _ = classifier.set_params(params).fit(X, y)
    assert_equal(len(classifier), 3)
    var extra = ExtraTreesClassifier(10, random_state=42)
    assert_true(not extra.bootstrap)
    _ = extra.fit(X, y)
    assert_equal(extra.score(X, y), Float64(1))
    var regressor = RandomForestRegressor(20, random_state=42, oob_score=True)
    regressor_contract(regressor)
    _ = regressor.fit(X, y)
    assert_equal(regressor.predict(X).ndim, 1)
    assert_equal(regressor.oob_prediction_.ndim, 1)
    assert_equal(regressor.estimators_samples_()[0].size, 12)
    var extra_regressor = ExtraTreesRegressor(10, random_state=42)
    _ = extra_regressor.fit(X, y)
    assert_equal(extra_regressor.score(X, y), Float64(1))
    var multiple = nm.zeros[DType.float64](nm.Shape(12, 2))
    for i in range(12):
        multiple.unsafe_set(i * 2, y.unsafe_get(i))
        multiple.unsafe_set(i * 2 + 1, Float64(i % 3))
    var multi = RandomForestClassifier[multi_output=True](8, bootstrap=False, random_state=12)
    _ = multi.fit(X, multiple)
    var multi_proba = multi.predict_proba(X)
    assert_equal(len(multi_proba), 2)
    assert_equal(multi.n_classes_[1], 3)
    assert_equal(multi.predict(X).shape[1], 2)
    _ = extra_regressor.fit(X, multiple)
    assert_equal(extra_regressor.predict(X).shape[1], 2)
    with assert_raises():
        _ = classifier.fit(X, multiple)
    # joblib rejects n_jobs=0; any other value only changes worker scheduling,
    # so serially built members equal the default ones.
    var invalid = RandomForestRegressor(n_jobs=0)
    with assert_raises(contains="n_jobs == 0 in Parallel has no meaning"):
        _ = invalid.fit(X, y)
    var serial = RandomForestRegressor(5, random_state=3)
    _ = serial.fit(X, y)
    var job_counts: List[Int] = [2, -1]
    for jobs in job_counts:
        var workers = RandomForestRegressor(5, random_state=3, n_jobs=Int(jobs))
        _ = workers.fit(X, y)
        var a = serial.predict(X)
        var b = workers.predict(X)
        for i in range(a.size):
            assert_equal(a.unsafe_get(i), b.unsafe_get(i))
    var weights = nm.full[DType.float64](nm.Shape(12), 2)
    _ = classifier.fit(X, y, sample_weight=weights)
    assert_equal(classifier.score(X, y, weights.copy()), Float64(1))
    with assert_raises():
        _ = classifier.predict(nm.zeros[DType.float32](nm.Shape(2, 3)))
    with assert_raises():
        _ = classifier.fit(X, nm.zeros[DType.float64](nm.Shape(3)))
    var continuous = y.copy()
    continuous.unsafe_set(0, 0.5)
    with assert_raises():
        _ = classifier.fit(X, continuous)
    var no_criterion = RandomForestClassifier(criterion="")
    with assert_raises():
        _ = no_criterion.fit(X, y)
    var zero_trees = RandomForestRegressor(0)
    with assert_raises():
        _ = zero_trees.fit(X, y)
    var no_bootstrap = ExtraTreesRegressor(oob_score=True)
    with assert_raises():
        _ = no_bootstrap.fit(X, y)
    var constant = nm.full[DType.float64](nm.Shape(12), 7)
    _ = classifier.fit(X, constant)
    assert_equal(classifier.n_classes_, 1)
    assert_equal(classifier.predict_proba(X).shape[1], 1)
    assert_equal(classifier.predict(X).unsafe_get(0), Float64(7))
    var one_class_log = classifier.predict_log_proba(X)
    assert_equal(one_class_log.unsafe_get(0), Float64(0))
    var config = classifier.get_params()
    var constraints: List[Int] = [1, 0]
    config.monotonic_cst = constraints^
    _ = classifier.set_params(config)
    var independent = clone(classifier)
    config.monotonic_cst.value()[0] = -1
    assert_equal(independent.monotonic_cst.value()[0], 1)
    print("PASS: public forest estimators")
