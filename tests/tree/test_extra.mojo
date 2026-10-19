import numojo as nm
from std.testing import assert_true, assert_equal, assert_raises
from moxie.tree import ExtraTreeClassifier, ExtraTreeRegressor, DenseDecisionTreeClassifierAPI, DenseDecisionTreeRegressorAPI
from moxie.tree._classes import ClassWeightMap
from moxie.base import ClassifierMixin, RegressorMixin, ProbabilityPredictor, WeightedSupervisedEstimator, clone

def main() raises:
    comptime assert conforms_to(ExtraTreeClassifier[], DenseDecisionTreeClassifierAPI)
    comptime assert conforms_to(ExtraTreeRegressor, DenseDecisionTreeRegressorAPI)
    comptime assert conforms_to(ExtraTreeClassifier[], WeightedSupervisedEstimator)
    comptime assert not conforms_to(ExtraTreeRegressor, ProbabilityPredictor)
    comptime assert not conforms_to(ExtraTreeClassifier[], RegressorMixin)
    comptime assert not conforms_to(ExtraTreeRegressor, ClassifierMixin)
    var c = ExtraTreeClassifier(random_state=42)
    var r = ExtraTreeRegressor(random_state=42)
    assert_equal(c.params.splitter, "random")
    assert_equal(c.params.max_features[String], "sqrt")
    assert_equal(r.params.max_features[Float64], 1.0)
    var X = nm.zeros[DType.float32](nm.Shape(12, 4))
    var y = nm.zeros[DType.float64](nm.Shape(12))
    for i in range(12):
        y.unsafe_set(i, Float64(i // 6))
        for j in range(4):
            X.unsafe_set(i * 4 + j, Float32((i * 7 + j) % 19))
    var weights = nm.full[DType.float64](nm.Shape(12), 1)
    _ = c.fit(X, y, sample_weight=weights)
    assert_equal(c.max_features_, 2)
    assert_equal(c.score(X, y), 1.0)
    var copy = clone(c)
    _ = copy.fit(X, y)
    var a = c.apply(X)
    var b = copy.apply(X)
    for i in range(12):
        assert_equal(a.unsafe_get(i), b.unsafe_get(i))
    var mapped = ClassWeightMap()
    mapped[Float64(0)] = 2
    mapped[Float64(1)] = 3
    c.params.class_weight = mapped^
    _ = c.fit(X, y)
    assert_equal(c.tree_.nodes[0].weighted_n_node_samples, 30.0)
    c.params.ccp_alpha = 1
    _ = c.fit(X, y)
    assert_equal(c.get_n_leaves(), 1)
    c.params.max_features = NoneType()
    _ = c.fit(X, y)
    assert_equal(c.max_features_, 4)
    _ = r.fit(X, y)
    assert_equal(r.max_features_, 4)
    assert_equal(r.score(X, y), 1.0)
    print("PASS ExtraTree defaults, weighted trait, class map, deterministic cloning and pruning")
